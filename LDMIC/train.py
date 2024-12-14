import argparse
import math
import random
import sys
import time

from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

import yaml
import wandb
import os
from tqdm import tqdm
from pytorch_msssim import ms_ssim
from compressai.zoo.pretrained import load_pretrained

from torch.hub import load_state_dict_from_url
from lib.utils import CropCityscapesArtefacts, MinimalCrop
import re

import compressai
from compressai.zoo.pretrained import load_pretrained
from torch.hub import load_state_dict_from_url
import re
from compressai.models.utils import update_registered_buffers, conv, deconv
from compressai.entropy_models import GaussianConditional
from compressai.models import CompressionModel, get_scale_table
from compressai.ops import ste_round
from compressai.layers import ResidualBlock, GDN, MaskedConv2d, conv3x3, ResidualBlockWithStride
from deepspeed.profiling.flops_profiler import get_model_profile
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.models.utils import update_registered_buffers, conv, deconv
from compressai.entropy_models import GaussianConditional
from compressai.models import CompressionModel, get_scale_table
from compressai.ops import ste_round
from compressai.layers import ResidualBlock, GDN, MaskedConv2d, conv3x3, ResidualBlockWithStride
import torch.nn.functional as F
import copy

os.environ["WANDB_API_KEY"] = "8b245c1d70846a5a5cfaf55ddc91bf43489d457d" # write your own wandb id


class StereoImageDataset(Dataset):
    """Dataset class for image compression datasets."""
    #/home/xzhangga/datasets/Instereo2K/train/
    def __init__(self, ds_type='train', ds_name='cityscapes', root='/home/xzhangga/datasets/Cityscapes/', crop_size=(256, 256), resize=False, **kwargs):
        """
        Args:
            name (str): name of dataset, template: ds_name#ds_type. No '#' in ds_name or ds_type allowed. ds_type in (train, eval, test).
            path (str): if given the dataset is loaded from path instead of by name.
            transforms (Transform): transforms to apply to image
            debug (bool, optional): If set to true, limits the list of files to 10. Defaults to False.
        """
        super().__init__()
        
        self.path = Path(f"{root}")
        self.ds_name = ds_name
        self.ds_type = ds_type
        if ds_type=="train":
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(crop_size), 
                transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),]) 
        else: 
            self.transform = transforms.Compose([transforms.ToTensor()])
        self.left_image_list, self.right_image_list = self.get_files()


        if ds_name == 'cityscapes':
            self.crop = CropCityscapesArtefacts()
        else:
            if ds_type == "test":
                self.crop = MinimalCrop(min_div=64)
            else:
                self.crop = None
        #self.index_count = 0

        print(f'Loaded dataset {ds_name} from {self.path}. Found {len(self.left_image_list)} files.')

    def __len__(self):
        return len(self.left_image_list)

    def __getitem__(self, index):
        #self.index_count += 1
        image_list = [Image.open(self.left_image_list[index]).convert('RGB'), Image.open(self.right_image_list[index]).convert('RGB')]
        if self.crop is not None:
            image_list = [self.crop(image) for image in image_list]
        frames = np.concatenate([np.asarray(image) for image in image_list], axis=-1)
        frames = torch.chunk(self.transform(frames), 2)
        if random.random() < 0.5:
            frames = frames[::-1]
        return frames

    def get_files(self):
        if self.ds_name == 'cityscapes':
            left_image_list, right_image_list, disparity_list = [], [], []
            for left_image_path in self.path.glob(f'leftImg8bit/{self.ds_type}/*/*.png'):
                left_image_list.append(str(left_image_path))
                right_image_list.append(str(left_image_path).replace("leftImg8bit", 'rightImg8bit'))
                disparity_list.append(str(left_image_path).replace("leftImg8bit", 'disparity'))

        elif self.ds_name == 'instereo2k':
            path = self.path / self.ds_type
            if self.ds_type == "test":
                folders = [f for f in path.iterdir() if f.is_dir()]
            else:
                folders = [f for f in path.glob('*/*') if f.is_dir()]
            left_image_list = [f / 'left.png' for f in folders]
            right_image_list = [f / 'right.png' for f in folders]

        elif self.ds_name == 'kitti':
            left_image_list, right_image_list = [], []
            ds_type = self.ds_type + "ing"
            for left_image_path in self.path.glob(f'stereo2012/{ds_type}/colored_0/*.png'):
                left_image_list.append(str(left_image_path))
                right_image_list.append(str(left_image_path).replace("colored_0", 'colored_1'))

            for left_image_path in self.path.glob(f'stereo2015/{ds_type}/image_2/*.png'):
                left_image_list.append(str(left_image_path))
                right_image_list.append(str(left_image_path).replace("image_2", 'image_3'))

        elif self.ds_name == 'wildtrack':
            C1_image_list, C4_image_list = [], []
            for image_path in self.path.glob(f'images/C1/*.png'):
                if self.ds_type == "train" and int(image_path.stem) <= 2000:
                    C1_image_list.append(str(image_path))
                    C4_image_list.append(str(image_path).replace("C1", 'C4'))
                elif self.ds_type == "test" and int(image_path.stem) > 2000:
                    C1_image_list.append(str(image_path))
                    C4_image_list.append(str(image_path).replace("C1", 'C4'))
            left_image_list, right_image_list = C1_image_list, C4_image_list
        else:
            raise NotImplementedError

        return left_image_list, right_image_list

class MultiCameraImageDataset(Dataset):
    def __init__(self, ds_type='train', ds_name='wildtrack', root='/home/xzhangga/datasets/WildTrack/', crop_size=(256, 256), num_camera=7, **kwargs):
        super().__init__()
        
        self.path = Path(f"{root}")
        self.ds_name = ds_name
        self.ds_type = ds_type
        if ds_type=="train":
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(crop_size), 
                transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),]) 
        else: 
            self.transform = transforms.Compose([transforms.ToTensor()])
        self.num_camera = num_camera
        self.image_lists = self.get_files()
        if ds_type == "test":
            self.crop = MinimalCrop(min_div=64)
        else:
            self.crop = None

        print(f'Loaded dataset {ds_name} from {self.path}. Found {len(self.image_lists[0])} files.')

    def __len__(self):
        return len(self.image_lists[0])

    def __getitem__(self, index):
        image_list = [Image.open(self.image_lists[i][index]).convert('RGB') for i in range(self.num_camera)]
        if self.crop is not None:
            image_list = [self.crop(image) for image in image_list]
        frames = np.concatenate([np.asarray(image) for image in image_list], axis=-1)
        frames = torch.chunk(self.transform(frames), self.num_camera)
        #if random.random() < 0.5:
        #    frames = frames[::-1]
        return frames

    def set_stage(self, stage):
        if stage == 0:
            print('Using (32, 32) crops')
            self.crop = transforms.RandomCrop((32, 32))
        elif stage == 1:
            print('Using (28, 28) crops')
            self.crop = transforms.RandomCrop((28, 28))

    def get_files(self):
        if self.ds_name == 'wildtrack':
            image_lists = [[] for i in range(self.num_camera)]
            for image_path in self.path.glob(f'images/C1/*.png'):
                if self.ds_type == "train" and int(image_path.stem) <= 2000:
                    image_lists[0].append(str(image_path))
                    for idx in range(1, self.num_camera):
                        image_lists[idx].append(str(image_path).replace("C1", 'C'+str(idx+1))) 
                elif self.ds_type == "test" and int(image_path.stem) > 2000:
                    image_lists[0].append(str(image_path))
                    for idx in range(1, self.num_camera):
                        image_lists[idx].append(str(image_path).replace("C1", 'C'+str(idx+1)))
        else:
            raise NotImplementedError

        return image_lists

class AdaptiveMultiCameraImageDataset(Dataset):
    def __init__(self, ds_type='train', ds_name='wildtrack', root='/home/xzhangga/datasets/WildTrack/', crop_size=(256, 256), **kwargs):
        super().__init__()
        
        self.path = Path(f"{root}")
        self.ds_name = ds_name
        self.ds_type = ds_type
        if ds_type=="train":
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(crop_size), 
                transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),]) 
        else: 
            self.transform = transforms.Compose([transforms.ToTensor()])
        self.image_lists = self.get_files()
        self.set_num_camera()
        if ds_type == "test":
            self.crop = MinimalCrop(min_div=64)
            self.num_camera = 7
        else:
            self.crop = None

        print(f'Loaded dataset {ds_name} from {self.path}. Found {len(self.image_lists[0])} files.')

    def __len__(self):
        return len(self.image_lists[0])

    def __getitem__(self, index):
        if self.ds_type == "train":
            image_list = [Image.open(self.image_lists[i][index]).convert('RGB') for i in self.images_index]
        else:
            image_list = [Image.open(self.image_lists[i][index]).convert('RGB') for i in range(self.num_camera)]
        if self.crop is not None:
            image_list = [self.crop(image) for image in image_list]
        frames = np.concatenate([np.asarray(image) for image in image_list], axis=-1)
        frames = torch.chunk(self.transform(frames), self.num_camera)
        #if random.random() < 0.5:
        #    frames = frames[::-1]
        return frames

    def set_num_camera(self):
        self.num_camera = random.randint(2, 7)
        self.images_index = random.sample(range(7), self.num_camera)
        #print("num_camera:",self.num_camera)

    def get_files(self, num_camera=7):
        if self.ds_name == 'wildtrack':
            image_lists = [[] for i in range(num_camera)]
            for image_path in self.path.glob(f'images/C1/*.png'):
                if self.ds_type == "train" and int(image_path.stem) <= 2000:
                    image_lists[0].append(str(image_path))
                    for idx in range(1, num_camera):
                        image_lists[idx].append(str(image_path).replace("C1", 'C'+str(idx+1))) 
                elif self.ds_type == "test" and int(image_path.stem) > 2000:
                    image_lists[0].append(str(image_path))
                    for idx in range(1, num_camera):
                        image_lists[idx].append(str(image_path).replace("C1", 'C'+str(idx+1)))
        else:
            raise NotImplementedError

        return image_lists


def save_checkpoint(state, is_best=False, log_dir=None, filename="ckpt.pth.tar"):
    save_file = os.path.join(log_dir, filename)
    print("save model in:", save_file)
    torch.save(state, save_file)
    if is_best:
        torch.save(state, os.path.join(log_dir, filename.replace(".pth.tar", ".best.pth.tar")))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def get_output_folder(parent_dir, env_name, output_current_folder=False):
    """Return save folder.
    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.
    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.
    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    if not output_current_folder: 
        experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir, experiment_id

class CheckMaskedConv2d(nn.Conv2d):
    """
    if kernel_size == (5, 5)
    then mask: A
        [[0., 1., 0., 1., 0.],
        [1., 0., 1., 0., 1.],
        [0., 1., 0., 1., 0.],
        [1., 0., 1., 0., 1.],
        [0., 1., 0., 1., 0.]]
    0: non-anchor
    1: anchor
    """
    def __init__(self, *args, mask_type: str = "A", **kwargs):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')
        self.register_buffer("mask", torch.zeros_like(self.weight.data))

        if mask_type == "A":
            self.mask[:, :, 0::2, 1::2] = 1
            self.mask[:, :, 1::2, 0::2] = 1
        else:
            self.mask[:, :, 0::2, 0::2] = 1
            self.mask[:, :, 1::2, 1::2] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        out = super().forward(x)
        return out

class Hyperprior(CompressionModel):
    def __init__(self, in_planes: int = 192, mid_planes: int = 192, out_planes: int=192):
        super().__init__(entropy_bottleneck_channels=mid_planes)
        self.hyper_encoder = nn.Sequential(
            conv(in_planes, mid_planes, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(mid_planes, mid_planes, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(mid_planes, mid_planes, stride=2, kernel_size=5),
        )
        if out_planes == 2 * in_planes:
            self.hyper_decoder = nn.Sequential(
                deconv(mid_planes, mid_planes, stride=2, kernel_size=5),
                nn.LeakyReLU(inplace=True),
                deconv(mid_planes, in_planes * 3 // 2, stride=2, kernel_size=5),
                nn.LeakyReLU(inplace=True),
                conv(in_planes * 3 // 2, out_planes, stride=1, kernel_size=3),
            )
        else:
            self.hyper_decoder = nn.Sequential(
                deconv(mid_planes, mid_planes, stride=2, kernel_size=5),
                nn.LeakyReLU(inplace=True),
                deconv(mid_planes, mid_planes, stride=2, kernel_size=5),
                nn.LeakyReLU(inplace=True),
                conv(mid_planes, out_planes, stride=1, kernel_size=3),
            )

    def forward(self, y, out_z=False):
        z = self.hyper_encoder(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = ste_round(z - z_offset) + z_offset
        params = self.hyper_decoder(z_hat)
        if out_z:
            return params, z_likelihoods, z_hat
        else:
            return params, z_likelihoods

    def compress(self, y):
        z = self.hyper_encoder(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        params = self.hyper_decoder(z_hat)
        return params, z_hat, z_strings #{"strings": z_string, "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        #assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings, shape)
        params = self.hyper_decoder(z_hat)
        return params, z_hat


class JointContextTransfer(nn.Module):
    def __init__(self, channels):
        super(JointContextTransfer, self).__init__()
        self.rb1 = ResidualBlock(channels, channels)
        self.rb2 = ResidualBlock(channels, channels)
        self.attn = EfficientAttention(key_in_channels=channels, query_in_channels=channels, key_channels=channels//8, 
            head_count=2, value_channels=channels//4)

        self.refine = nn.Sequential(
            ResidualBlock(channels*2, channels),
            ResidualBlock(channels, channels))

    def forward(self, x_left, x_right):
        B, C, H, W = x_left.size()
        identity_left, identity_right = x_left, x_right
        x_left, x_right = self.rb2(self.rb1(x_left)), self.rb2(self.rb1(x_right))
        A_right_to_left, A_left_to_right = self.attn(x_left, x_right), self.attn(x_right, x_left)
        compact_left = identity_left + self.refine(torch.cat((A_right_to_left, x_left), dim=1))
        compact_right = identity_right + self.refine(torch.cat((A_left_to_right, x_right), dim=1))
        return compact_left, compact_right


class Multi_JointContextTransfer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.rb = nn.Sequential(
            ResidualBlock(channels, channels),
            ResidualBlock(channels, channels),
        )
        self.aggeregate_module = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, None, None)),
        )
        self.attn = EfficientAttention(key_in_channels=channels, query_in_channels=channels, key_channels=channels//8, 
            head_count=2, value_channels=channels//4)

        self.refine = nn.Sequential(
            ResidualBlock(channels*2, channels),
            ResidualBlock(channels, channels))

    def forward(self, x, num_camera):
        identity_list = x.chunk(num_camera, 0)
        rb_x = self.rb(x)
        rb_x_list = rb_x.chunk(num_camera, 0)
        compact_list = []
        for idx, rb in enumerate(rb_x_list):
            other_rb = [r.unsqueeze(2) for i, r in enumerate(rb_x_list) if i!=idx]
            other_rb = torch.cat(other_rb, dim=2)
            aggeregate_rb = self.aggeregate_module(other_rb).squeeze(2)
            #print(rb.shape, aggeregate_rb.shape)
            A_other_camera_to_current = self.attn(rb, aggeregate_rb)
            compact = identity_list[idx] + self.refine(torch.cat([A_other_camera_to_current, rb], dim=1))
            compact_list.append(compact)
        
        return torch.cat(compact_list, dim=0)


class EfficientAttention(nn.Module):
    def __init__(self, key_in_channels, query_in_channels, key_channels=32, head_count=8, value_channels=64):
        super().__init__()
        self.in_channels = query_in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(key_in_channels, key_channels, 1)
        self.queries = nn.Conv2d(query_in_channels, key_channels, 1)
        self.values = nn.Conv2d(key_in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, query_in_channels, 1)

    def forward(self, target, input):
        n, _, h, w = input.size()
        keys = self.keys(input).reshape((n, self.key_channels, h * w))
        queries = self.queries(target).reshape(n, self.key_channels, h * w)
        values = self.values(input).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
        
        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels,:], dim=2)
            query = F.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels,:], dim=1)
            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]
            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value #+ input_

        return attention

    def parallel_forward(self, target, input):
        n, _, h, w = input.size()
        keys = self.keys(input).reshape((n, self.key_channels, h * w))
        queries = self.queries(target).reshape(n, self.key_channels, h * w)
        values = self.values(input).reshape((n, self.value_channels, h * w))
        
        keys = keys.permute(0, 2, 1).reshape(n, h*w, self.head_count, -1).permute(0, 2, 3, 1).reshape(n*self.head_count, -1, h*w)
        queries = queries.permute(0, 2, 1).reshape(n, h*w, self.head_count, -1).permute(0, 2, 3, 1).reshape(n*self.head_count, -1, h*w)
        values = values.permute(0, 2, 1).reshape(n, h*w, self.head_count, -1).permute(0, 2, 3, 1).reshape(n*self.head_count, -1, h*w)

        key = F.softmax(keys, dim=2)
        queries = F.softmax(queries, dim=1)
        context = key @ value.transpose(1, 2)
        attended_values = (context.transpose(1, 2) @ query).reshape(n, -1, h, w)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value #+ target
        return attention


class LDMIC(nn.Module):
    def __init__(self, N = 128, M = 192, decode_atten=JointContextTransfer):
        super().__init__()
        self.encoder = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2)
        )
        self.hyperprior = Hyperprior(in_planes=M, mid_planes=N, out_planes=M*2)
        self.context_prediction = MaskedConv2d(
            M, M*2, kernel_size=5, padding=2, stride=1
        )
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )
        self.gaussian_conditional = GaussianConditional(None)
        self.M = M
        self.atten_3 = decode_atten(M)
        self.decoder_1 = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),)
        self.atten_4 = decode_atten(N)
        self.decoder_2 = nn.Sequential(
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2)
        )

    def forward(self, x):
        x_left, x_right = x[0], x[1] #x.chunk(2, 1) 

        y_left, y_right = self.encoder(x_left), self.encoder(x_right)
        left_params, z_left_likelihoods, z_left_hat = self.hyperprior(y_left, out_z=True)
        right_params, z_right_likelihoods, z_right_hat = self.hyperprior(y_right, out_z=True)
        y_left_hat = self.gaussian_conditional.quantize(
            y_left, "noise" if self.training else "dequantize"
        )
        y_right_hat = self.gaussian_conditional.quantize(
            y_right, "noise" if self.training else "dequantize"
        )
        ctx_left_params = self.context_prediction(y_left_hat)
        ctx_right_params = self.context_prediction(y_right_hat)

        gaussian_left_params = self.entropy_parameters(torch.cat([left_params, ctx_left_params], 1))
        gaussian_right_params = self.entropy_parameters(torch.cat([right_params, ctx_right_params], 1))
        
        left_means_hat, left_scales_hat = gaussian_left_params.chunk(2, 1)
        right_means_hat, right_scales_hat = gaussian_right_params.chunk(2, 1)
 
        _, y_left_likelihoods = self.gaussian_conditional(y_left, left_scales_hat, means=left_means_hat)
        _, y_right_likelihoods = self.gaussian_conditional(y_right, right_scales_hat, means=right_means_hat)


        y_left_ste, y_right_ste = ste_round(y_left - left_means_hat) + left_means_hat, ste_round(y_right - right_means_hat) + right_means_hat
        y_left, y_right = self.atten_3(y_left_ste, y_right_ste)
        y_left, y_right = self.atten_4(self.decoder_1(y_left), self.decoder_1(y_right))
        x_left_hat, x_right_hat = self.decoder_2(y_left), self.decoder_2(y_right)
        
        return {
            "x_hat": [x_left_hat, x_right_hat],
            "likelihoods": [{"y": y_left_likelihoods, "z": z_left_likelihoods}, {"y":y_right_likelihoods, "z":z_right_likelihoods}],
            "feature": [y_left_ste, y_right_ste, z_left_hat, z_right_hat, left_means_hat, right_means_hat],
        }

    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""
        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel):
                aux_loss_list.append(m.aux_loss())
        return aux_loss_list

    def fix_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.hyperprior.parameters():
            p.requires_grad = False
        for p in self.context_prediction.parameters():
            p.requires_grad = False
        for p in self.entropy_parameters.parameters():
            p.requires_grad = False
        for p in self.gaussian_conditional.parameters():
            p.requires_grad = False

    def load_encoder(self, current_state_dict, checkpoint):
        encoder_dict = {k.replace("g_a", "encoder"): v for k, v in checkpoint.items() if "g_a" in k}
        context_prediction_dict = {k: v for k, v in checkpoint.items() if "context_prediction" in k}
        entropy_parameters_dict = {k: v for k, v in checkpoint.items() if "entropy_parameters" in k}
        gaussian_conditional_dict = {k: v for k, v in checkpoint.items() if "gaussian_conditional" in k}
        hyperprior_dict = {}
        for k, v in checkpoint.items():
            if "h_a" in k:
                hyperprior_dict[k.replace("h_a", "hyperprior.hyper_encoder")] = v
            elif "h_s" in k:
                hyperprior_dict[k.replace("h_s", "hyperprior.hyper_decoder")] = v
            elif "entropy_bottleneck" in k:
                hyperprior_dict[k.replace("entropy_bottleneck", "hyperprior.entropy_bottleneck")] = v

        current_state_dict.update(encoder_dict)
        current_state_dict.update(hyperprior_dict)
        current_state_dict.update(context_prediction_dict)
        current_state_dict.update(entropy_parameters_dict)
        current_state_dict.update(gaussian_conditional_dict)
        #print(current_state_dict.keys())
        #input()
        return current_state_dict

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.hyperprior.entropy_bottleneck,
            "hyperprior.entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.hyperprior.entropy_bottleneck.update(force=force)
        return updated

    def compress(self, x):
        x_left, x_right = x[0], x[1]
        left_dict = self.encode(x_left)
        right_dict = self.encode(x_right)
        return left_dict, right_dict

    def decompress(self, left_dict, right_dict):
        y_left_hat = self.decode(left_dict["strings"], left_dict["shape"])
        y_right_hat = self.decode(right_dict["strings"], right_dict["shape"])
        y_left, y_right = self.atten_3(y_left_hat, y_right_hat)
        y_left, y_right = self.atten_4(self.decoder_1(y_left), self.decoder_1(y_right))
        x_left_hat, x_right_hat = self.decoder_2(y_left).clamp_(0, 1), self.decoder_2(y_right).clamp_(0, 1)
        return {
            "x_hat": [x_left_hat, x_right_hat],
        }  

    def encode(self, x):
        y = self.encoder(x)
        params, z_hat, z_strings = self.hyperprior.compress(y)
        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z_hat.size()[-2:]}

    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                means_hat, scales_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)

                y_crop = y_crop[:, :, padding, padding]
                y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string

    def decode(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder
        params, z_hat = self.hyperprior.decompress(strings[1], shape)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        #x_hat = self.g_s(y_hat).clamp_(0, 1)
        return y_hat

    def _decompress_ar(self, y_string, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                means_hat, scales_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv

class LDMIC_checkboard(nn.Module):
    def __init__(self, N = 128, M = 192, decode_atten=JointContextTransfer, training=False):
        super().__init__()
        self.encoder = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2)
        )
        self.atten_3 = decode_atten(M)
        self.decoder_1 = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),)
        self.atten_4 = decode_atten(N)
        self.decoder_2 = nn.Sequential(
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2)
        )
        
        self.hyperprior = Hyperprior(in_planes=M, mid_planes=N, out_planes=M*2)
        self.context_prediction = CheckMaskedConv2d(
            M, M*2, kernel_size=5, padding=2, stride=1
        )
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )
        self.gaussian_conditional = GaussianConditional(None)
        self.M = M
        self.N = N
        if training:
            self.training_ctx_params_anchor = torch.zeros([8, self.M * 2, 16, 16]).cuda()

    def forward(self, x):
        x_left, x_right = x[0], x[1] #x.chunk(2, 1)
        y_left, y_right = self.encoder(x_left), self.encoder(x_right)
        

        y_left_ste, y_left_likelihoods = self.forward_entropy(y_left)
        y_right_ste, y_right_likelihoods = self.forward_entropy(y_right) 

        y_left, y_right = self.atten_3(y_left_ste, y_right_ste)
        y_left, y_right = self.atten_4(self.decoder_1(y_left), self.decoder_1(y_right))
        x_left_hat, x_right_hat = self.decoder_2(y_left), self.decoder_2(y_right)

        return {
            "x_hat": [x_left_hat, x_right_hat],
            "likelihoods": [y_left_likelihoods, y_right_likelihoods],
        }

    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""
        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel):
                aux_loss_list.append(m.aux_loss())
        return aux_loss_list

    def forward_entropy(self, y):
        params, z_likelihoods = self.hyperprior(y)

        batch_size, _, y_height, y_width = y.size()
        # compress anchor
        if self.training:
            ctx_params_anchor = self.training_ctx_params_anchor[:batch_size]
        else:
            ctx_params_anchor = torch.zeros([batch_size, self.M * 2, y_height, y_width]).to(y.device)
        gaussian_params_anchor = self.entropy_parameters(
            torch.cat([ctx_params_anchor, params], dim=1)
        )
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        anchor = self.get_anchor(y_hat)
        ctx_params_non_anchor = self.context_prediction(anchor)
        gaussian_params_non_anchor = self.entropy_parameters(
            torch.cat([ctx_params_non_anchor, params], dim=1)
        )
        scales_non_anchor, means_non_anchor = gaussian_params_non_anchor.chunk(2, 1)
        scales_hat, means_hat = self.merge(scales_anchor, means_anchor, 
            scales_non_anchor, means_non_anchor)
        _, y_likelihoods = self.gaussian_conditional(y, scales=scales_hat, means=means_hat)
        y_ste = ste_round(y-means_hat) + means_hat

        return y_ste, {"y": y_likelihoods, "z": z_likelihoods}

    def merge(self, scales_anchor, means_anchor, scales_non_anchor, means_non_anchor, mask_type="A"):
        scales_hat = scales_anchor.clone()
        means_hat = means_anchor.clone()
        if mask_type == "A":
            scales_hat[:, :, 0::2, 0::2] = scales_non_anchor[:, :, 0::2, 0::2]
            scales_hat[:, :, 1::2, 1::2] = scales_non_anchor[:, :, 1::2, 1::2]
            means_hat[:, :, 0::2, 0::2] = means_non_anchor[:, :, 0::2, 0::2]
            means_hat[:, :, 1::2, 1::2] = means_non_anchor[:, :, 1::2, 1::2]
        else:
            scales_hat[:, :, 0::2, 1::2] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_hat[:, :, 1::2, 0::2] = scales_non_anchor[:, :, 1::2, 0::2]
            means_hat[:, :, 0::2, 1::2] = means_non_anchor[:, :, 0::2, 1::2]
            means_hat[:, :, 1::2, 0::2] = means_non_anchor[:, :, 1::2, 0::2]
        return scales_hat, means_hat

    def get_anchor(self, y_hat, mask_type="A"):
        y_anchor = y_hat.clone()
        if mask_type == "A":
            y_anchor[:, :, 0::2, 0::2] = 0
            y_anchor[:, :, 1::2, 1::2] = 0
        else:
            y_anchor[:, :, 0::2, 1::2] = 0
            y_anchor[:, :, 1::2, 0::2] = 0
        return y_anchor

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.hyperprior.entropy_bottleneck,
            "hyperprior.entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.hyperprior.entropy_bottleneck.update(force=force)
        return updated

    def compress(self, x):
        x_left, x_right = x[0], x[1]
        left_dict = self.encode(x_left)
        right_dict = self.encode(x_right)
        return left_dict, right_dict

    def decompress(self, left_dict, right_dict):
        y_left_hat = self.decode(left_dict["strings"], left_dict["shape"])
        y_right_hat = self.decode(right_dict["strings"], right_dict["shape"])
        #print(y_left_hat[0, 0, 0, 0:10], y_right_hat[0, 0, 0, 0:10])
        y_left, y_right = self.atten_3(y_left_hat, y_right_hat)
        #print(y_left[0, 0, 0, 0:10], y_right[0, 0, 0, 0:10])
        y_left, y_right = self.atten_4(self.decoder_1(y_left), self.decoder_1(y_right))
        #print(y_left[0, 0, 0, 0:10], y_right[0, 0, 0, 0:10])
        x_left_hat, x_right_hat = self.decoder_2(y_left), self.decoder_2(y_right) #.clamp_(0, 1), self.decoder_2(y_right).clamp_(0, 1)
        
        return {
            "x_hat": [x_left_hat, x_right_hat],
        }   

    def encode(self, x):
        """
        Slice y into four parts A, B, C and D. Encode and Decode them separately.
        Workflow is described in Figure 1 and Figure 2 in Supplementary Material.
        NA  A   ->  A B
        A  NA       C D
        After padding zero:
            anchor : 0 B     non-anchor: A 0
                     c 0                 0 D
        """
        batch_size, channel, x_height, x_width = x.shape

        y = self.encoder(x)

        y_a = y[:, :, 0::2, 0::2]
        y_d = y[:, :, 1::2, 1::2]
        y_b = y[:, :, 0::2, 1::2]
        y_c = y[:, :, 1::2, 0::2]

        params, z_hat, z_strings = self.hyperprior.compress(y)

        anchor = torch.zeros_like(y).to(x.device)
        anchor[:, :, 0::2, 1::2] = y[:, :, 0::2, 1::2]
        anchor[:, :, 1::2, 0::2] = y[:, :, 1::2, 0::2]
        ctx_params_anchor = torch.zeros([batch_size, self.M * 2, x_height // 16, x_width // 16]).to(x.device)
        gaussian_params_anchor = self.entropy_parameters(
            torch.cat([ctx_params_anchor, params], dim=1)
        )
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)

        scales_b = scales_anchor[:, :, 0::2, 1::2]
        means_b = means_anchor[:, :, 0::2, 1::2]
        indexes_b = self.gaussian_conditional.build_indexes(scales_b)
        y_b_strings = self.gaussian_conditional.compress(y_b, indexes_b, means_b)
        y_b_quantized = self.gaussian_conditional.decompress(y_b_strings, indexes_b, means=means_b)

        scales_c = scales_anchor[:, :, 1::2, 0::2]
        means_c = means_anchor[:, :, 1::2, 0::2]
        indexes_c = self.gaussian_conditional.build_indexes(scales_c)
        y_c_strings = self.gaussian_conditional.compress(y_c, indexes_c, means_c)
        y_c_quantized = self.gaussian_conditional.decompress(y_c_strings, indexes_c, means=means_c)

        anchor_quantized = torch.zeros_like(y).to(x.device)
        anchor_quantized[:, :, 0::2, 1::2] = y_b_quantized[:, :, :, :]
        anchor_quantized[:, :, 1::2, 0::2] = y_c_quantized[:, :, :, :]

        ctx_params_non_anchor = self.context_prediction(anchor_quantized)
        gaussian_params_non_anchor = self.entropy_parameters(
            torch.cat([ctx_params_non_anchor, params], dim=1)
        )
        scales_non_anchor, means_non_anchor = gaussian_params_non_anchor.chunk(2, 1)

        scales_a = scales_non_anchor[:, :, 0::2, 0::2]
        means_a = means_non_anchor[:, :, 0::2, 0::2]
        indexes_a = self.gaussian_conditional.build_indexes(scales_a)
        y_a_strings = self.gaussian_conditional.compress(y_a, indexes_a, means=means_a)

        scales_d = scales_non_anchor[:, :, 1::2, 1::2]
        means_d = means_non_anchor[:, :, 1::2, 1::2]
        indexes_d = self.gaussian_conditional.build_indexes(scales_d)
        y_d_strings = self.gaussian_conditional.compress(y_d, indexes_d, means=means_d)

        return {
            "strings": [y_a_strings, y_b_strings, y_c_strings, y_d_strings, z_strings],
            "shape": z_hat.size()[-2:]
        }

    def decode(self, strings, shape):
        """
        Slice y into four parts A, B, C and D. Encode and Decode them separately.
        Workflow is described in Figure 1 and Figure 2 in Supplementary Material.
        NA  A   ->  A B
        A  NA       C D
        After padding zero:
            anchor : 0 B     non-anchor: A 0
                     c 0                 0 D
        """
        params, z_hat = self.hyperprior.decompress(strings[4], shape)
        #z_hat = self.entropy_bottleneck.decompress(strings[4], shape)
        #params = self.h_s(z_hat)

        batch_size, channel, z_height, z_width = z_hat.shape
        ctx_params_anchor = torch.zeros([batch_size, self.M * 2, z_height * 4, z_width * 4]).to(z_hat.device)
        gaussian_params_anchor = self.entropy_parameters(
            torch.cat([ctx_params_anchor, params], dim=1)
        )
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)

        scales_b = scales_anchor[:, :, 0::2, 1::2]
        means_b = means_anchor[:, :, 0::2, 1::2]
        indexes_b = self.gaussian_conditional.build_indexes(scales_b)
        y_b_quantized = self.gaussian_conditional.decompress(strings[1], indexes_b, means=means_b)

        scales_c = scales_anchor[:, :, 1::2, 0::2]
        means_c = means_anchor[:, :, 1::2, 0::2]
        indexes_c = self.gaussian_conditional.build_indexes(scales_c)
        y_c_quantized = self.gaussian_conditional.decompress(strings[2], indexes_c, means=means_c)

        anchor_quantized = torch.zeros([batch_size, self.M, z_height * 4, z_width * 4]).to(z_hat.device)
        anchor_quantized[:, :, 0::2, 1::2] = y_b_quantized[:, :, :, :]
        anchor_quantized[:, :, 1::2, 0::2] = y_c_quantized[:, :, :, :]

        ctx_params_non_anchor = self.context_prediction(anchor_quantized)
        gaussian_params_non_anchor = self.entropy_parameters(
            torch.cat([ctx_params_non_anchor, params], dim=1)
        )
        scales_non_anchor, means_non_anchor = gaussian_params_non_anchor.chunk(2, 1)

        scales_a = scales_non_anchor[:, :, 0::2, 0::2]
        means_a = means_non_anchor[:, :, 0::2, 0::2]
        indexes_a = self.gaussian_conditional.build_indexes(scales_a)
        y_a_quantized = self.gaussian_conditional.decompress(strings[0], indexes_a, means=means_a)

        scales_d = scales_non_anchor[:, :, 1::2, 1::2]
        means_d = means_non_anchor[:, :, 1::2, 1::2]
        indexes_d = self.gaussian_conditional.build_indexes(scales_d)
        y_d_quantized = self.gaussian_conditional.decompress(strings[3], indexes_d, means=means_d)

        # Add non_anchor_quantized
        anchor_quantized[:, :, 0::2, 0::2] = y_a_quantized[:, :, :, :]
        anchor_quantized[:, :, 1::2, 1::2] = y_d_quantized[:, :, :, :]

        #print(anchor_quantized[0, 0, 0, :])
        return anchor_quantized 

class Multi_LDMIC(nn.Module):
    def __init__(self, N = 128, M = 192, decode_atten=Multi_JointContextTransfer):
        super().__init__()
        self.encoder = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2)
        )

        self.atten_3 = decode_atten(M)
        self.decoder_1 = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),)
        self.atten_4 = decode_atten(N)
        self.decoder_2 = nn.Sequential(
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2)
        )
        
        self.hyperprior = Hyperprior(in_planes=M, mid_planes=N, out_planes=M*2)
        self.context_prediction = MaskedConv2d(
            M, M*2, kernel_size=5, padding=2, stride=1
        )
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        num_camera = len(x)

        x = torch.cat(x, dim=0)
        y = self.encoder(x)
        params, z_likelihoods = self.hyperprior(y)
        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(torch.cat([params, ctx_params], 1))
        means_hat, scales_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        y_ste = ste_round(y-means_hat)+means_hat

        y_ste = self.decoder_1(self.atten_3(y_ste, num_camera))
        x_hat = self.decoder_2(self.atten_4(y_ste, num_camera))
        
        x_hat_list = x_hat.chunk(num_camera, 0)
        z_likelihoods_list = z_likelihoods.chunk(num_camera, 0)
        y_likelihoods_list = y_likelihoods.chunk(num_camera, 0)
        likelihoods = [{"y": y_likelihood, "z": z_likelihood} for y_likelihood, z_likelihood in zip(y_likelihoods_list, z_likelihoods_list)]

        return {
            "x_hat": x_hat_list,
            "likelihoods": likelihoods,
        }

    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""
        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel):
                aux_loss_list.append(m.aux_loss())
        return aux_loss_list

    def fix_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.hyperprior.parameters():
            p.requires_grad = False
        for p in self.context_prediction.parameters():
            p.requires_grad = False
        for p in self.entropy_parameters.parameters():
            p.requires_grad = False
        for p in self.gaussian_conditional.parameters():
            p.requires_grad = False

    def load_encoder(self, current_state_dict, checkpoint):
        encoder_dict = {k.replace("g_a", "encoder"): v for k, v in checkpoint.items() if "g_a" in k}
        context_prediction_dict = {k: v for k, v in checkpoint.items() if "context_prediction" in k}
        entropy_parameters_dict = {k: v for k, v in checkpoint.items() if "entropy_parameters" in k}
        gaussian_conditional_dict = {k: v for k, v in checkpoint.items() if "gaussian_conditional" in k}
        hyperprior_dict = {}
        for k, v in checkpoint.items():
            if "h_a" in k:
                hyperprior_dict[k.replace("h_a", "hyperprior.hyper_encoder")] = v
            elif "h_s" in k:
                hyperprior_dict[k.replace("h_s", "hyperprior.hyper_decoder")] = v
            elif "entropy_bottleneck" in k:
                hyperprior_dict[k.replace("entropy_bottleneck", "hyperprior.entropy_bottleneck")] = v

        current_state_dict.update(encoder_dict)
        current_state_dict.update(hyperprior_dict)
        current_state_dict.update(context_prediction_dict)
        current_state_dict.update(entropy_parameters_dict)
        current_state_dict.update(gaussian_conditional_dict)
        return current_state_dict

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.hyperprior.entropy_bottleneck,
            "hyperprior.entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.hyperprior.entropy_bottleneck.update(force=force)
        return updated

class Multi_LDMIC_checkboard(nn.Module):
    def __init__(self, N = 128, M = 192, decode_atten=Multi_JointContextTransfer, training=False):
        super().__init__()
        self.encoder = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2)
        )
    
        self.atten_3 = decode_atten(M)
        self.decoder_1 = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),)
        self.atten_4 = decode_atten(N)
        self.decoder_2 = nn.Sequential(
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2)
        )
        
        self.hyperprior = Hyperprior(in_planes=M, mid_planes=N, out_planes=M*2)
        self.context_prediction = CheckMaskedConv2d(
            M, M*2, kernel_size=5, padding=2, stride=1
        )
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )
        self.gaussian_conditional = GaussianConditional(None)
        self.M = M

        if training:
            self.training_ctx_params_anchor = torch.zeros([8*7, M * 2, 16, 16]).cuda()

    def forward(self, x):
        num_camera = len(x)
        x = torch.cat(x, dim=0)
        y = self.encoder(x)

        y_ste, y_likelihoods, z_likelihoods = self.forward_entropy(y)
        y_ste = self.decoder_1(self.atten_3(y_ste, num_camera))
        x_hat = self.decoder_2(self.atten_4(y_ste, num_camera))
        
        x_hat_list = x_hat.chunk(num_camera, 0)
        z_likelihoods_list = z_likelihoods.chunk(num_camera, 0)
        y_likelihoods_list = y_likelihoods.chunk(num_camera, 0)
        likelihoods = [{"y": y_likelihood, "z": z_likelihood} for y_likelihood, z_likelihood in zip(y_likelihoods_list, z_likelihoods_list)]

        return {
            "x_hat": x_hat_list,
            "likelihoods": likelihoods,
        }

    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""
        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel):
                aux_loss_list.append(m.aux_loss())
        return aux_loss_list


    def forward_entropy(self, y):
        params, z_likelihoods = self.hyperprior(y)

        batch_size, _, y_height, y_width = y.size()
        # compress anchor
        if self.training:
            ctx_params_anchor = self.training_ctx_params_anchor[:batch_size]
        else:
            ctx_params_anchor = torch.zeros([batch_size, self.M * 2, y_height, y_width]).to(y.device)
        gaussian_params_anchor = self.entropy_parameters(
            torch.cat([ctx_params_anchor, params], dim=1)
        )
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        anchor = self.get_anchor(y_hat)
        ctx_params_non_anchor = self.context_prediction(anchor)
        gaussian_params_non_anchor = self.entropy_parameters(
            torch.cat([ctx_params_non_anchor, params], dim=1)
        )
        scales_non_anchor, means_non_anchor = gaussian_params_non_anchor.chunk(2, 1)
        scales_hat, means_hat = self.merge(scales_anchor, means_anchor, 
            scales_non_anchor, means_non_anchor)
        _, y_likelihoods = self.gaussian_conditional(y, scales=scales_hat, means=means_hat)
        y_ste = ste_round(y-means_hat)+means_hat

        return y_ste, y_likelihoods, z_likelihoods

    def merge(self, scales_anchor, means_anchor, scales_non_anchor, means_non_anchor, mask_type="A"):
        scales_hat = scales_anchor.clone()
        means_hat = means_anchor.clone()
        if mask_type == "A":
            scales_hat[:, :, 0::2, 0::2] = scales_non_anchor[:, :, 0::2, 0::2]
            scales_hat[:, :, 1::2, 1::2] = scales_non_anchor[:, :, 1::2, 1::2]
            means_hat[:, :, 0::2, 0::2] = means_non_anchor[:, :, 0::2, 0::2]
            means_hat[:, :, 1::2, 1::2] = means_non_anchor[:, :, 1::2, 1::2]
        else:
            scales_hat[:, :, 0::2, 1::2] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_hat[:, :, 1::2, 0::2] = scales_non_anchor[:, :, 1::2, 0::2]
            means_hat[:, :, 0::2, 1::2] = means_non_anchor[:, :, 0::2, 1::2]
            means_hat[:, :, 1::2, 0::2] = means_non_anchor[:, :, 1::2, 0::2]
        return scales_hat, means_hat

    def get_anchor(self, y_hat, mask_type="A"):
        y_anchor = y_hat.clone()
        if mask_type == "A":
            y_anchor[:, :, 0::2, 0::2] = 0
            y_anchor[:, :, 1::2, 1::2] = 0
        else:
            y_anchor[:, :, 0::2, 1::2] = 0
            y_anchor[:, :, 1::2, 0::2] = 0
        return y_anchor

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.hyperprior.entropy_bottleneck,
            "hyperprior.entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.hyperprior.entropy_bottleneck.update(force=force)
        return updated
 

class Multi_MSE_Loss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""
    def __init__(self, device):
        super().__init__()
        self.mse = nn.MSELoss().to(device)
        #self.num_camera = num_camera
        #self.lmbda = lmbda

    def forward(self, output, target, lmbda):
        #target1, target2 = target[0], target[1]
        num_camera = len(target)
        N, _, H, W = target[0].size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = 0
        out["mse_loss"] = 0
        out["psnr"] = 0

        # 计算误差
        for i in range(num_camera):
            out['bpp'+str(i)] = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in output['likelihoods'][i].values())
            out["mse"+str(i)] = self.mse(output['x_hat'][i], target[i])
            out["bpp_loss"] += out['bpp'+str(i)]/num_camera
            out['mse_loss'] += lmbda * out["mse"+str(i)] /num_camera
            if out["mse"+str(i)] > 0:
                out["psnr"+str(i)] = 10 * (torch.log10(1 / out["mse"+str(i)])).mean()
            else:
                out["psnr"+str(i)] = 0
            out["psnr"] += out["psnr"+str(i)]/num_camera
        
        out['loss'] = out['mse_loss'] + out['bpp_loss']
        return out

class Multi_MS_SSIM_Loss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""
    def __init__(self, device, size_average=True, max_val=1):
        super().__init__()
        self.ms_ssim = MS_SSIM(size_average, max_val).to(device)
        #self.num_camera = num_camera
        #self.lmbda = lmbda

    def forward(self, output, target, lmbda):
        #target1, target2 = target[0], target[1]
        num_camera = len(target)
        N, _, H, W = target[0].size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = 0
        out["ms_ssim_loss"] = 0
        out["ms_db"] = 0

        # 计算误差
        for i in range(num_camera):
            out['bpp'+str(i)] = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in output['likelihoods'][i].values())
            out["ms_ssim"+str(i)] = 1 - self.ms_ssim(output['x_hat'][i], target[i])
            out["bpp_loss"] += out['bpp'+str(i)]/num_camera
            out['ms_ssim_loss'] += lmbda * out["ms_ssim"+str(i)] /num_camera
            if out["ms_ssim"+str(i)] > 0:
                out["ms_db"+str(i)] = 10 * (torch.log10(1 / out["ms_ssim"+str(i)])).mean()
            else:
                out["ms_db"+str(i)] = 0
            out["ms_db"] += out["ms_db"+str(i)]/num_camera
        
        out['loss'] = out['ms_ssim_loss'] + out['bpp_loss']
        return out

class MSE_Loss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        #self.lmbda = lmbda

    def forward(self, output, target, lmbda):
        target1, target2 = target[0], target[1]
        N, _, H, W = target1.size()
        out = {}
        num_pixels = N * H * W

        # 计算误差
        out['bpp0'] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output['likelihoods'][0].values())
        out['bpp1'] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output['likelihoods'][1].values())        
        out["bpp_loss"] = (out['bpp0'] + out['bpp1'])/2
        out["mse0"] = self.mse(output['x_hat'][0], target1)
        out["mse1"] = self.mse(output['x_hat'][1], target2)
        
        if isinstance(lmbda, list):
            out['mse_loss'] = (lmbda[0] * out["mse0"] + lmbda[1] * out["mse1"])/2 
        else:
            out['mse_loss'] = lmbda * (out["mse0"] + out["mse1"])/2        #end to end
        out['loss'] = out['mse_loss'] + out['bpp_loss']

        return out

class MS_SSIM_Loss(nn.Module):
    def __init__(self, device, size_average=True, max_val=1):
        super().__init__()
        self.ms_ssim = MS_SSIM(size_average, max_val).to(device)
        #self.lmbda = lmbda

    def forward(self, output, target, lmbda):
        target1, target2 = target[0], target[1]
        N, _, H, W = target1.size()
        out = {}
        num_pixels = N * H * W

        # 计算误差
        out['bpp0'] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output['likelihoods'][0].values())
        out['bpp1'] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output['likelihoods'][1].values())        
        out["bpp_loss"] = (out['bpp0'] + out['bpp1'])/2

        out["ms_ssim0"] = 1 - self.ms_ssim(output['x_hat'][0], target1)
        out["ms_ssim1"] = 1- self.ms_ssim(output['x_hat'][1], target2)
 
        out['ms_ssim_loss'] = (out["ms_ssim0"] + out["ms_ssim1"])/2        #end to end
        out['loss'] = lmbda * out['ms_ssim_loss'] + out['bpp_loss']
        return out


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, sigma, channel):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(
        channel, 1, window_size, window_size).contiguous())
    return window

class MS_SSIM(nn.Module):
    def __init__(self, size_average=True, max_val=255, device_id=0):
        super(MS_SSIM, self).__init__()
        self.size_average = size_average
        self.channel = 3
        self.max_val = max_val
        self.device_id = device_id

    def _ssim(self, img1, img2):

        _, c, w, h = img1.size()
        window_size = min(w, h, 11)
        sigma = 1.5 * window_size / 11

        window = create_window(window_size, sigma, self.channel)
        if self.device_id != None:
            window = window.cuda(self.device_id)

        mu1 = F.conv2d(img1, window, padding=window_size //
                       2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=window_size //
                       2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(
            img1*img1, window, padding=window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(
            img2*img2, window, padding=window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size //
                           2, groups=self.channel) - mu1_mu2

        C1 = (0.01*self.max_val)**2
        C2 = (0.03*self.max_val)**2
        V1 = 2.0 * sigma12 + C2
        V2 = sigma1_sq + sigma2_sq + C2
        ssim_map = ((2*mu1_mu2 + C1)*V1)/((mu1_sq + mu2_sq + C1)*V2)
        mcs_map = V1 / V2
        if self.size_average:
            return ssim_map.mean(), mcs_map.mean()

    def ms_ssim(self, img1, img2, levels=5):

        weight = Variable(torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]))
        msssim=Variable(torch.Tensor(levels,))
        mcs=Variable(torch.Tensor(levels,))
        # if self.device_id != None:
        #     weight = weight.cuda(self.device_id)
        #     weight = msssim.cuda(self.device_id)
        #     weight = mcs.cuda(self.device_id)
        #     print(weight.device)

        for i in range(levels):
            ssim_map, mcs_map=self._ssim(img1, img2)
            msssim[i]=ssim_map
            mcs[i]=mcs_map
            filtered_im1=F.avg_pool2d(img1, kernel_size=2, stride=2)
            filtered_im2=F.avg_pool2d(img2, kernel_size=2, stride=2)
            img1=filtered_im1
            img2=filtered_im2

        value=(torch.prod(mcs[0:levels-1]**weight[0:levels-1]) *
                                    (msssim[levels-1]**weight[levels-1]))
        return value


    def forward(self, img1, img2, levels=5):
        return self.ms_ssim(img1, img2, levels)



def compute_aux_loss(aux_list: List, backward=False):
    aux_loss_sum = 0
    for aux_loss in aux_list:
        aux_loss_sum += aux_loss
        if backward is True:
            aux_loss.backward()

    return aux_loss_sum

def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(p for p in net.named_parameters() if p[1].requires_grad)
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )

    aux_optimizer = optim.Adam(
            (params_dict[n] for n in sorted(aux_parameters)),
            lr=args.learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, args):
    model.train()
    device = next(model.parameters()).device
    print(device)
    if args.metric == "mse":
        metric_dB_name, left_db_name, right_db_name = 'psnr', "left_PSNR", "right_PSNR"
        metric_name = "mse_loss" 
    else:
        metric_dB_name, left_db_name, right_db_name = "ms_db", "left_ms_db", "right_ms_db"
        metric_name = "ms_ssim_loss"

    metric_dB = AverageMeter(metric_dB_name, ':.4e')
    metric_loss = AverageMeter(args.metric, ':.4e')  
    left_db, right_db = AverageMeter(left_db_name, ':.4e'), AverageMeter(right_db_name, ':.4e')
    metric0, metric1 = args.metric+"0", args.metric+"1"

    loss = AverageMeter('Loss', ':.4e')
    bpp_loss = AverageMeter('BppLoss', ':.4e')
    aux_loss = AverageMeter('AuxLoss', ':.4e')
    left_bpp, right_bpp = AverageMeter('LBpp', ':.4e'), AverageMeter('RBpp', ':.4e')

    train_dataloader = tqdm(train_dataloader)
    print('Train epoch:', epoch)
    for i, batch in enumerate(train_dataloader):
        d = [frame.to(device) for frame in batch]
        optimizer.zero_grad()
        if aux_optimizer is not None:
            aux_optimizer.zero_grad()
        #aux_optimizer.zero_grad()
        out_net = model(d)
        out_criterion = criterion(out_net, d, args.lmbda)

        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        if aux_optimizer is not None:
            out_aux_loss = compute_aux_loss(model.aux_loss(), backward=True)
            aux_optimizer.step()
        else:
            out_aux_loss = compute_aux_loss(model.aux_loss(), backward=False)
        #out_aux_loss = compute_aux_loss(model.aux_loss(), backward=True)
        #aux_optimizer.step()

        loss.update(out_criterion["loss"].item())
        bpp_loss.update((out_criterion["bpp_loss"]).item())
        aux_loss.update(out_aux_loss.item())
        metric_loss.update(out_criterion[metric_name].item())
        
        left_bpp.update(out_criterion["bpp0"].item())
        right_bpp.update(out_criterion["bpp1"].item())
        if out_criterion[metric0] > 0 and out_criterion[metric1] > 0:
            left_metric = 10 * (torch.log10(1 / out_criterion[metric0])).mean().item()
            right_metric = 10 * (torch.log10(1 / out_criterion[metric1])).mean().item()
            left_db.update(left_metric)
            right_db.update(right_metric)
            metric_dB.update((left_metric+right_metric)/2)
        train_dataloader.set_description('[{}/{}]'.format(i, len(train_dataloader)))
        train_dataloader.set_postfix({"Loss":loss.avg, 'Bpp':bpp_loss.avg, args.metric: metric_loss.avg, 'Aux':aux_loss.avg,
            metric_dB_name:metric_dB.avg})

    out = {"loss": loss.avg, metric_name: metric_loss.avg, "bpp_loss": bpp_loss.avg, 
            "aux_loss":aux_loss.avg, metric_dB_name: metric_dB.avg, "left_bpp": left_bpp.avg, "right_bpp": right_bpp.avg,
            left_db_name:left_db.avg, right_db_name: right_db.avg,}
    print(out)
    return out

def test_epoch(epoch, val_dataloader, model, criterion, args):
    model.eval()
    device = next(model.parameters()).device

    if args.metric == "mse":
        metric_dB_name, left_db_name, right_db_name = 'psnr', "left_PSNR", "right_PSNR"
        metric_name = "mse_loss" 
    else:
        metric_dB_name, left_db_name, right_db_name = "ms_db", "left_ms_db", "right_ms_db"
        metric_name = "ms_ssim_loss"

    metric_dB = AverageMeter(metric_dB_name, ':.4e')
    metric_loss = AverageMeter(args.metric, ':.4e')  
    left_db, right_db = AverageMeter(left_db_name, ':.4e'), AverageMeter(right_db_name, ':.4e')
    metric0, metric1 = args.metric+"0", args.metric+"1"

    loss = AverageMeter('Loss', ':.4e')
    bpp_loss = AverageMeter('BppLoss', ':.4e')
    aux_loss = AverageMeter('AuxLoss', ':.4e')
    left_bpp, right_bpp = AverageMeter('LBpp', ':.4e'), AverageMeter('RBpp', ':.4e')
    loop = tqdm(val_dataloader)

    with torch.no_grad():
        for i, batch in enumerate(loop):
            d = [frame.to(device) for frame in batch]
            
            out_net = model(d)
            out_criterion = criterion(out_net, d, args.lmbda)

            out_aux_loss = compute_aux_loss(model.aux_loss(), backward=False)

            loss.update(out_criterion["loss"].item())
            bpp_loss.update((out_criterion["bpp_loss"]).item())
            aux_loss.update(out_aux_loss.item())
            metric_loss.update(out_criterion[metric_name].item())
        
            left_bpp.update(out_criterion["bpp0"].item())
            right_bpp.update(out_criterion["bpp1"].item())

            if out_criterion[metric0] > 0 and out_criterion[metric1] > 0:
                left_metric = 10 * (torch.log10(1 / out_criterion[metric0])).mean().item()
                right_metric = 10 * (torch.log10(1 / out_criterion[metric1])).mean().item()
                left_db.update(left_metric)
                right_db.update(right_metric)
                metric_dB.update((left_metric+right_metric)/2)

            loop.set_description('[{}/{}]'.format(i, len(val_dataloader)))
            loop.set_postfix({"Loss":loss.avg, 'Bpp':bpp_loss.avg, args.metric: metric_loss.avg, 'Aux':aux_loss.avg,
                metric_dB_name:metric_dB.avg})

    out = {"loss": loss.avg, metric_name: metric_loss.avg, "bpp_loss": bpp_loss.avg, 
            "aux_loss":aux_loss.avg, metric_dB_name: metric_dB.avg, "left_bpp": left_bpp.avg, "right_bpp": right_bpp.avg,
            left_db_name:left_db.avg, right_db_name: right_db.avg,}

    return out


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, default='./datasets/Instereo2K/', help="Training dataset"
    )
    parser.add_argument(
        "--data-name", type=str, default='instereo2K', help="Training dataset"
    )
    parser.add_argument(
        "--model-name", type=str, default='LDMIC', help="Training dataset"
    )

    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=2,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=2048,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=3, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--epochs", type=int, default=400, help="number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", help="Save model to disk"
    )
    parser.add_argument(
        "--resize", action="store_true", help="training use resize or randomcrop"
    )
    parser.add_argument(
        "--seed", type=float, default=1, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--i_model_path", type=str, help="Path to a checkpoint")
    parser.add_argument("--metric", type=str, default="mse", help="metric: mse, ms-ssim")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    # Warning, the order of the transform composition should be kept.
    train_dataset = StereoImageDataset(ds_type='train', ds_name=args.data_name, root=args.dataset, crop_size=args.patch_size, resize=args.resize)
    test_dataset = StereoImageDataset(ds_type='test', ds_name=args.data_name, root=args.dataset, crop_size=args.patch_size, resize=args.resize)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, 
        shuffle=True, pin_memory=(device == "cuda"))
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=(device == "cuda"))

    if args.model_name == "LDMIC":
        net = LDMIC(N=192, M=192, decode_atten=JointContextTransfer)
    elif args.model_name == "LDMIC_checkboard":
        net = LDMIC_checkboard(N=192, M=192, decode_atten=JointContextTransfer)

    net = net.to(device) 

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100, 200, 300, 400], 0.5) #optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10, factor=0.2)
    if args.metric == "mse":
        criterion = MSE_Loss() #MSE_Loss(lmbda=args.lmbda)
    else:
        criterion = MS_SSIM_Loss(device) #(device, lmbda=args.lmbda)
    last_epoch = 0
    best_loss = float("inf")

    if args.i_model_path:  #load from previous checkpoint
        print("Loading model: ", args.i_model_path)
        checkpoint = torch.load(args.i_model_path, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])   
        last_epoch = checkpoint["epoch"] + 1
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        best_b_model_path = os.path.join(os.path.split(args.i_model_path)[0], 'ckpt.best.pth.tar')
        best_loss = torch.load(best_b_model_path)["loss"]


    log_dir, experiment_id = get_output_folder('./checkpoints/{}/{}/{}/lamda{}/'.format(args.data_name, args.metric, args.model_name, int(args.lmbda)), 'train')
    display_name = "{}_{}_lmbda{}".format(args.model_name, args.metric, int(args.lmbda))
    tags = "lmbda{}".format(args.lmbda)

    with open(os.path.join(log_dir, 'args.yaml'), 'w') as f:
        f.write(args_text)

    project_name = "DSIC_" + args.data_name
    wandb.init(project=project_name, name=display_name, tags=[tags],) #notes="lmbda{}".format(args.lmbda))
    wandb.watch_called = False  # Re-run the model without restarting the runtime, unnecessary after our next release
    wandb.config.update(args) # config is a variable that holds and saves hyper parameters and inputs
  
    if args.metric == "mse":
        metric_dB_name, left_db_name, right_db_name = 'psnr', "left_PSNR", "right_PSNR"
        metric_name = "mse_loss" 
    else:
        metric_dB_name, left_db_name, right_db_name = "ms_db", "left_ms_db", "right_ms_db"
        metric_name = "ms_ssim_loss"

    #val_loss = test_epoch(0, test_dataloader, net, criterion, args)
    for epoch in range(last_epoch, args.epochs):
        #adjust_learning_rate(optimizer, aux_optimizer, epoch, args)
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_loss = train_one_epoch(net, criterion, train_dataloader, optimizer, aux_optimizer, epoch, args.clip_max_norm, args)
        print(train_loss)
        lr_scheduler.step()

        wandb.log({"train": {"loss": train_loss["loss"], metric_name: train_loss[metric_name], "bpp_loss": train_loss["bpp_loss"],
            "aux_loss": train_loss["aux_loss"], metric_dB_name: train_loss[metric_dB_name], "left_bpp": train_loss["left_bpp"], "right_bpp": train_loss["right_bpp"],
            left_db_name:train_loss[left_db_name], right_db_name: train_loss[right_db_name]}, }
        )
        if epoch%10==0:
            val_loss = test_epoch(epoch, test_dataloader, net, criterion, args)
            wandb.log({ 
                "test": {"loss": val_loss["loss"], metric_name: val_loss[metric_name], "bpp_loss": val_loss["bpp_loss"],
                "aux_loss": val_loss["aux_loss"], metric_dB_name: val_loss[metric_dB_name], "left_bpp": val_loss["left_bpp"], "right_bpp": val_loss["right_bpp"],
                left_db_name:val_loss[left_db_name], right_db_name: val_loss[right_db_name],}
                })
        
            loss = val_loss["loss"]
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
        else:
            loss = best_loss
            is_best = False
        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict() if aux_optimizer is not None else None,
                    'lr_scheduler': lr_scheduler.state_dict(),
                },
                is_best, log_dir
            )

if __name__ == "__main__":
    main(sys.argv[1:])
