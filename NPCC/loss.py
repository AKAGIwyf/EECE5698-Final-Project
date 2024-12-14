import torch

def chamfer_loss(pred, target):
    """
    Chamfer 距离的实现，用于点云重构任务。
    :param pred: 预测点云, shape=(B, N, 3)
    :param target: 目标点云, shape=(B, M, 3)
    :return: Chamfer 距离
    """
    B, N, _ = pred.size()
    B, M, _ = target.size()

    # 计算每个点到另一组点的最近距离
    pred_dist = torch.cdist(pred, target, p=2)
    target_dist = torch.cdist(target, pred, p=2)

    # 平均最小距离
    loss = torch.mean(torch.min(pred_dist, dim=-1)[0]) + torch.mean(torch.min(target_dist, dim=-1)[0])
    return loss


