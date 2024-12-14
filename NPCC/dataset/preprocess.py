import open3d as o3d
import numpy as np
from scipy.spatial import KDTree


def random_sampling(points, num_samples):
    """
    随机采样点云点。
    """
    idx = np.random.choice(len(points), num_samples, replace=False)
    return points[idx]


def approximate_curvature(local_points):
    """
    计算局部点云的曲率近似值。
    """
    centroid = np.mean(local_points, axis=0)
    centered_points = local_points - centroid
    covariance = np.cov(centered_points.T)
    eigenvalues, _ = np.linalg.eig(covariance)

    return eigenvalues.min() / eigenvalues.max()


def curvature_based_sampling_optimized(points, num_samples, random_sample_ratio=0.05, k=5):
    """
    使用优化的曲率采样从点云中选取点。

    参数:
    - points: 点云点坐标 (numpy 数组)。
    - num_samples: 目标采样点数。
    - random_sample_ratio: 初步随机采样比例。
    - k: 每个点的邻域点数，用于计算曲率。

    返回:
    - 采样后的点坐标 (numpy 数组)。
    """
    # 初步随机采样
    reduced_points = random_sampling(points, int(len(points) * random_sample_ratio))

    # 使用 KDTree 查询最近邻点
    kdtree = KDTree(reduced_points)
    curvatures = []
    for i in range(len(reduced_points)):
        _, idx = kdtree.query(reduced_points[i], k=k)
        local_points = reduced_points[idx]
        curvature = approximate_curvature(local_points)
        curvatures.append(curvature)

    # 根据曲率选取最高的点
    curvatures = np.array(curvatures)
    sampled_idx = np.argsort(-curvatures)[:num_samples]
    return reduced_points[sampled_idx]


def voxel_downsample(points, voxel_size=0.01):
    """
    使用体素下采样减少点云点数。

    参数:
    - points: 点云点坐标 (numpy 数组)。
    - voxel_size: 体素大小，单位为点云坐标系单位，越大降采样越多。

    返回:
    - 下采样后的点云点坐标 (numpy 数组)。
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(downsampled_pcd.points)


def load_and_normalize_point_cloud(file_path, model_height=1.8, voxel_resolution=1024, num_points=1024,
                                   random_sample_ratio=0.05, use_voxel_downsample=True, voxel_size=0.01):
    """
    加载点云文件并进行归一化和采样。

    参数:
    - file_path: 点云文件路径 (如 .ply 文件)。
    - model_height: 模型的实际高度，单位为米，默认 1.8 米。
    - voxel_resolution: 体素分辨率 (通常为 1024)。
    - num_points: 固定点数，将点云采样到该点数。
    - random_sample_ratio: 曲率采样初步随机采样比例。
    - use_voxel_downsample: 是否启用体素下采样。
    - voxel_size: 体素下采样的体素大小。

    返回:
    - pcd: 归一化后的点云对象 (open3d.geometry.PointCloud)。
    - processed_points: 处理后的点坐标 (numpy 数组)。
    """
    try:
        # 加载点云文件
        pcd = o3d.io.read_point_cloud(file_path)

        # 检查点云是否为空
        if len(pcd.points) == 0:
            raise ValueError(f"加载的点云为空: {file_path}")

        # 提取点云点坐标
        points = np.asarray(pcd.points)

        # 去中心化：将点云的质心移动到原点
        centroid = np.mean(points, axis=0)
        points -= centroid

        # 缩放到单位球
        max_distance = np.max(np.linalg.norm(points, axis=1))
        points /= max_distance

        # 按比例缩放到模型高度和体素分辨率
        # scale_factor = model_height / voxel_resolution
        # points *= scale_factor

        # 体素下采样
        if use_voxel_downsample:
            points = voxel_downsample(points, voxel_size=voxel_size)

        # 曲率采样
        if len(points) > num_points:
            points = curvature_based_sampling_optimized(points, num_samples=num_points,
                                                        random_sample_ratio=random_sample_ratio)

        # 更新点云对象
        pcd.points = o3d.utility.Vector3dVector(points)

        return pcd, points

    except Exception as e:
        raise RuntimeError(f"加载点云文件失败: {file_path}, 错误信息: {e}")


# 示例用法
if __name__ == "__main__":
    # 点云文件路径
    file_path = "your_point_cloud_file.ply"

    # 加载并处理点云
    processed_pcd, processed_points = load_and_normalize_point_cloud(
        file_path,
        voxel_resolution=1024,
        num_points=1024,
        random_sample_ratio=0.05,
        use_voxel_downsample=True,
        voxel_size=0.02  # 根据需要调整
    )

    # 保存处理后的点云
    o3d.io.write_point_cloud("processed_point_cloud.ply", processed_pcd)
    print(f"处理完成的点数: {len(processed_points)}")
