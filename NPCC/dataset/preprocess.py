import open3d as o3d
import numpy as np
from scipy.spatial import KDTree


def random_sampling(points, num_samples):
    """
    Randomly sample points from the point cloud.
    """
    idx = np.random.choice(len(points), num_samples, replace=False)
    return points[idx]


def approximate_curvature(local_points):
    """
    Compute an approximate curvature value for a local point cloud.
    """
    centroid = np.mean(local_points, axis=0)
    centered_points = local_points - centroid
    covariance = np.cov(centered_points.T)
    eigenvalues, _ = np.linalg.eig(covariance)

    return eigenvalues.min() / eigenvalues.max()


def curvature_based_sampling_optimized(points, num_samples, random_sample_ratio=0.05, k=5):
    """
    Perform optimized curvature-based sampling from a point cloud.

    Parameters:
    - points: Point cloud coordinates (numpy array).
    - num_samples: Target number of sampled points.
    - random_sample_ratio: Initial random sampling ratio.
    - k: Number of neighbors to consider for curvature computation.

    Returns:
    - Sampled point coordinates (numpy array).
    """
    # Initial random sampling
    reduced_points = random_sampling(points, int(len(points) * random_sample_ratio))

    # Use KDTree to query nearest neighbors
    kdtree = KDTree(reduced_points)
    curvatures = []
    for i in range(len(reduced_points)):
        _, idx = kdtree.query(reduced_points[i], k=k)
        local_points = reduced_points[idx]
        curvature = approximate_curvature(local_points)
        curvatures.append(curvature)

    # Select points with the highest curvature
    curvatures = np.array(curvatures)
    sampled_idx = np.argsort(-curvatures)[:num_samples]
    return reduced_points[sampled_idx]


def voxel_downsample(points, voxel_size=0.01):
    """
    Perform voxel downsampling to reduce the number of points in a point cloud.

    Parameters:
    - points: Point cloud coordinates (numpy array).
    - voxel_size: Voxel size in the coordinate units of the point cloud; larger size results in more downsampling.

    Returns:
    - Downsampled point cloud coordinates (numpy array).
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(downsampled_pcd.points)


def load_and_normalize_point_cloud(file_path, model_height=1.8, voxel_resolution=1024, num_points=1024,
                                   random_sample_ratio=0.05, use_voxel_downsample=True, voxel_size=0.01):
    """
    Load a point cloud file and perform normalization and sampling.

    Parameters:
    - file_path: Path to the point cloud file (e.g., .ply file).
    - model_height: Actual height of the model in meters (default: 1.8 meters).
    - voxel_resolution: Voxel resolution (typically 1024).
    - num_points: Fixed number of points to sample from the point cloud.
    - random_sample_ratio: Initial random sampling ratio for curvature-based sampling.
    - use_voxel_downsample: Whether to perform voxel downsampling.
    - voxel_size: Voxel size for downsampling.

    Returns:
    - pcd: Normalized point cloud object (open3d.geometry.PointCloud).
    - processed_points: Processed point coordinates (numpy array).
    """
    try:
        # Load the point cloud file
        pcd = o3d.io.read_point_cloud(file_path)

        # Check if the point cloud is empty
        if len(pcd.points) == 0:
            raise ValueError(f"The loaded point cloud is empty: {file_path}")

        # Extract point cloud coordinates
        points = np.asarray(pcd.points)

        # Decentralize: Move the centroid of the point cloud to the origin
        centroid = np.mean(points, axis=0)
        points -= centroid

        # Scale to unit sphere
        max_distance = np.max(np.linalg.norm(points, axis=1))
        points /= max_distance

        # Perform voxel downsampling
        if use_voxel_downsample:
            points = voxel_downsample(points, voxel_size=voxel_size)

        # Perform curvature-based sampling
        if len(points) > num_points:
            points = curvature_based_sampling_optimized(points, num_samples=num_points,
                                                        random_sample_ratio=random_sample_ratio)

        # Update the point cloud object
        pcd.points = o3d.utility.Vector3dVector(points)

        return pcd, points

    except Exception as e:
        raise RuntimeError(f"Failed to load point cloud file: {file_path}, Error: {e}")
