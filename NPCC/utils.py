import open3d as o3d

# 读取点云文件
point_cloud = o3d.io.read_point_cloud("./dataset/exercise_vox11_organized/sequence_01/exercise_vox11_00000001.ply")

# 获取点的数量
num_points = len(point_cloud.points)
print(f"点云中的点数: {num_points}")