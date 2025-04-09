import numpy as np
from det3d.models.centerpoint.ld_base_v1 import LD_base
from det3d.types.PointCloud import PointCloud
from det3d.interface.infer import infer_pointcloud_detection
import torch

# 假设你已经有一个点云
dummy_points = np.random.rand(10000, 4).astype(np.float32)  # x, y, z, intensity
pc = PointCloud(dummy_points)

# 初始化模型并加载 checkpoint
model = LD_base()
checkpoint = torch.load("/workspace/git/3D-Detection/checkpoints/livox_model_1.pt", map_location="cpu")
model.load_state_dict({k.replace("module.", ""): v for k, v in checkpoint["model_state_dict"].items()})
model.cuda()

# 进行推理
preds = infer_pointcloud_detection(model, pc)
print(preds)