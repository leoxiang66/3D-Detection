import torch
import torch.nn as nn
from .ld_base_v1 import LD_base

class CenterPoint(nn.Module):
    def __init__(self, pc_range:list = None, voxel_size:list = None,feature_map_stride=1):
        super(CenterPoint, self).__init__()
        self.inter_model = LD_base(
            pc_range=pc_range,
            voxel_size=voxel_size,
            feature_map_stride=feature_map_stride
        )
        
    def forward(self,x:torch.Tensor):
        if x.dim() != 3 or x.size()[-1] != 5:
            raise RuntimeError("The dimension of <x> should be [batch_size, #x, 5], where 5 stands for (bacth index, x, y, z, intensity)")
        
        B,_,D = x.size()
        x = x.view(-1,D)
        return self.inter_model(dict(
            batch_size = B,
            points = x
        ))
            
            
            