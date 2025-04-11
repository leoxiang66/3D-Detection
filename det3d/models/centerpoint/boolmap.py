import copy
import torch
import torch.nn as nn

class BoolMap(nn.Module):
    '''
             ^ X,W
             |
             |
     Y,H     |
    <--------o
    '''
    def __init__(self, point_cloud_range,voxel_size=[0.2,0.2,0.2], **kwargs):
        super().__init__()
        # VOXEL SIZE
        self.DX, self.DY, self.DZ = \
            voxel_size[0], voxel_size[1],voxel_size[2]
        
        # ROI in meter
        self.m_x_min = point_cloud_range[0]
        self.m_x_max = point_cloud_range[3]

        self.m_y_min = point_cloud_range[1]
        self.m_y_max = point_cloud_range[4]

        self.m_z_min = point_cloud_range[2]
        self.m_z_max = point_cloud_range[5]

        # SIZE of BEV map
        self.BEV_W = round((point_cloud_range[3]-point_cloud_range[0])/self.DX)
        self.BEV_H = round((point_cloud_range[4]-point_cloud_range[1])/self.DY)
        self.BEV_C = round((point_cloud_range[5]-point_cloud_range[2])/self.DZ)

        self.num_bev_features = self.BEV_C
        

    def forward(self, batch_dict):
        """
        将点云数据转换到 BEV 图像中。
        """
        pc_lidar = batch_dict['points'].clone() 
        # 创建 BEV tensor
        bev_img = torch.cuda.BoolTensor(
            batch_dict['batch_size'], self.BEV_C, self.BEV_H, self.BEV_W
        ).fill_(0)
        
        # 将点云 (x, y, z) 转换成 BEV 中的索引（浮点数）
        pc_lidar[:, 1] = ((pc_lidar[:, 1] - self.m_x_min) / self.DX)
        pc_lidar[:, 2] = ((pc_lidar[:, 2] - self.m_y_min) / self.DY)
        pc_lidar[:, 3] = ((pc_lidar[:, 3] - self.m_z_min) / self.DZ)
        
        # 对 x, y, z 的映射结果同时做下界和上界 clmap
        pc_lidar[:,1] = pc_lidar[:,1].clamp(min=0, max=self.BEV_W - 1)
        pc_lidar[:,2] = pc_lidar[:,2].clamp(min=0, max=self.BEV_H - 1)
        pc_lidar[:,3] = pc_lidar[:,3].clamp(min=0, max=self.BEV_C - 1)
        
        # 使用 floor 转换成整数索引
        pc_lidar = pc_lidar.floor().long()
        
        # 利用索引填充 BEV 图像
        bev_img[pc_lidar[:, 0], pc_lidar[:, 3], pc_lidar[:, 2], pc_lidar[:, 1]] = 1
        bev_img = bev_img.float() 
        batch_dict['spatial_features'] = bev_img

        return batch_dict
