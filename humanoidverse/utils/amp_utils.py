import torch
import numpy as np
from loguru import logger

class AMPDataProcessor:
    """AMP数据处理工具类"""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.amp_obs_dim = config.robot.algo_obs_dim_dict["amp_obs"]
        
    def extract_amp_features(self, motion_state, include_body_pos=False):
        """从motion状态提取AMP特征"""
        features = []
        
        # 基础特征：关节位置和速度
        if "dof_pos" in motion_state:
            features.append(motion_state["dof_pos"])
        if "dof_vel" in motion_state:
            features.append(motion_state["dof_vel"])
            
        # 根节点速度
        if "root_vel" in motion_state:
            features.append(motion_state["root_vel"])
        if "root_ang_vel" in motion_state:
            features.append(motion_state["root_ang_vel"])
            
        # 可选：身体关键点相对位置
        if include_body_pos and "rg_pos_t" in motion_state and "root_pos" in motion_state:
            body_pos = motion_state["rg_pos_t"]
            root_pos = motion_state["root_pos"]
            relative_pos = (body_pos - root_pos.unsqueeze(1)).flatten(1)
            features.append(relative_pos)
            
        return torch.cat(features, dim=-1) if features else torch.empty(0, device=self.device)
    
    def normalize_amp_obs(self, amp_obs, obs_mean=None, obs_std=None):
        """标准化AMP观测"""
        if obs_mean is not None and obs_std is not None:
            return (amp_obs - obs_mean) / (obs_std + 1e-8)
        return amp_obs
    
    def compute_amp_obs_stats(self, amp_data):
        """计算AMP观测的统计信息"""
        obs_mean = amp_data.mean(dim=0)
        obs_std = amp_data.std(dim=0)
        return obs_mean, obs_std