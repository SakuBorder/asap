import torch
import numpy as np
from humanoidverse.envs.motion_tracking.motion_tracking import LeggedRobotMotionTracking
from loguru import logger

class AMPMotionTracking(LeggedRobotMotionTracking):
    def __init__(self, config, device):
        super().__init__(config, device)
        self._init_amp_data()

    def _init_amp_data(self):
        """初始化AMP相关数据"""
        # 加载专家演示数据
        self.expert_amp_loader = self._load_expert_amp_data()
        logger.info(f"Loaded {len(self.expert_amp_loader)} expert AMP observations")
        
        # AMP观测缓冲区
        # import ipdb;ipdb.set_trace()
        self.amp_obs_buf = torch.zeros(
            self.num_envs, 
            self.config.robot.algo_obs_dim_dict["amp_obs"], 
            device=self.device
        )

    def _load_expert_amp_data(self):
        """加载专家AMP观测数据"""
        expert_states = []
        # import ipdb;ipdb.set_trace()
        # 设置AMP数据采样参数
        sample_dt = self.dt * 2  # 每2个时间步采样一次，减少数据量
        min_length = 1.0  # 最小动作长度（秒）
        
        logger.info("Loading expert AMP data from motion library...")
        
        # 遍历所有motion数据
        for motion_id in range(min(self._motion_lib._num_unique_motions, 100)):  # 限制数量避免内存问题
            motion_length = self._motion_lib.get_motion_length([motion_id]).item()
            
            if motion_length < min_length:
                continue
                
            num_frames = int((motion_length - sample_dt) / sample_dt)
            
            for frame in range(num_frames):
                time = frame * sample_dt
                try:
                    # motion_state = self._motion_lib.get_motion_state([motion_id], [time], offset=torch.zeros(1, 3, device=self.device))
                    motion_state = self._motion_lib.get_motion_state(
                        torch.tensor([motion_id], device=self.device),  # 确保 motion_id 是张量
                        torch.tensor([time], device=self.device),       # 确保 time 是张量
                        offset=torch.zeros(1, 3, device=self.device)
                    )
                    # 构建AMP观测
                    amp_obs = self._build_amp_obs_from_state(motion_state)
                    expert_states.append(amp_obs)
                    
                except Exception as e:
                    logger.warning(f"Error processing motion {motion_id}, frame {frame}: {e}")
                    continue
        
        if len(expert_states) == 0:
            logger.error("No expert AMP observations loaded!")
            raise RuntimeError("Failed to load expert AMP data")
            
        return torch.stack(expert_states)

    def _build_amp_obs_from_state(self, motion_state):
        """从motion状态构建AMP观测"""
        # 根据DeepMimic/AMP论文，AMP观测通常包括：
        # - 关节位置和速度
        # - 根节点线速度和角速度  
        # - 可选：关键身体部位的局部位置
        
        dof_pos = motion_state["dof_pos"][0]  # [num_dofs]
        dof_vel = motion_state["dof_vel"][0]  # [num_dofs] 
        root_vel = motion_state["root_vel"][0]  # [3]
        root_ang_vel = motion_state["root_ang_vel"][0]  # [3]
        
        # 可以添加更多特征，如身体关键点相对位置
        # body_pos = motion_state["rg_pos_t"][0]  # [num_bodies, 3]
        # root_pos = motion_state["root_pos"][0:1]  # [3]
        # relative_body_pos = (body_pos - root_pos).flatten()  # 相对位置
        
        amp_obs = torch.cat([
            dof_pos,
            dof_vel, 
            root_vel,
            root_ang_vel,
            # relative_body_pos  # 可选：添加身体相对位置
        ])
        
        return amp_obs

    def get_expert_amp_observations(self, num_samples=None):
        """获取专家AMP观测"""
        if num_samples is None:
            num_samples = self.num_envs
            
        # 随机采样专家观测
        if len(self.expert_amp_loader) < num_samples:
            # 如果专家数据不足，重复采样
            indices = torch.randint(0, len(self.expert_amp_loader), (num_samples,))
        else:
            indices = torch.randperm(len(self.expert_amp_loader))[:num_samples]
            
        return self.expert_amp_loader[indices].to(self.device)

    def _compute_amp_observations(self):
        """计算当前状态的AMP观测"""
        # 构建当前的AMP观测，保持与expert数据相同的格式
        dof_pos = self.simulator.dof_pos
        dof_vel = self.simulator.dof_vel
        root_vel = self.base_lin_vel
        root_ang_vel = self.base_ang_vel
        
        # 可选：添加身体相对位置特征
        # root_pos = self.simulator.robot_root_states[:, :3]
        # body_pos = self.simulator._rigid_body_pos
        # relative_body_pos = (body_pos - root_pos.unsqueeze(1)).flatten(1)
        amp_obs = torch.cat([
            torch.as_tensor(dof_pos, device=self.device),
            torch.as_tensor(dof_vel, device=self.device),
            torch.as_tensor(root_vel, device=self.device),
            torch.as_tensor(root_ang_vel, device=self.device),  # ✅ 添加这一行
        ], dim=-1)
        
        self.amp_obs_buf[:] = amp_obs
        return amp_obs

    def _compute_observations(self):
        """重写观测计算，添加AMP观测"""
        super()._compute_observations()
        self._compute_amp_observations()
        

    def _get_obs_amp_obs(self):
        """获取AMP观测，用于obs系统"""
        return self.amp_obs_bufreward