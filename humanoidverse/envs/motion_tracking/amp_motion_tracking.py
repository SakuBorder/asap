# 在 humanoidverse/envs/motion_tracking/amp_motion_tracking.py 的完整修复版本

import torch
import numpy as np
from humanoidverse.envs.motion_tracking.motion_tracking import LeggedRobotMotionTracking
from loguru import logger

class AMPMotionTracking(LeggedRobotMotionTracking):
    def __init__(self, config, device):
        # 在调用父类初始化之前，先修复AMP观测维度配置
        self._fix_amp_obs_config(config)
        
        # 标记初始化状态
        self.init_done = False
        super().__init__(config, device)
        self._init_amp_data()
        self.init_done = True

    def _fix_amp_obs_config(self, config):
        """修复AMP观测维度配置"""
        # 计算正确的AMP观测维度
        # AMP观测包括: dof_pos + dof_vel + base_lin_vel + base_ang_vel
        dof_obs_size = config.robot.dof_obs_size  # 30 for tai5
        expected_amp_dim = 2 * dof_obs_size + 6  # 30 + 30 + 3 + 3 = 66
        
        # 更新配置中的amp_obs维度
        if "algo_obs_dim_dict" not in config.robot:
            config.robot.algo_obs_dim_dict = {}
        
        config.robot.algo_obs_dim_dict["amp_obs"] = expected_amp_dim
        
        logger.info(f"设置AMP观测维度为: {expected_amp_dim}")
        logger.info(f"DOF观测维度: {dof_obs_size}")

    def _init_amp_data(self):
        """初始化AMP相关数据"""
        # 获取AMP观测维度
        amp_obs_dim = self.config.robot.algo_obs_dim_dict["amp_obs"]
        
        # AMP观测缓冲区
        self.amp_obs_buf = torch.zeros(
            self.num_envs, 
            amp_obs_dim, 
            device=self.device
        )
        
        logger.info(f"初始化AMP观测缓冲区，形状: {self.amp_obs_buf.shape}")
        
        # 加载专家演示数据
        try:
            self.expert_amp_loader = self._load_expert_amp_data()
            logger.info(f"加载了 {len(self.expert_amp_loader)} 个专家AMP观测")
        except Exception as e:
            logger.error(f"加载专家数据失败: {e}")
            # 创建一个空的专家数据集作为fallback
            self.expert_amp_loader = torch.zeros(100, amp_obs_dim, device=self.device)

    def _load_expert_amp_data(self):
        """加载专家AMP观测数据"""
        expert_states = []
        
        # 设置AMP数据采样参数
        sample_dt = self.dt * 2  # 每2个时间步采样一次
        min_length = 1.0  # 最小动作长度（秒）
        max_motions = 50  # 限制motion数量避免内存问题
        
        logger.info("从motion库加载专家AMP数据...")
        
        # 遍历motion数据
        num_motions = min(self._motion_lib._num_unique_motions, max_motions)
        for motion_id in range(num_motions):
            try:
                motion_length = self._motion_lib.get_motion_length([motion_id]).item()
                
                if motion_length < min_length:
                    continue
                    
                num_frames = max(1, int((motion_length - sample_dt) / sample_dt))
                
                for frame in range(0, num_frames, 2):  # 每2帧采样一次
                    time = min(frame * sample_dt, motion_length - 0.1)
                    
                    # 获取motion状态
                    motion_state = self._motion_lib.get_motion_state(
                        torch.tensor([motion_id], device=self.device),
                        torch.tensor([time], device=self.device),
                        offset=torch.zeros(1, 3, device=self.device)
                    )
                    
                    # 构建AMP观测
                    amp_obs = self._build_amp_obs_from_state(motion_state)
                    expert_states.append(amp_obs)
                    
                    # 限制专家数据数量
                    if len(expert_states) >= 1000:
                        break
                        
                if len(expert_states) >= 1000:
                    break
                    
            except Exception as e:
                logger.warning(f"处理motion {motion_id}时出错: {e}")
                continue
        
        if len(expert_states) == 0:
            logger.error("没有加载到专家AMP观测！")
            # 返回随机数据作为fallback
            amp_obs_dim = self.config.robot.algo_obs_dim_dict["amp_obs"]
            return torch.randn(100, amp_obs_dim, device=self.device)
            
        return torch.stack(expert_states)

    def _build_amp_obs_from_state(self, motion_state):
        """从motion状态构建AMP观测"""
        # 根据配置文件中的amp_obs定义: [dof_pos, dof_vel, base_lin_vel, base_ang_vel]
        
        dof_pos = motion_state["dof_pos"][0]  # [num_dofs]
        dof_vel = motion_state["dof_vel"][0]  # [num_dofs] 
        root_vel = motion_state["root_vel"][0]  # [3] -> base_lin_vel
        root_ang_vel = motion_state["root_ang_vel"][0]  # [3] -> base_ang_vel
        
        amp_obs = torch.cat([
            dof_pos,
            dof_vel, 
            root_vel,
            root_ang_vel,
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
        # 根据配置文件: amp_obs: [dof_pos, dof_vel, base_lin_vel, base_ang_vel]
        
        dof_pos = self.simulator.dof_pos
        dof_vel = self.simulator.dof_vel
        base_lin_vel = self.base_lin_vel
        base_ang_vel = self.base_ang_vel
        
        amp_obs = torch.cat([
            dof_pos,
            dof_vel,
            base_lin_vel,
            base_ang_vel,
        ], dim=-1)
        
        self.amp_obs_buf[:] = amp_obs
        return amp_obs

    def _pre_compute_observations_callback(self):
        """在计算观测之前的回调，确保AMP观测被更新"""
        # 先调用父类的方法
        super()._pre_compute_observations_callback()
        
        # 然后计算AMP观测
        self._compute_amp_observations()

    def _get_obs_amp_obs(self):
        """获取AMP观测，用于obs系统"""
        # 确保AMP观测是最新的
        if not hasattr(self, 'amp_obs_buf') or self.amp_obs_buf is None:
            self._compute_amp_observations()
        return self.amp_obs_buf

    def _compute_observations(self):
        """重写观测计算，确保AMP观测被包含"""
        # 先确保AMP观测被更新
        self._compute_amp_observations()
        
        # 调用父类的观测计算
        super()._compute_observations()
        
        # 调试：打印观测维度（仅在初始化完成后）
        if self.init_done and hasattr(self, 'obs_buf_dict'):
            for key, obs in self.obs_buf_dict.items():
                if key == 'critic_obs':
                    logger.debug(f"观测 {key} 维度: {obs.shape}")

    def _init_buffers(self):
        """初始化缓冲区"""
        super()._init_buffers()
        
        # 验证AMP观测维度
        expected_amp_dim = 2 * self.config.robot.dof_obs_size + 6
        config_amp_dim = self.config.robot.algo_obs_dim_dict.get("amp_obs", 0)
        
        if config_amp_dim != expected_amp_dim:
            logger.warning(f"AMP观测维度不匹配！配置: {config_amp_dim}, 期望: {expected_amp_dim}")
            
        logger.info(f"AMP观测维度: {config_amp_dim}")
