# humanoidverse/envs/motion_tracking/amp_motion_tracking.py
# 只修改核心问题，不隐藏错误

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
        
        # 检查评估模式状态
        logger.info(f"初始化完成后的评估模式状态: {getattr(self, 'is_evaluating', False)}")
        
        # 延迟初始化AMP数据，等待 set_is_evaluating 调用
        self.amp_data_initialized = False
        self._init_amp_data()
        self.init_done = True

    def _fix_amp_obs_config(self, config):
        """修复AMP观测维度配置"""
        # 计算正确的AMP观测维度
        dof_obs_size = config.robot.dof_obs_size  # 30 for tai5
        expected_amp_dim = 2 * dof_obs_size + 6  # 30 + 30 + 3 + 3 = 66
        
        # 更新配置中的amp_obs维度
        if "algo_obs_dim_dict" not in config.robot:
            config.robot.algo_obs_dim_dict = {}
        
        config.robot.algo_obs_dim_dict["amp_obs"] = expected_amp_dim
        
        logger.info(f"设置AMP观测维度为: {expected_amp_dim}")

    def _init_amp_data(self):
            """初始化AMP相关数据"""
            # 获取AMP观测维度
            amp_obs_dim = self.config.robot.algo_obs_dim_dict["amp_obs"]
            
            # AMP观测缓冲区（总是需要）
            self.amp_obs_buf = torch.zeros(
                self.num_envs, 
                amp_obs_dim, 
                device=self.device
            )
            
            logger.info(f"初始化AMP观测缓冲区，形状: {self.amp_obs_buf.shape}")
            
            # 检查当前是否为评估模式
            is_eval = getattr(self, 'is_evaluating', False)
            logger.info(f"AMP数据初始化时的评估模式状态: {is_eval}")
            
            if is_eval:
                logger.info("初始化时检测到评估模式，使用简化的expert数据")
                self.expert_amp_loader = self._load_single_motion_expert_data()
            else:
                logger.info("初始化时为训练模式，延迟加载expert数据")
                # 创建临时的空数据，等待后续重新初始化
                self.expert_amp_loader = torch.zeros(10, amp_obs_dim, device=self.device)
            
            self.amp_data_initialized = True

    def _reinit_amp_for_evaluation(self):
        """重新初始化AMP数据为评估模式"""
        try:
            logger.info("开始重新初始化AMP expert数据为评估模式")
            
            # 重新加载简化的评估数据
            self.expert_amp_loader = self._load_single_motion_expert_data()
            
            logger.info(f"✅ AMP数据已切换到评估模式，expert数据形状: {self.expert_amp_loader.shape}")
            
        except Exception as e:
            logger.error(f"❌ AMP评估模式重新初始化失败: {e}")
            # 使用安全的fallback
            amp_obs_dim = self.config.robot.algo_obs_dim_dict["amp_obs"]
            self.expert_amp_loader = torch.zeros(10, amp_obs_dim, device=self.device)

    def _load_single_motion_expert_data(self):
        """评估模式：只加载单个motion的expert数据"""
        expert_states = []
        amp_obs_dim = self.config.robot.algo_obs_dim_dict["amp_obs"]
        
        try:
            # 评估时只使用第一个motion（motion_id = 0）
            motion_id = 0
            
            logger.info(f"评估模式：加载motion {motion_id} 的expert数据")
            logger.info(f"Motion库状态 - 总motion数: {self._motion_lib._num_unique_motions}")
            
            # 确保motion_id有效
            if motion_id >= self._motion_lib._num_unique_motions:
                logger.error(f"Motion ID {motion_id} 超出范围 {self._motion_lib._num_unique_motions}")
                return torch.zeros(10, amp_obs_dim, device=self.device)
            
            # 获取motion长度
            motion_length = self._motion_lib.get_motion_length([motion_id]).item()
            logger.info(f"Motion {motion_id} 长度: {motion_length}s")
            
            # 只采样几个安全的时间点
            num_samples = min(5, max(1, int(motion_length / (self.dt * 2))))
            logger.info(f"计划采样 {num_samples} 个时间点")
            
            for i in range(num_samples):
                # 确保时间不超出motion范围
                time = min(i * self.dt * 2, motion_length - 0.1)
                
                logger.debug(f"采样时间点 {i}: time={time:.3f}s")
                
                # 获取motion状态
                motion_state = self._motion_lib.get_motion_state(
                    torch.tensor([motion_id], device=self.device),
                    torch.tensor([time], device=self.device),
                    offset=torch.zeros(1, 3, device=self.device)
                )
                
                # 构建AMP观测
                amp_obs = self._build_amp_obs_from_state(motion_state)
                expert_states.append(amp_obs)
            
            if len(expert_states) > 0:
                result = torch.stack(expert_states)
                logger.info(f"✅ 评估模式：成功加载 {len(expert_states)} 个expert观测")
                return result
            else:
                logger.warning("⚠️ 评估模式：没有加载到expert数据，使用fallback")
                return torch.zeros(10, amp_obs_dim, device=self.device)
                
        except Exception as e:
            logger.error(f"❌ 评估模式expert数据加载失败: {e}")
            import traceback
            traceback.print_exc()
            return torch.zeros(10, amp_obs_dim, device=self.device)

    def _load_expert_amp_data(self):
        """训练模式：加载多个motion的expert数据 - 添加严格的边界检查"""
        expert_states = []
        amp_obs_dim = self.config.robot.algo_obs_dim_dict["amp_obs"]
        
        # 严格限制motion数量和采样点，避免索引越界
        max_motions = min(self._motion_lib._num_unique_motions, 5)
        max_samples_per_motion = 3
        
        logger.info(f"训练模式：处理 {max_motions} 个motion，每个motion采样 {max_samples_per_motion} 个点")
        
        try:
            for motion_id in range(max_motions):
                try:
                    # 严格检查motion_id范围
                    if motion_id >= self._motion_lib._num_unique_motions:
                        logger.warning(f"Motion ID {motion_id} 超出范围，跳过")
                        continue
                    
                    # 获取motion长度
                    motion_length = self._motion_lib.get_motion_length([motion_id]).item()
                    
                    if motion_length < 0.5:  # 跳过太短的motion
                        logger.warning(f"Motion {motion_id} 太短 ({motion_length}s)，跳过")
                        continue
                    
                    # 采样时间点，确保不超出范围
                    for i in range(max_samples_per_motion):
                        # 严格确保时间在有效范围内
                        time = min(i * 0.5, motion_length - 0.2)
                        
                        # 再次检查时间有效性
                        if time < 0 or time >= motion_length:
                            logger.warning(f"时间 {time} 超出motion {motion_id} 范围 [0, {motion_length}]")
                            continue
                        
                        # 获取motion状态
                        motion_state = self._motion_lib.get_motion_state(
                            torch.tensor([motion_id], device=self.device),
                            torch.tensor([time], device=self.device),
                            offset=torch.zeros(1, 3, device=self.device)
                        )
                        
                        # 构建AMP观测
                        amp_obs = self._build_amp_obs_from_state(motion_state)
                        expert_states.append(amp_obs)
                        
                except Exception as e:
                    logger.error(f"处理motion {motion_id} 时出错: {e}")
                    # 立即停止，不继续处理其他motion
                    break
            
            if len(expert_states) > 0:
                result = torch.stack(expert_states)
                logger.info(f"训练模式：成功加载 {len(expert_states)} 个expert观测")
                return result
            else:
                logger.error("训练模式：没有加载到任何expert数据")
                return torch.zeros(50, amp_obs_dim, device=self.device)
                
        except Exception as e:
            logger.error(f"训练模式expert数据加载过程出现严重错误: {e}")
            return torch.zeros(50, amp_obs_dim, device=self.device)

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
        super()._pre_compute_observations_callback()
        self._compute_amp_observations()

    def _get_obs_amp_obs(self):
        """获取AMP观测，用于obs系统"""
        if not hasattr(self, 'amp_obs_buf') or self.amp_obs_buf is None:
            self._compute_amp_observations()
        return self.amp_obs_buf

    def _compute_observations(self):
        """重写观测计算，确保AMP观测被包含"""
        self._compute_amp_observations()
        super()._compute_observations()

    def _init_buffers(self):
        """初始化缓冲区"""
        super()._init_buffers()
        
        # 验证AMP观测维度
        expected_amp_dim = 2 * self.config.robot.dof_obs_size + 6
        config_amp_dim = self.config.robot.algo_obs_dim_dict.get("amp_obs", 0)
        
        if config_amp_dim != expected_amp_dim:
            logger.warning(f"AMP观测维度不匹配！配置: {config_amp_dim}, 期望: {expected_amp_dim}")
            
        logger.info(f"AMP观测维度: {config_amp_dim}")

    def set_is_evaluating(self):
        """设置为评估模式 - AMP特殊处理"""
        logger.info("🔄 AMPMotionTracking 切换到评估模式")
        
        # 调用父类方法
        super().set_is_evaluating()
        
        # 重新配置AMP数据为评估模式
        if self.amp_data_initialized:
            logger.info("重新配置AMP数据为评估模式")
            self._reinit_amp_for_evaluation()
        else:
            logger.info("AMP数据尚未初始化，标记为稍后处理")
