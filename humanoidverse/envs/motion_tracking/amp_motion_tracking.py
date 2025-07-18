# humanoidverse/envs/motion_tracking/amp_motion_tracking.py
# 基于标准humanoid AMP实现的完整修改版本

import torch
import numpy as np
from humanoidverse.envs.motion_tracking.motion_tracking import LeggedRobotMotionTracking
from isaac_utils.rotations import (
    my_quat_rotate,
    calc_heading_quat_inv,
    quat_mul,
    quat_conjugate,
    quat_rotate_inverse,
)
from humanoidverse.utils.torch_utils import quat_rotate, normalize
from loguru import logger

def quat_to_tan_norm(q):
    """
    Convert quaternion to tangent-normal form (6D representation)
    Based on the paper "On the Continuity of Rotation Representations in Neural Networks"
    """
    # Normalize quaternion
    q = normalize(q)
    
    # 获取旋转矩阵的前两列作为6D表示
    # q = [x, y, z, w] format
    w, x, y, z = q[..., 3], q[..., 0], q[..., 1], q[..., 2]
    
    # 转换为旋转矩阵的前两列
    # First column of rotation matrix
    col1_x = 1 - 2 * (y*y + z*z)
    col1_y = 2 * (x*y + w*z)
    col1_z = 2 * (x*z - w*y)
    
    # Second column of rotation matrix  
    col2_x = 2 * (x*y - w*z)
    col2_y = 1 - 2 * (x*x + z*z)
    col2_z = 2 * (y*z + w*x)
    
    # 组合成6D表示
    col1 = torch.stack([col1_x, col1_y, col1_z], dim=-1)
    col2 = torch.stack([col2_x, col2_y, col2_z], dim=-1)
    
    return torch.cat([col1, col2], dim=-1)

class AMPMotionTracking(LeggedRobotMotionTracking):
    def __init__(self, config, device):
        # 在调用父类初始化之前，先修复AMP观测维度配置
        self._fix_amp_obs_config(config)
        
        # 标记初始化状态
        self.init_done = False
        super().__init__(config, device)
        
        # 设置关键身体点索引（用于AMP观测）
        self._setup_key_body_ids()
        
        # 检查评估模式状态
        logger.info(f"初始化完成后的评估模式状态: {getattr(self, 'is_evaluating', False)}")
        
        # 延迟初始化AMP数据，等待 set_is_evaluating 调用
        self.amp_data_initialized = False
        self._init_amp_data()
        self.init_done = True

    def _fix_amp_obs_config(self, config):
        """修复AMP观测维度配置 - 参考标准实现"""
        # 计算标准AMP观测维度
        dof_obs_size = config.robot.dof_obs_size  # 30 for tai5
        num_key_bodies = len(config.robot.key_bodies)  # 通常是脚部关键点
        
        # 标准AMP观测包含：root_h(1) + root_rot(6) + root_vel(3) + root_ang_vel(3) + dof_obs + dof_vel + key_body_pos
        expected_amp_dim = 1 + 6 + 3 + 3 + dof_obs_size + dof_obs_size + (3 * num_key_bodies)
        
        # 更新配置中的amp_obs维度
        if "algo_obs_dim_dict" not in config.robot:
            config.robot.algo_obs_dim_dict = {}
        
        config.robot.algo_obs_dim_dict["amp_obs"] = expected_amp_dim
        
        logger.info(f"设置标准AMP观测维度为: {expected_amp_dim}")
        logger.info(f"组成: root_h(1) + root_rot(6) + root_vel(3) + root_ang_vel(3) + dof_obs({dof_obs_size}) + dof_vel({dof_obs_size}) + key_body_pos({3 * num_key_bodies})")

    def _setup_key_body_ids(self):
        """设置关键身体点索引"""
        # 使用脚部作为关键身体点
        self._key_body_ids = self.feet_indices
        logger.info(f"设置关键身体点索引: {self._key_body_ids}")

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
            logger.info("初始化时检测到评估模式，加载expert数据")
            self.expert_amp_loader = self._load_expert_amp_data_for_eval()
        else:
            logger.info("初始化时为训练模式，加载完整expert数据")
            self.expert_amp_loader = self._load_expert_amp_data_for_training()
        
        self.amp_data_initialized = True

    def _reinit_amp_for_evaluation(self):
        """重新初始化AMP数据为评估模式"""
        try:
            logger.info("开始重新初始化AMP expert数据为评估模式")
            
            # 重新加载评估模式的expert数据
            self.expert_amp_loader = self._load_expert_amp_data_for_eval()
            
            logger.info(f"✅ AMP数据已切换到评估模式，expert数据形状: {self.expert_amp_loader.shape}")
            
        except Exception as e:
            logger.error(f"❌ AMP评估模式重新初始化失败: {e}")
            # 使用安全的fallback
            amp_obs_dim = self.config.robot.algo_obs_dim_dict["amp_obs"]
            self.expert_amp_loader = torch.zeros(100, amp_obs_dim, device=self.device)

    def _load_expert_amp_data_for_eval(self):
        """评估模式：加载少量expert数据"""
        expert_states = []
        amp_obs_dim = self.config.robot.algo_obs_dim_dict["amp_obs"]
        
        try:
            # 评估时使用前几个motion
            max_motions = min(3, self._motion_lib._num_unique_motions)
            samples_per_motion = 20
            
            logger.info(f"评估模式：加载 {max_motions} 个motion，每个采样 {samples_per_motion} 个点")
            
            for motion_id in range(max_motions):
                try:
                    motion_length = self._motion_lib.get_motion_length([motion_id]).item()
                    
                    if motion_length < 0.5:
                        continue
                    
                    # 均匀采样
                    for i in range(samples_per_motion):
                        time = (i / max(1, samples_per_motion - 1)) * (motion_length - 0.1)
                        
                        motion_state = self._motion_lib.get_motion_state(
                            torch.tensor([motion_id], device=self.device),
                            torch.tensor([time], device=self.device),
                            offset=torch.zeros(1, 3, device=self.device)
                        )
                        
                        amp_obs = self._build_amp_obs_from_state(motion_state)
                        expert_states.append(amp_obs)
                        
                except Exception as e:
                    logger.error(f"处理motion {motion_id} 时出错: {e}")
                    continue
            
            if len(expert_states) > 0:
                result = torch.stack(expert_states)
                logger.info(f"✅ 评估模式：成功加载 {len(expert_states)} 个expert观测")
                return result
            else:
                logger.warning("⚠️ 评估模式：没有加载到expert数据，使用fallback")
                return torch.zeros(100, amp_obs_dim, device=self.device)
                
        except Exception as e:
            logger.error(f"❌ 评估模式expert数据加载失败: {e}")
            return torch.zeros(100, amp_obs_dim, device=self.device)

    def _load_expert_amp_data_for_training(self):
        """训练模式：加载大量expert数据"""
        expert_states = []
        amp_obs_dim = self.config.robot.algo_obs_dim_dict["amp_obs"]
        
        try:
            # 训练时使用所有motion，大量采样
            num_motions = self._motion_lib._num_unique_motions
            samples_per_motion = 50  # 减少采样点，避免内存问题
            
            logger.info(f"训练模式：处理 {num_motions} 个motion，每个motion采样 {samples_per_motion} 个点")
            
            for motion_id in range(num_motions):
                try:
                    motion_length = self._motion_lib.get_motion_length([motion_id]).item()
                    
                    if motion_length < 0.5:  # 跳过太短的motion
                        logger.debug(f"Motion {motion_id} 太短 ({motion_length}s)，跳过")
                        continue
                    
                    # 密集采样整个motion
                    for i in range(samples_per_motion):
                        # 均匀分布采样
                        time = (i / max(1, samples_per_motion - 1)) * (motion_length - 0.1)
                        
                        motion_state = self._motion_lib.get_motion_state(
                            torch.tensor([motion_id], device=self.device),
                            torch.tensor([time], device=self.device),
                            offset=torch.zeros(1, 3, device=self.device)
                        )
                        
                        amp_obs = self._build_amp_obs_from_state(motion_state)
                        expert_states.append(amp_obs)
                        
                except Exception as e:
                    logger.error(f"处理motion {motion_id} 时出错: {e}")
                    continue
            
            if len(expert_states) > 0:
                result = torch.stack(expert_states)
                logger.info(f"训练模式：成功加载 {len(expert_states)} 个expert观测")
                return result
            else:
                logger.error("训练模式：没有加载到任何expert数据")
                return torch.zeros(1000, amp_obs_dim, device=self.device)
                
        except Exception as e:
            logger.error(f"训练模式expert数据加载过程出现严重错误: {e}")
            return torch.zeros(1000, amp_obs_dim, device=self.device)

    def _process_dof_obs(self, dof_pos):
        """处理DOF观测 - 参考标准实现"""
        # 简化版本，直接返回相对于默认位置的偏移
        if hasattr(self, 'default_dof_pos') and self.default_dof_pos is not None:
            return dof_pos - self.default_dof_pos.squeeze(0)
        else:
            return dof_pos

    def _build_standard_amp_obs(self, root_pos, root_rot, root_vel, root_ang_vel, 
                               dof_pos, dof_vel, key_body_pos):
        """标准AMP观测构建，参考humanoid实现"""
        # 根节点高度
        root_h = root_pos[:, 2:3]
        
        # 计算heading rotation（去除pitch和roll）
        heading_rot = calc_heading_quat_inv(root_rot, w_last=True)
        
        # 局部坐标系下的根节点旋转（转换为6D表示）
        local_root_rot = quat_mul(heading_rot, root_rot,w_last=True)
        root_rot_obs = quat_to_tan_norm(local_root_rot)
        
        # 局部坐标系下的根节点速度
        local_root_vel = my_quat_rotate(heading_rot, root_vel)
        local_root_ang_vel = my_quat_rotate(heading_rot, root_ang_vel)
        
        # 局部坐标系下的关键身体点位置
        root_pos_expand = root_pos.unsqueeze(-2)
        local_key_body_pos = key_body_pos - root_pos_expand
        
        # 转换关键身体点到局部坐标系
        heading_rot_expand = heading_rot.unsqueeze(-2)
        heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
        flat_local_key_pos = my_quat_rotate(
            heading_rot_expand.view(-1, 4), 
            local_key_body_pos.view(-1, 3)
        ).view(local_key_body_pos.shape[0], -1)
        
        # DOF观测处理
        dof_obs = self._process_dof_obs(dof_pos)
        
        # 组合最终观测 - 严格按照标准格式
        obs = torch.cat([
            root_h,                 # [B, 1] 根节点高度
            root_rot_obs,          # [B, 6] 局部根节点旋转（6D）
            local_root_vel,        # [B, 3] 局部根节点线速度  
            local_root_ang_vel,    # [B, 3] 局部根节点角速度
            dof_obs,              # [B, 30] 处理后的关节位置
            dof_vel,              # [B, 30] 关节速度
            flat_local_key_pos,   # [B, 6] 局部关键身体点位置（2个脚 * 3维）
        ], dim=-1)
        
        return obs

    def _build_amp_obs_from_state(self, motion_state):
        """从motion状态构建标准AMP观测"""
        root_pos = motion_state["root_pos"][0]      # [3]
        root_rot = motion_state["root_rot"][0]      # [4] 
        root_vel = motion_state["root_vel"][0]      # [3]
        root_ang_vel = motion_state["root_ang_vel"][0]  # [3]
        dof_pos = motion_state["dof_pos"][0]        # [30]
        dof_vel = motion_state["dof_vel"][0]        # [30]
        
        # 获取关键身体点
        if "rg_pos_t" in motion_state:
            all_body_pos = motion_state["rg_pos_t"][0]  # [num_bodies, 3]
            # 选择关键身体点（脚部）
            key_body_pos = all_body_pos[self._key_body_ids]  # [num_key_bodies, 3]
        else:
            # Fallback: 使用零向量
            key_body_pos = torch.zeros(len(self._key_body_ids), 3, device=self.device)
        
        # 构建标准AMP观测
        amp_obs = self._build_standard_amp_obs(
            root_pos.unsqueeze(0),      # [1, 3]
            root_rot.unsqueeze(0),      # [1, 4]
            root_vel.unsqueeze(0),      # [1, 3] 
            root_ang_vel.unsqueeze(0),  # [1, 3]
            dof_pos.unsqueeze(0),       # [1, 30]
            dof_vel.unsqueeze(0),       # [1, 30]
            key_body_pos.unsqueeze(0)   # [1, num_key_bodies, 3]
        )
        
        return amp_obs.squeeze(0)

    def _compute_amp_observations(self):
        """计算当前状态的标准AMP观测"""
        # 获取关键身体点位置
        if hasattr(self, '_rigid_body_pos_extend'):
            key_body_pos = self._rigid_body_pos_extend[:, self._key_body_ids, :]
        else:
            key_body_pos = self.simulator._rigid_body_pos[:, self._key_body_ids, :]
        
        # 构建标准AMP观测，确保与expert数据格式完全一致
        amp_obs = self._build_standard_amp_obs(
            self.simulator.robot_root_states[:, :3],      # root_pos [N, 3]
            self.simulator.robot_root_states[:, 3:7],     # root_rot [N, 4]
            self.simulator.robot_root_states[:, 7:10],    # root_vel [N, 3]
            self.simulator.robot_root_states[:, 10:13],   # root_ang_vel [N, 3]
            self.simulator.dof_pos,                       # dof_pos [N, 30]
            self.simulator.dof_vel,                       # dof_vel [N, 30]
            key_body_pos                                  # key_body_pos [N, num_key_bodies, 3]
        )
        
        self.amp_obs_buf[:] = amp_obs
        return amp_obs

    def get_expert_amp_observations(self, num_samples=None):
        """获取专家AMP观测 - 改进的采样策略"""
        if num_samples is None:
            num_samples = self.num_envs
        
        expert_data_size = len(self.expert_amp_loader)
        
        if expert_data_size == 0:
            logger.error("Expert数据为空！")
            amp_obs_dim = self.config.robot.algo_obs_dim_dict["amp_obs"]
            return torch.zeros(num_samples, amp_obs_dim, device=self.device)
        
        if expert_data_size < num_samples:
            logger.debug(f"Expert数据不足: {expert_data_size} < {num_samples}，使用重复采样")
            # 重复采样
            repeats = (num_samples // expert_data_size) + 1
            expanded_data = self.expert_amp_loader.repeat(repeats, 1)
            indices = torch.randperm(len(expanded_data))[:num_samples]
            return expanded_data[indices]
        else:
            # 随机采样
            indices = torch.randperm(expert_data_size)[:num_samples]
            return self.expert_amp_loader[indices]

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
        expected_amp_dim = self.config.robot.algo_obs_dim_dict["amp_obs"]
        logger.info(f"AMP观测维度: {expected_amp_dim}")

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

    def _log_amp_debug_info(self):
        """记录AMP调试信息"""
        if hasattr(self, 'amp_obs_buf') and hasattr(self, 'expert_amp_loader'):
            current_amp_mean = self.amp_obs_buf.mean().item()
            expert_amp_mean = self.expert_amp_loader.mean().item()
            current_amp_std = self.amp_obs_buf.std().item()
            expert_amp_std = self.expert_amp_loader.std().item()
            
            logger.debug(f"AMP Debug - Current: mean={current_amp_mean:.4f}, std={current_amp_std:.4f}")
            logger.debug(f"AMP Debug - Expert: mean={expert_amp_mean:.4f}, std={expert_amp_std:.4f}")
            logger.debug(f"AMP Debug - Expert数据量: {len(self.expert_amp_loader)}")

    def _post_physics_step(self):
        """重写后处理步骤，添加AMP调试信息"""
        super()._post_physics_step()
        
        # 定期记录AMP调试信息
        if self.common_step_counter % 1000 == 0:
            self._log_amp_debug_info()
