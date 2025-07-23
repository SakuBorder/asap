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
        """修复AMP观测维度配置 - 使用安全的默认值"""
        try:
            # 确保基础配置存在
            if not hasattr(config.robot, "algo_obs_dim_dict"):
                config.robot.algo_obs_dim_dict = {}
            
            # 安全地获取DOF观测大小
            dof_obs_size = getattr(config.robot, 'dof_obs_size', 30)  # 默认30
            
            # 默认使用2个关键身体点（脚部）
            num_key_bodies = 2
            
            # 计算标准AMP观测维度
            # root_h(1) + root_rot(6) + root_vel(3) + root_ang_vel(3) + dof_obs + dof_vel + key_body_pos
            expected_amp_dim = 1 + 6 + 3 + 3 + dof_obs_size + dof_obs_size + (3 * num_key_bodies)
            
            # 更新配置中的amp_obs维度
            config.robot.algo_obs_dim_dict["amp_obs"] = expected_amp_dim
            
            logger.info(f"设置AMP观测维度为: {expected_amp_dim}")
            
        except Exception as e:
            logger.warning(f"配置AMP观测维度时出错: {e}，使用默认值")
            # 使用安全的默认值
            if not hasattr(config.robot, "algo_obs_dim_dict"):
                config.robot.algo_obs_dim_dict = {}
            config.robot.algo_obs_dim_dict["amp_obs"] = 79  # 默认值

    def _setup_key_body_ids(self):
        """设置关键身体点索引 - 使用安全的方法（完全参考版本1）"""
        try:
            # 优先使用脚部索引
            if hasattr(self, 'feet_indices') and len(self.feet_indices) > 0:
                self._key_body_ids = self.feet_indices
                logger.info(f"使用脚部索引作为关键身体点: {self._key_body_ids}")
            else:
                # 使用安全的默认值
                self._key_body_ids = torch.tensor([6, 12], device=self.device)
                logger.warning(f"使用默认关键身体点索引: {self._key_body_ids}")
                
        except Exception as e:
            logger.error(f"设置关键身体点时出错: {e}")
            # 使用最安全的默认值
            self._key_body_ids = torch.tensor([6, 12], device=self.device)

    def _init_amp_data(self):
        """初始化AMP相关数据 - 完全参考版本1的方法"""
        try:
            # 获取AMP观测维度
            amp_obs_dim = self.config.robot.algo_obs_dim_dict["amp_obs"]
            
            # 创建AMP观测缓冲区
            self.amp_obs_buf = torch.zeros(
                self.num_envs, 
                amp_obs_dim, 
                device=self.device
            )
            
            logger.info(f"初始化AMP观测缓冲区，形状: {self.amp_obs_buf.shape}")
            
            # 版本1的方法：初始化时使用最小的expert数据集，不进行复杂的预加载
            self.expert_amp_loader = torch.zeros(100, amp_obs_dim, device=self.device)
            
            # 标记为已初始化，但expert数据稍后加载
            self.amp_data_initialized = True
            self.expert_data_loaded = False
            
            logger.info("AMP数据初始化完成，expert数据将在稍后加载")
            
        except Exception as e:
            logger.error(f"AMP数据初始化失败: {e}")
            # 使用最小的fallback
            amp_obs_dim = 79
            self.amp_obs_buf = torch.zeros(self.num_envs, amp_obs_dim, device=self.device)
            self.expert_amp_loader = torch.zeros(100, amp_obs_dim, device=self.device)
            self.amp_data_initialized = False
            self.expert_data_loaded = False

    def _safe_get_motion_state(self, motion_id, time):
        """安全地获取motion状态 - 避免越界"""
        try:
            # 确保motion_id在有效范围内
            num_motions = self._motion_lib._num_unique_motions
            if motion_id >= num_motions:
                logger.warning(f"Motion ID {motion_id} 超出范围，使用0")
                motion_id = 0
            
            motion_state = self._motion_lib.get_motion_state(
                torch.tensor([motion_id], device=self.device),
                torch.tensor([time], device=self.device),
                offset=torch.zeros(1, 3, device=self.device)
            )
            return motion_state
            
        except Exception as e:
            logger.error(f"获取motion状态失败 (motion_id={motion_id}): {e}")
            return None

    def _load_expert_amp_data_simple(self):
        """简单的expert数据加载 - 避免复杂验证（参考版本1）"""
        expert_states = []
        amp_obs_dim = self.config.robot.algo_obs_dim_dict["amp_obs"]
        
        try:
            # 只使用motion 0，通过时间采样获得多样性
            motion_id = 0
            
            # 先测试能否获取motion长度
            try:
                motion_length = self._motion_lib.get_motion_length([motion_id]).item()
                logger.info(f"Motion {motion_id} 长度: {motion_length}s")
            except Exception as e:
                logger.error(f"无法获取motion长度: {e}")
                # 使用默认值
                motion_length = 1.0
            
            # 采样时间点
            num_samples = 100
            for i in range(num_samples):
                try:
                    # 均匀分布的时间点
                    time_ratio = i / max(1, num_samples - 1)
                    time = time_ratio * max(0.1, motion_length - 0.1)
                    
                    # 安全地获取motion状态
                    motion_state = self._safe_get_motion_state(motion_id, time)
                    if motion_state is None:
                        continue
                    
                    # 构建AMP观测
                    amp_obs = self._build_amp_obs_from_state_simple(motion_state)
                    if amp_obs is not None:
                        expert_states.append(amp_obs)
                        
                except Exception as e:
                    logger.debug(f"处理时间点 {i} 失败: {e}")
                    continue
            
            if len(expert_states) > 0:
                result = torch.stack(expert_states)
                logger.info(f"成功加载 {len(expert_states)} 个expert观测")
                return result
            else:
                logger.error("没有加载到任何expert数据，返回零数据")
                return torch.zeros(100, amp_obs_dim, device=self.device)
                
        except Exception as e:
            logger.error(f"Expert数据加载失败: {e}")
            return torch.zeros(100, amp_obs_dim, device=self.device)

    def _build_amp_obs_from_state_simple(self, motion_state):
        """从motion状态构建AMP观测 - 简化版本（参考版本1）"""
        try:
            # 检查必要的状态是否存在
            required_keys = ["root_pos", "root_rot", "root_vel", "root_ang_vel", "dof_pos", "dof_vel"]
            for key in required_keys:
                if key not in motion_state:
                    logger.error(f"Motion状态缺少 {key}")
                    return None
            
            root_pos = motion_state["root_pos"][0]      # [3]
            root_rot = motion_state["root_rot"][0]      # [4] 
            root_vel = motion_state["root_vel"][0]      # [3]
            root_ang_vel = motion_state["root_ang_vel"][0]  # [3]
            dof_pos = motion_state["dof_pos"][0]        # [30]
            dof_vel = motion_state["dof_vel"][0]        # [30]
            
            # 简化处理关键身体点：如果获取失败就用零向量
            if "rg_pos_t" in motion_state:
                try:
                    all_body_pos = motion_state["rg_pos_t"][0]  # [num_bodies, 3]
                    num_bodies = all_body_pos.shape[0]
                    
                    # 简单处理：确保索引不越界
                    safe_indices = []
                    for idx in self._key_body_ids:
                        if isinstance(idx, torch.Tensor):
                            idx = idx.item()
                        safe_idx = min(idx, num_bodies - 1)
                        safe_indices.append(safe_idx)
                    
                    if len(safe_indices) > 0:
                        indices_tensor = torch.tensor(safe_indices, device=self.device)
                        key_body_pos = all_body_pos[indices_tensor]
                    else:
                        key_body_pos = torch.zeros(len(self._key_body_ids), 3, device=self.device)
                        
                except Exception as e:
                    logger.debug(f"处理关键身体点失败: {e}")
                    key_body_pos = torch.zeros(len(self._key_body_ids), 3, device=self.device)
            else:
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
            
        except Exception as e:
            logger.error(f"构建AMP观测时出错: {e}")
            return None

    def _build_standard_amp_obs(self, root_pos, root_rot, root_vel, root_ang_vel, 
                               dof_pos, dof_vel, key_body_pos):
        """标准AMP观测构建，参考humanoid实现"""
        try:
            # 根节点高度
            root_h = root_pos[:, 2:3]
            
            # 计算heading rotation（去除pitch和roll）
            heading_rot = calc_heading_quat_inv(root_rot, w_last=True)
            
            # 局部坐标系下的根节点旋转（转换为6D表示）
            local_root_rot = quat_mul(heading_rot, root_rot, w_last=True)
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
            
        except Exception as e:
            logger.error(f"构建标准AMP观测时出错: {e}")
            # 返回零观测作为fallback
            amp_obs_dim = self.config.robot.algo_obs_dim_dict["amp_obs"]
            return torch.zeros(root_pos.shape[0], amp_obs_dim, device=self.device)

    def _process_dof_obs(self, dof_pos):
        """处理DOF观测 - 参考标准实现"""
        try:
            # 简化版本，直接返回相对于默认位置的偏移
            if hasattr(self, 'default_dof_pos') and self.default_dof_pos is not None:
                return dof_pos - self.default_dof_pos.squeeze(0)
            else:
                return dof_pos
        except:
            return dof_pos

    def _compute_amp_observations(self):
        """计算当前状态的标准AMP观测"""
        try:
            # 获取关键身体点位置
            if hasattr(self, '_rigid_body_pos_extend'):
                all_body_pos = self._rigid_body_pos_extend
            else:
                all_body_pos = self.simulator._rigid_body_pos
            
            # 安全地选择关键身体点
            num_bodies = all_body_pos.shape[1]
            valid_key_body_ids = []
            
            for idx in self._key_body_ids:
                if isinstance(idx, torch.Tensor):
                    idx = idx.item()
                safe_idx = min(idx, num_bodies - 1)
                valid_key_body_ids.append(safe_idx)
            
            if len(valid_key_body_ids) > 0:
                valid_indices = torch.tensor(valid_key_body_ids, device=self.device)
                key_body_pos = all_body_pos[:, valid_indices, :]
                
                # 如果索引数量不足，复制最后一个
                if len(valid_key_body_ids) < len(self._key_body_ids):
                    missing_count = len(self._key_body_ids) - len(valid_key_body_ids)
                    last_pos = key_body_pos[:, -1:, :].repeat(1, missing_count, 1)
                    key_body_pos = torch.cat([key_body_pos, last_pos], dim=1)
            else:
                key_body_pos = torch.zeros(self.num_envs, len(self._key_body_ids), 3, device=self.device)
            
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
            
        except Exception as e:
            logger.error(f"计算AMP观测时出错: {e}")
            return self.amp_obs_buf

    def get_expert_amp_observations(self, num_samples=None):
        """获取专家AMP观测 - 完全参考版本1的简单方法"""
        if num_samples is None:
            num_samples = self.num_envs
        
        try:
            # 如果expert数据还没有加载，尝试加载
            if not self.expert_data_loaded:
                self.expert_amp_loader = self._load_expert_amp_data_simple()
                self.expert_data_loaded = True
            
            expert_data_size = len(self.expert_amp_loader)
            
            if expert_data_size == 0:
                logger.error("Expert数据为空！")
                amp_obs_dim = self.config.robot.algo_obs_dim_dict["amp_obs"]
                fallback_data = torch.zeros(num_samples, amp_obs_dim, device=self.device)
                return fallback_data
            
            # 检查数据质量
            if torch.isnan(self.expert_amp_loader).any() or torch.isinf(self.expert_amp_loader).any():
                logger.error("Expert数据包含NaN或Inf！")
                amp_obs_dim = self.config.robot.algo_obs_dim_dict["amp_obs"]
                fallback_data = torch.zeros(num_samples, amp_obs_dim, device=self.device)
                return fallback_data
            
            if expert_data_size < num_samples:
                logger.debug(f"Expert数据不足: {expert_data_size} < {num_samples}，使用重复采样")
                # 重复采样
                repeats = (num_samples // expert_data_size) + 1
                expanded_data = self.expert_amp_loader.repeat(repeats, 1)
                indices = torch.randperm(len(expanded_data))[:num_samples]
                sampled_data = expanded_data[indices]
            else:
                # 随机采样
                indices = torch.randperm(expert_data_size)[:num_samples]
                sampled_data = self.expert_amp_loader[indices]
            
            # 验证采样数据
            if torch.isnan(sampled_data).any() or torch.isinf(sampled_data).any():
                logger.error("采样的expert数据包含NaN或Inf！")
                amp_obs_dim = self.config.robot.algo_obs_dim_dict["amp_obs"]
                return torch.zeros(num_samples, amp_obs_dim, device=self.device)
            
            logger.debug(f"成功获取expert观测: shape={sampled_data.shape}, "
                        f"mean={sampled_data.mean().item():.4f}, "
                        f"std={sampled_data.std().item():.4f}")
            
            return sampled_data
            
        except Exception as e:
            logger.error(f"获取expert观测时出错: {e}")
            amp_obs_dim = self.config.robot.algo_obs_dim_dict["amp_obs"]
            return torch.zeros(num_samples, amp_obs_dim, device=self.device)

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
        self.terminate_when_motion_far_threshold = self.config.termination_curriculum.terminate_when_motion_far_initial_threshold

        # 验证AMP观测维度
        expected_amp_dim = self.config.robot.algo_obs_dim_dict["amp_obs"]
        logger.info(f"AMP观测维度: {expected_amp_dim}")

    def set_is_evaluating(self):
        """设置为评估模式 - 完全参考版本1"""
        logger.info("🔄 AMPMotionTracking 切换到评估模式")
        
        # 调用父类方法
        super().set_is_evaluating()
        
        # 重新配置AMP数据为评估模式
        if self.amp_data_initialized:
            logger.info("重新配置AMP数据为评估模式")
            try:
                # 重新加载数据（使用相同的简单方法）
                self.expert_amp_loader = self._load_expert_amp_data_simple()
                self.expert_data_loaded = True
                logger.info(f"✅ AMP数据已切换到评估模式，expert数据形状: {self.expert_amp_loader.shape}")
            except Exception as e:
                logger.error(f"❌ AMP评估模式重新初始化失败: {e}")
                # 使用安全的fallback
                amp_obs_dim = self.config.robot.algo_obs_dim_dict["amp_obs"]
                self.expert_amp_loader = torch.zeros(100, amp_obs_dim, device=self.device)
                self.expert_data_loaded = True

    def _post_physics_step(self):
        """重写后处理步骤，参考版本1"""
        super()._post_physics_step()
        
        # 延迟加载expert数据（如果还没有正确加载）
        if not self.expert_data_loaded and self.common_step_counter % 1000 == 0:
            try:
                self.expert_amp_loader = self._load_expert_amp_data_simple()
                self.expert_data_loaded = True
                logger.info(f"延迟加载expert数据完成，形状: {self.expert_amp_loader.shape}")
            except Exception as e:
                logger.debug(f"延迟加载expert数据失败: {e}")

    def _reset_tasks_callback(self, env_ids):
        if len(env_ids) == 0:
            return
        super()._reset_tasks_callback(env_ids)
        self._resample_motion_times(env_ids) # need to resample before reset root states
        if self.config.termination.terminate_when_motion_far and self.config.termination_curriculum.terminate_when_motion_far_curriculum:
            self._update_terminate_when_motion_far_curriculum()
    
    def _update_terminate_when_motion_far_curriculum(self):
        assert self.config.termination.terminate_when_motion_far and self.config.termination_curriculum.terminate_when_motion_far_curriculum
        if self.average_episode_length < self.config.termination_curriculum.terminate_when_motion_far_curriculum_level_down_threshold:
            self.terminate_when_motion_far_threshold *= (1 + self.config.termination_curriculum.terminate_when_motion_far_curriculum_degree)
        elif self.average_episode_length > self.config.termination_curriculum.terminate_when_motion_far_curriculum_level_up_threshold:
            self.terminate_when_motion_far_threshold *= (1 - self.config.termination_curriculum.terminate_when_motion_far_curriculum_degree)
        self.terminate_when_motion_far_threshold = np.clip(self.terminate_when_motion_far_threshold, 
                                                         self.config.termination_curriculum.terminate_when_motion_far_threshold_min, 
                                                         self.config.termination_curriculum.terminate_when_motion_far_threshold_max)
