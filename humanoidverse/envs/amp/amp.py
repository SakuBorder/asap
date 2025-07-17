import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import os
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from isaac_utils.rotations import (
    my_quat_rotate,
    calc_heading_quat_inv,
    calc_heading_quat,
    quat_mul,
    quat_conjugate,
    quat_to_angle_axis,
    quat_rotate_inverse,
    xyzw_to_wxyz,
    wxyz_to_xyzw
)
from humanoidverse.envs.env_utils.visualization import Point
from humanoidverse.utils.motion_lib.skeleton import SkeletonTree
from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot
from termcolor import colored
from loguru import logger
from scipy.spatial.transform import Rotation as sRot
import joblib
import random
from collections import deque

class AMPDiscriminator(nn.Module):
    """AMP判别器网络：区分真实动作和策略生成的动作"""
    
    def __init__(self, input_dim, hidden_dims=[1024, 512, 256], activation='relu'):
        super().__init__()
        self.input_dim = input_dim
        
        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        # 输出层 - 输出单个概率值
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class AMPReplayBuffer:
    """AMP专用经验回放缓冲区"""
    
    def __init__(self, capacity, obs_dim, device):
        self.capacity = capacity
        self.device = device
        self.obs_dim = obs_dim
        
        # 使用deque实现环形缓冲区
        self.buffer = deque(maxlen=capacity)
        
    def add(self, observations):
        """添加观察数据到缓冲区"""
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        
        for obs in observations:
            self.buffer.append(obs.clone())
    
    def sample(self, batch_size):
        """从缓冲区随机采样"""
        if len(self.buffer) < batch_size:
            # 如果缓冲区数据不足，则使用全部数据
            batch_size = len(self.buffer)
        
        indices = random.sample(range(len(self.buffer)), batch_size)
        batch = [self.buffer[i] for i in indices]
        
        return torch.stack(batch).to(self.device)
    
    def size(self):
        return len(self.buffer)

class LeggedRobotAMP(LeggedRobotBase):
    """基于原有代码的完整AMP实现"""
    
    def __init__(self, config, device):
        self.init_done = False
        self.debug_viz = True
        
        # AMP相关配置
        self.amp_config = config.get('amp', {})
        self.discriminator_lr = self.amp_config.get('discriminator_lr', 3e-4)
        self.discriminator_weight_decay = self.amp_config.get('discriminator_weight_decay', 1e-4)
        self.discriminator_batch_size = self.amp_config.get('discriminator_batch_size', 512)
        self.discriminator_replay_buffer_size = self.amp_config.get('discriminator_replay_buffer_size', 100000)
        self.discriminator_update_freq = self.amp_config.get('discriminator_update_freq', 2)
        
        # 奖励权重
        self.task_reward_weight = self.amp_config.get('task_reward_weight', 0.5)
        self.style_reward_weight = self.amp_config.get('style_reward_weight', 0.5)
        self.discriminator_reward_scale = self.amp_config.get('discriminator_reward_scale', 2.0)
        
        # 判别器损失类型
        self.discriminator_loss_type = self.amp_config.get('discriminator_loss_type', 'bce')  # 'bce' or 'least_square'
        
        super().__init__(config, device)
        
        # 初始化AMP相关组件
        self._init_motion_lib()
        self._init_motion_extend()
        self._init_tracking_config()
        self._init_amp_components()
        
        self.init_done = True
        self.debug_viz = True
        
        self._init_save_motion()
        
        # 训练统计
        self.discriminator_update_count = 0
        self.amp_loss_history = deque(maxlen=100)
        self.style_reward_history = deque(maxlen=100)
    
    def _init_amp_components(self):
        """初始化AMP组件"""
        # 计算判别器输入维度
        self.amp_obs_dim = self._calculate_amp_obs_dim()
        
        # 创建判别器网络
        self.discriminator = AMPDiscriminator(
            input_dim=self.amp_obs_dim,
            hidden_dims=self.amp_config.get('discriminator_hidden_dims', [1024, 512, 256]),
            activation=self.amp_config.get('discriminator_activation', 'relu')
        ).to(self.device)
        
        # 判别器优化器
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.discriminator_lr,
            weight_decay=self.discriminator_weight_decay
        )
        
        # 经验回放缓冲区
        self.amp_replay_buffer = AMPReplayBuffer(
            capacity=self.discriminator_replay_buffer_size,
            obs_dim=self.amp_obs_dim,
            device=self.device
        )
        
        # 参考动作数据缓冲区
        self.reference_motion_buffer = AMPReplayBuffer(
            capacity=self.discriminator_replay_buffer_size,
            obs_dim=self.amp_obs_dim,
            device=self.device
        )
        
        logger.info(f"AMP组件初始化完成:")
        logger.info(f"  - 判别器输入维度: {self.amp_obs_dim}")
        logger.info(f"  - 判别器网络: {self.discriminator}")
        logger.info(f"  - 缓冲区容量: {self.discriminator_replay_buffer_size}")
    
    def _calculate_amp_obs_dim(self):
        """计算AMP观察维度"""
        # 基础维度：身体位置 + 旋转 + 速度 + 角速度 + 关节角度 + 关节速度
        base_dim = 0
        
        # 身体位置差异 (num_bodies * 3)
        base_dim += (self.num_bodies + getattr(self, 'num_extend_bodies', 0)) * 3
        
        # 身体旋转差异 (num_bodies * 4, 四元数)
        base_dim += (self.num_bodies + getattr(self, 'num_extend_bodies', 0)) * 4
        
        # 身体速度差异 (num_bodies * 3)
        base_dim += (self.num_bodies + getattr(self, 'num_extend_bodies', 0)) * 3
        
        # 身体角速度差异 (num_bodies * 3)
        base_dim += (self.num_bodies + getattr(self, 'num_extend_bodies', 0)) * 3
        
        # 关节角度差异 (num_dof)
        base_dim += self.num_dof
        
        # 关节速度差异 (num_dof)
        base_dim += self.num_dof
        
        return base_dim
    
    def _init_motion_lib(self):
        """初始化动作库"""
        self.config.robot.motion.step_dt = self.dt
        self._motion_lib = MotionLibRobot(self.config.robot.motion, num_envs=self.num_envs, device=self.device)
        
        if self.is_evaluating:
            self._motion_lib.load_motions(random_sample=False)
        else:
            self._motion_lib.load_motions(random_sample=True)
            
        res = self._resample_motion_times(torch.arange(self.num_envs))
        self.motion_dt = self._motion_lib._motion_dt
        self.motion_start_idx = 0
        self.num_motions = self._motion_lib._num_unique_motions
        
        logger.info(f"动作库初始化完成，共{self.num_motions}个动作")
    
    def _init_tracking_config(self):
        """初始化跟踪配置"""
        if "motion_tracking_link" in self.config.robot.motion:
            self.motion_tracking_id = [self.simulator._body_list.index(link) for link in self.config.robot.motion.motion_tracking_link]
        if "lower_body_link" in self.config.robot.motion:
            self.lower_body_id = [self.simulator._body_list.index(link) for link in self.config.robot.motion.lower_body_link]
        if "upper_body_link" in self.config.robot.motion:
            self.upper_body_id = [self.simulator._body_list.index(link) for link in self.config.robot.motion.upper_body_link]
        if self.config.resample_motion_when_training:
            self.resample_time_interval = np.ceil(self.config.resample_time_interval_s / self.dt)
    
    def _init_motion_extend(self):
        """初始化动作扩展"""
        if "extend_config" in self.config.robot.motion:
            if len(self.config.robot.motion.extend_config) > 0:
                extend_parent_ids, extend_pos, extend_rot = [], [], []
                for extend_config in self.config.robot.motion.extend_config:
                    extend_parent_ids.append(self.simulator._body_list.index(extend_config["parent_name"]))
                    extend_pos.append(extend_config["pos"])
                    extend_rot.append(extend_config["rot"])
                    self.simulator._body_list.append(extend_config["joint_name"])

                self.extend_body_parent_ids = torch.tensor(extend_parent_ids, device=self.device, dtype=torch.long)
                self.extend_body_pos_in_parent = torch.tensor(extend_pos).repeat(self.num_envs, 1, 1).to(self.device)
                self.extend_body_rot_in_parent_wxyz = torch.tensor(extend_rot).repeat(self.num_envs, 1, 1).to(self.device)
                self.extend_body_rot_in_parent_xyzw = self.extend_body_rot_in_parent_wxyz[:, :, [1, 2, 3, 0]]
                self.num_extend_bodies = len(extend_parent_ids)
            else:
                self.num_extend_bodies = 0

            self.marker_coords = torch.zeros(self.num_envs, 
                                        self.num_bodies + self.num_extend_bodies, 
                                        3, 
                                        dtype=torch.float, 
                                        device=self.device, 
                                        requires_grad=False)
            self.ref_body_pos_extend = torch.zeros(self.num_envs, self.num_bodies + self.num_extend_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
            self.dif_global_body_pos = torch.zeros(self.num_envs, self.num_bodies + self.num_extend_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
    
    def _init_save_motion(self):
        """初始化动作保存"""
        if "save_motion" in self.config:
            self.save_motion = self.config.save_motion
            if self.save_motion:
                os.makedirs(Path(self.config.ckpt_dir) / "motions", exist_ok=True)
                if hasattr(self.config, 'dump_motion_name'):
                    self.save_motion_dir = Path(self.config.ckpt_dir) / "motions" / (str(self.config.eval_timestamp) + "_" + self.config.dump_motion_name)
                else:
                    self.save_motion_dir = Path(self.config.ckpt_dir) / "motions" / f"{self.config.save_note}_{self.config.eval_timestamp}"
                self.save_motion = True
                self.num_augment_joint = len(self.config.robot.motion.extend_config)
                self.motions_for_saving = {'root_trans_offset':[], 'pose_aa':[], 'dof':[], 'root_rot':[], 'actor_obs':[], 'action':[], 'terminate':[],
                                            'root_lin_vel':[], 'root_ang_vel':[], 'dof_vel':[]}
                self.motion_times_buf = []
                self.start_save = False
        else:
            self.save_motion = False
    
    def _init_buffers(self):
        """初始化缓冲区"""
        super()._init_buffers()
        self.vr_3point_marker_coords = torch.zeros(self.num_envs, 3, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.realtime_vr_keypoints_pos = torch.zeros(3, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.realtime_vr_keypoints_vel = torch.zeros(3, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.motion_ids = torch.arange(self.num_envs).to(self.device)
        self.motion_start_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.motion_len = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        
        # AMP相关缓冲区
        self.amp_obs_buf = torch.zeros(self.num_envs, self.amp_obs_dim, dtype=torch.float, device=self.device, requires_grad=False)
        self.style_rewards = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        
    def _init_domain_rand_buffers(self):
        """初始化域随机化缓冲区"""
        super()._init_domain_rand_buffers()
        self.ref_episodic_offset = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
    
    def _extract_amp_observations(self):
        """提取AMP观察特征"""
        # 连接所有相关状态信息
        amp_obs_components = []
        
        # 身体位置差异
        amp_obs_components.append(self.dif_global_body_pos.view(self.num_envs, -1))
        
        # 身体旋转差异
        amp_obs_components.append(self.dif_global_body_rot.view(self.num_envs, -1))
        
        # 身体速度差异
        amp_obs_components.append(self.dif_global_body_vel.view(self.num_envs, -1))
        
        # 身体角速度差异
        amp_obs_components.append(self.dif_global_body_ang_vel.view(self.num_envs, -1))
        
        # 关节角度差异
        amp_obs_components.append(self.dif_joint_angles.view(self.num_envs, -1))
        
        # 关节速度差异
        amp_obs_components.append(self.dif_joint_velocities.view(self.num_envs, -1))
        
        # 连接所有组件
        amp_obs = torch.cat(amp_obs_components, dim=-1)
        
        return amp_obs
    
    def _sample_reference_motions(self, batch_size):
        """采样参考动作数据"""
        # 随机选择环境和时间
        random_env_ids = torch.randint(0, self.num_envs, (batch_size,), device=self.device)
        random_motion_times = torch.rand(batch_size, device=self.device) * self.motion_len[random_env_ids] + self.motion_start_times[random_env_ids]
        
        # 获取参考动作状态
        offset = self.env_origins[random_env_ids] if hasattr(self, 'env_origins') else None
        motion_res = self._motion_lib.get_motion_state(self.motion_ids[random_env_ids], random_motion_times, offset=offset)
        
        # 计算参考状态的AMP观察
        # 这里需要临时计算参考状态的差异（实际上应该是0，因为它们是参考状态）
        # 但为了保持一致性，我们可以直接使用参考状态本身
        ref_amp_obs = torch.zeros(batch_size, self.amp_obs_dim, device=self.device)
        
        # 这里应该根据实际需求计算参考状态的特征
        # 简化版本：直接使用零向量或者从现有的参考状态计算
        
        return ref_amp_obs
    
    def _compute_discriminator_loss(self, policy_obs, reference_obs):
        """计算判别器损失"""
        # 策略观察的标签为0（假），参考观察的标签为1（真）
        policy_labels = torch.zeros(policy_obs.shape[0], 1, device=self.device)
        reference_labels = torch.ones(reference_obs.shape[0], 1, device=self.device)
        
        # 前向传播
        policy_logits = self.discriminator(policy_obs)
        reference_logits = self.discriminator(reference_obs)
        
        if self.discriminator_loss_type == 'bce':
            # 二元交叉熵损失
            policy_loss = F.binary_cross_entropy_with_logits(policy_logits, policy_labels)
            reference_loss = F.binary_cross_entropy_with_logits(reference_logits, reference_labels)
            discriminator_loss = policy_loss + reference_loss
        else:
            # 最小二乘损失
            policy_loss = F.mse_loss(policy_logits, policy_labels)
            reference_loss = F.mse_loss(reference_logits, reference_labels)
            discriminator_loss = 0.5 * (policy_loss + reference_loss)
        
        return discriminator_loss, policy_logits, reference_logits
    
    def _compute_style_reward(self, amp_obs):
        """计算风格奖励"""
        with torch.no_grad():
            logits = self.discriminator(amp_obs)
            
            if self.discriminator_loss_type == 'bce':
                # 使用sigmoid和对数变换
                probs = torch.sigmoid(logits)
                style_reward = -torch.log(torch.clamp(1 - probs, min=0.0001))
            else:
                # 最小二乘版本
                style_reward = torch.clamp(1 - 0.25 * (logits - 1)**2, min=0.0001)
            
            style_reward = style_reward.squeeze(-1) * self.discriminator_reward_scale
            
        return style_reward
    
    def _update_discriminator(self):
        """更新判别器"""
        if self.amp_replay_buffer.size() < self.discriminator_batch_size:
            return
        
        # 采样策略数据
        policy_obs = self.amp_replay_buffer.sample(self.discriminator_batch_size // 2)
        
        # 采样参考数据
        reference_obs = self._sample_reference_motions(self.discriminator_batch_size // 2)
        
        # 计算损失
        discriminator_loss, policy_logits, reference_logits = self._compute_discriminator_loss(policy_obs, reference_obs)
        
        # 反向传播
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        
        self.discriminator_optimizer.step()
        
        # 记录统计信息
        self.discriminator_update_count += 1
        self.amp_loss_history.append(discriminator_loss.item())
        
        # 计算准确率
        with torch.no_grad():
            policy_acc = (torch.sigmoid(policy_logits) < 0.5).float().mean()
            reference_acc = (torch.sigmoid(reference_logits) > 0.5).float().mean()
            total_acc = (policy_acc + reference_acc) / 2
        
        if self.discriminator_update_count % 100 == 0:
            logger.info(f"判别器更新 {self.discriminator_update_count}: 损失={discriminator_loss.item():.4f}, 准确率={total_acc.item():.4f}")
    
    def _reset_tasks_callback(self, env_ids):
        """重置任务回调"""
        if len(env_ids) == 0:
            return
        super()._reset_tasks_callback(env_ids)
        self._resample_motion_times(env_ids)
        
        if self.config.termination.terminate_when_motion_far and self.config.termination_curriculum.terminate_when_motion_far_curriculum:
            self._update_terminate_when_motion_far_curriculum()
    
    def _update_terminate_when_motion_far_curriculum(self):
        """更新动作距离终止课程"""
        assert self.config.termination.terminate_when_motion_far and self.config.termination_curriculum.terminate_when_motion_far_curriculum
        if self.average_episode_length < self.config.termination_curriculum.terminate_when_motion_far_curriculum_level_down_threshold:
            self.terminate_when_motion_far_threshold *= (1 + self.config.termination_curriculum.terminate_when_motion_far_curriculum_degree)
        elif self.average_episode_length > self.config.termination_curriculum.terminate_when_motion_far_curriculum_level_up_threshold:
            self.terminate_when_motion_far_threshold *= (1 - self.config.termination_curriculum.terminate_when_motion_far_curriculum_degree)
        self.terminate_when_motion_far_threshold = np.clip(self.terminate_when_motion_far_threshold, 
                                                         self.config.termination_curriculum.terminate_when_motion_far_threshold_min, 
                                                         self.config.termination_curriculum.terminate_when_motion_far_threshold_max)
    
    def _update_tasks_callback(self):
        """更新任务回调"""
        super()._update_tasks_callback()
        if self.config.resample_motion_when_training:
            if self.common_step_counter % self.resample_time_interval == 0:
                logger.info(f"在步骤{self.common_step_counter}重新采样动作")
                self.resample_motion()
    
    def _resample_motion_times(self, env_ids):
        """重新采样动作时间"""
        if len(env_ids) == 0:
            return
        self.motion_len[env_ids] = self._motion_lib.get_motion_length(self.motion_ids[env_ids])
        if self.is_evaluating and not self.config.enforce_randomize_motion_start_eval:
            self.motion_start_times[env_ids] = torch.zeros(len(env_ids), dtype=torch.float32, device=self.device)
        else:
            self.motion_start_times[env_ids] = self._motion_lib.sample_time(self.motion_ids[env_ids])
    
    def resample_motion(self):
        """重新采样动作"""
        self._motion_lib.load_motions(random_sample=True)
        self.reset_envs_idx(torch.arange(self.num_envs, device=self.device))
    
    def next_task(self):
        """下一个任务（评估时使用）"""
        self.motion_start_idx += self.num_envs
        if self.motion_start_idx >= self.num_motions:
            self.motion_start_idx = 0
        self._motion_lib.load_motions(random_sample=False, start_idx=self.motion_start_idx)
        self.reset_all()
    
    def _pre_compute_observations_callback(self):
        """预计算观察回调"""
        super()._pre_compute_observations_callback()
        
        offset = self.env_origins if hasattr(self, 'env_origins') else None
        B = self.motion_ids.shape[0]
        motion_times = (self.episode_length_buf + 1) * self.dt + self.motion_start_times
        
        # 获取动作状态
        motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)
        
        ref_body_pos_extend = motion_res["rg_pos_t"]
        self.ref_body_pos_extend[:] = ref_body_pos_extend
        ref_body_vel_extend = motion_res["body_vel_t"]
        self.ref_body_rot_extend = ref_body_rot_extend = motion_res["rg_rot_t"]
        ref_body_ang_vel_extend = motion_res["body_ang_vel_t"]
        ref_joint_pos = motion_res["dof_pos"]
        ref_joint_vel = motion_res["dof_vel"]
        
        # 计算扩展刚体状态
        if self.num_extend_bodies > 0:
            rotated_pos_in_parent = my_quat_rotate(
                self.simulator._rigid_body_rot[:, self.extend_body_parent_ids].reshape(-1, 4),
                self.extend_body_pos_in_parent.reshape(-1, 3)
            )
            extend_curr_pos = my_quat_rotate(
                self.extend_body_rot_in_parent_xyzw.reshape(-1, 4),
                rotated_pos_in_parent
            ).view(self.num_envs, -1, 3) + self.simulator._rigid_body_pos[:, self.extend_body_parent_ids]
            self._rigid_body_pos_extend = torch.cat([self.simulator._rigid_body_pos, extend_curr_pos], dim=1)
            
            extend_curr_rot = quat_mul(self.simulator._rigid_body_rot[:, self.extend_body_parent_ids].reshape(-1, 4),
                                        self.extend_body_rot_in_parent_xyzw.reshape(-1, 4),
                                        w_last=True).view(self.num_envs, -1, 4)
            self._rigid_body_rot_extend = torch.cat([self.simulator._rigid_body_rot, extend_curr_rot], dim=1)
            
            self._rigid_body_ang_vel_extend = torch.cat([self.simulator._rigid_body_ang_vel, self.simulator._rigid_body_ang_vel[:, self.extend_body_parent_ids]], dim=1)
            
            self._rigid_body_ang_vel_global = self.simulator._rigid_body_ang_vel[:, self.extend_body_parent_ids]
            angular_velocity_contribution = torch.cross(self._rigid_body_ang_vel_global, self.extend_body_pos_in_parent.view(self.num_envs, -1, 3), dim=2)
            extend_curr_vel = self.simulator._rigid_body_vel[:, self.extend_body_parent_ids] + angular_velocity_contribution.view(self.num_envs, -1, 3)
            self._rigid_body_vel_extend = torch.cat([self.simulator._rigid_body_vel, extend_curr_vel], dim=1)
        else:
            self._rigid_body_pos_extend = self.simulator._rigid_body_pos
            self._rigid_body_rot_extend = self.simulator._rigid_body_rot
            self._rigid_body_vel_extend = self.simulator._rigid_body_vel
            self._rigid_body_ang_vel_extend = self.simulator._rigid_body_ang_vel
        
        # 计算差异
        self.dif_global_body_pos = ref_body_pos_extend - self._rigid_body_pos_extend
        self.dif_global_body_rot = quat_mul(ref_body_rot_extend, quat_conjugate(self._rigid_body_rot_extend, w_last=True), w_last=True)
        self.dif_global_body_vel = ref_body_vel_extend - self._rigid_body_vel_extend
        self.dif_global_body_ang_vel = ref_body_ang_vel_extend - self._rigid_body_ang_vel_extend
        self.dif_joint_angles = ref_joint_pos - self.simulator.dof_pos
        self.dif_joint_velocities = ref_joint_vel - self.simulator.dof_vel
        
        # 提取AMP观察
        self.amp_obs_buf = self._extract_amp_observations()
        
        # 计算风格奖励
        self.style_rewards = self._compute_style_reward(self.amp_obs_buf)
        
        # 将当前观察添加到回放缓冲区
        self.amp_replay_buffer.add(self.amp_obs_buf.detach())
        
        # 更新判别器
        if self.common_step_counter % self.discriminator_update_freq == 0:
            self._update_discriminator()
        
        # 可视化标记
        self.marker_coords[:] = ref_body_pos_extend.reshape(B, -1, 3)
        
        # 计算局部观察
        env_batch_size = self.simulator._rigid_body_pos.shape[0]
        num_rigid_bodies = self.simulator._rigid_body_pos.shape[1]
        
        heading_inv_rot = calc_heading_quat_inv(self.simulator.robot_root_states[:, 3:7].clone(), w_last=True)
        heading_inv_rot_expand = heading_inv_rot.unsqueeze(1).expand(-1, num_rigid_bodies+self.num_extend_bodies, -1).reshape(-1, 4)
        
        heading_rot = calc_heading_quat(self.simulator.robot_root_states[:, 3:7].clone(), w_last=True)
        heading_rot_expand = heading_rot.unsqueeze(1).expand(-1, num_rigid_bodies, -1).reshape(-1, 4)
        
        dif_global_body_pos_for_obs_compute = ref_body_pos_extend.view(env_batch_size, -1, 3) - self._rigid_body_pos_extend.view(env_batch_size, -1, 3)
        dif_local_body_pos_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), dif_global_body_pos_for_obs_compute.view(-1, 3))
        
        self._obs_dif_local_rigid_body_pos = dif_local_body_pos_flat.view(env_batch_size, -1)
        
        global_ref_rigid_body_pos = ref_body_pos_extend.view(env_batch_size, -1, 3) - self.simulator.robot_root_states[:, :3].view(env_batch_size, 1, 3)
        local_ref_rigid_body_pos_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), global_ref_rigid_body_pos.view(-1, 3))
        self._obs_local_ref_rigid_body_pos = local_ref_rigid_body_pos_flat.view(env_batch_size, -1)
        
        global_ref_body_vel = ref_body_vel_extend.view(env_batch_size, -1, 3)
        local_ref_rigid_body_vel_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), global_ref_body_vel.view(-1, 3))
        self._obs_local_ref_rigid_body_vel = local_ref_rigid_body_vel_flat.view(env_batch_size, -1)
        
        # 计算动作阶段
        self._ref_motion_length = self._motion_lib.get_motion_length(self.motion_ids)
        self._ref_motion_phase = motion_times / self._ref_motion_length
        self._ref_motion_phase = self._ref_motion_phase.unsqueeze(1)
        
        self._log_motion_tracking_info()
    
    def _compute_reward(self):
        """计算奖励"""
        # 计算任务奖励
        task_reward = self._compute_task_reward()
        
        # 组合任务奖励和风格奖励
        combined_reward = (self.task_reward_weight * task_reward + 
                          self.style_reward_weight * self.style_rewards)
        
        # 更新奖励缓冲区
        self.rew_buf[:] = combined_reward
        
        # 记录统计信息
        self.style_reward_history.extend(self.style_rewards.cpu().numpy())
        
        # 额外信息
        self.extras["ref_body_pos_extend"] = self.ref_body_pos_extend.clone()
        self.extras["ref_body_rot_extend"] = self.ref_body_rot_extend.clone()
        self.extras["style_rewards"] = self.style_rewards.clone()
        self.extras["task_rewards"] = task_reward.clone()
        self.extras["combined_rewards"] = combined_reward.clone()
        
        # 日志记录
        if self.common_step_counter % 1000 == 0:
            avg_style_reward = np.mean(list(self.style_reward_history)) if self.style_reward_history else 0
            avg_task_reward = task_reward.mean().item()
            logger.info(f"步骤 {self.common_step_counter}: 风格奖励={avg_style_reward:.4f}, 任务奖励={avg_task_reward:.4f}")
    
    def _compute_task_reward(self):
        """计算任务特定奖励"""
        # 基础任务奖励组合
        reward_components = []
        
        # 身体位置跟踪奖励
        if hasattr(self, 'upper_body_id') and hasattr(self, 'lower_body_id'):
            body_pos_reward = self._reward_teleop_body_position_extend()
            reward_components.append(body_pos_reward)
        
        # VR 3点跟踪奖励
        if hasattr(self, 'motion_tracking_id'):
            vr_3point_reward = self._reward_teleop_vr_3point()
            reward_components.append(vr_3point_reward)
        
        # 关节位置跟踪奖励
        joint_pos_reward = self._reward_teleop_joint_position()
        reward_components.append(joint_pos_reward)
        
        # 关节速度跟踪奖励
        joint_vel_reward = self._reward_teleop_joint_velocity()
        reward_components.append(joint_vel_reward)
        
        # 身体旋转跟踪奖励
        body_rot_reward = self._reward_teleop_body_rotation_extend()
        reward_components.append(body_rot_reward)
        
        # 身体速度跟踪奖励
        body_vel_reward = self._reward_teleop_body_velocity_extend()
        reward_components.append(body_vel_reward)
        
        # 身体角速度跟踪奖励
        body_ang_vel_reward = self._reward_teleop_body_ang_velocity_extend()
        reward_components.append(body_ang_vel_reward)
        
        # 组合所有奖励
        if reward_components:
            task_reward = sum(reward_components) / len(reward_components)
        else:
            task_reward = torch.zeros(self.num_envs, device=self.device)
        
        return task_reward
    
    def _log_motion_tracking_info(self):
        """记录动作跟踪信息"""
        if hasattr(self, 'upper_body_id') and hasattr(self, 'lower_body_id'):
            upper_body_diff = self.dif_global_body_pos[:, self.upper_body_id, :]
            lower_body_diff = self.dif_global_body_pos[:, self.lower_body_id, :]
            upper_body_diff_norm = upper_body_diff.norm(dim=-1).mean()
            lower_body_diff_norm = lower_body_diff.norm(dim=-1).mean()
            self.log_dict["upper_body_diff_norm"] = upper_body_diff_norm
            self.log_dict["lower_body_diff_norm"] = lower_body_diff_norm
        
        if hasattr(self, 'motion_tracking_id'):
            vr_3point_diff = self.dif_global_body_pos[:, self.motion_tracking_id, :]
            vr_3point_diff_norm = vr_3point_diff.norm(dim=-1).mean()
            self.log_dict["vr_3point_diff_norm"] = vr_3point_diff_norm
        
        joint_pos_diff = self.dif_joint_angles
        joint_pos_diff_norm = joint_pos_diff.norm(dim=-1).mean()
        self.log_dict["joint_pos_diff_norm"] = joint_pos_diff_norm
        
        # AMP相关日志
        if self.amp_loss_history:
            self.log_dict["amp_discriminator_loss"] = np.mean(list(self.amp_loss_history))
        if self.style_reward_history:
            self.log_dict["amp_style_reward"] = np.mean(list(self.style_reward_history))
        
        self.log_dict["amp_buffer_size"] = self.amp_replay_buffer.size()
    
    def _check_termination(self):
        """检查终止条件"""
        super()._check_termination()
        if self.config.termination.terminate_when_motion_far:
            reset_buf_motion_far = torch.any(torch.norm(self.dif_global_body_pos, dim=-1) > self.terminate_when_motion_far_threshold, dim=-1)
            self.reset_buf |= reset_buf_motion_far
            if self.config.termination_curriculum.terminate_when_motion_far_curriculum:
                self.log_dict["terminate_when_motion_far_threshold"] = torch.tensor(self.terminate_when_motion_far_threshold, dtype=torch.float)
    
    def _update_timeout_buf(self):
        """更新超时缓冲区"""
        super()._update_timeout_buf()
        if self.config.termination.terminate_when_motion_end:
            current_time = (self.episode_length_buf) * self.dt + self.motion_start_times
            self.time_out_buf |= current_time > self.motion_len
    
    def _reset_root_states(self, env_ids):
        """重置根状态"""
        motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times
        offset = self.env_origins if hasattr(self, 'env_origins') else None
        motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)
        
        if hasattr(self, 'custom_origins') and self.custom_origins:
            self.simulator.robot_root_states[env_ids, :3] = motion_res['root_pos'][env_ids]
            if self.config.simulator.config.name == 'isaacgym':
                self.simulator.robot_root_states[env_ids, 3:7] = motion_res['root_rot'][env_ids]
            elif self.config.simulator.config.name == 'isaacsim':
                self.simulator.robot_root_states[env_ids, 3:7] = xyzw_to_wxyz(motion_res['root_rot'][env_ids])
            elif self.config.simulator.config.name == 'genesis':
                self.simulator.robot_root_states[env_ids, 3:7] = motion_res['root_rot'][env_ids]
            self.simulator.robot_root_states[env_ids, 7:10] = motion_res['root_vel'][env_ids]
            self.simulator.robot_root_states[env_ids, 10:13] = motion_res['root_ang_vel'][env_ids]
        else:
            root_pos_noise = self.config.init_noise_scale.root_pos * self.config.noise_to_initial_level
            root_rot_noise = self.config.init_noise_scale.root_rot * 3.14 / 180 * self.config.noise_to_initial_level
            root_vel_noise = self.config.init_noise_scale.root_vel * self.config.noise_to_initial_level
            root_ang_vel_noise = self.config.init_noise_scale.root_ang_vel * self.config.noise_to_initial_level
            
            root_pos = motion_res['root_pos'][env_ids]
            root_rot = motion_res['root_rot'][env_ids]
            root_vel = motion_res['root_vel'][env_ids]
            root_ang_vel = motion_res['root_ang_vel'][env_ids]
            
            self.simulator.robot_root_states[env_ids, :3] = root_pos + torch.randn_like(root_pos) * root_pos_noise
            if self.config.simulator.config.name == 'isaacgym':
                self.simulator.robot_root_states[env_ids, 3:7] = quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), root_rot, w_last=True)
            elif self.config.simulator.config.name == 'isaacsim':
                self.simulator.robot_root_states[env_ids, 3:7] = xyzw_to_wxyz(quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), root_rot, w_last=True))
            elif self.config.simulator.config.name == 'genesis':
                self.simulator.robot_root_states[env_ids, 3:7] = quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), root_rot, w_last=True)
            elif self.config.simulator.config.name == 'mujoco':
                self.simulator.robot_root_states[env_ids, 3:7] = quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), root_rot, w_last=True)
            self.simulator.robot_root_states[env_ids, 7:10] = root_vel + torch.randn_like(root_vel) * root_vel_noise
            self.simulator.robot_root_states[env_ids, 10:13] = root_ang_vel + torch.randn_like(root_ang_vel) * root_ang_vel_noise
    
    def small_random_quaternions(self, n, max_angle):
        """生成小的随机四元数"""
        axis = torch.randn((n, 3), device=self.device)
        axis = axis / torch.norm(axis, dim=1, keepdim=True)
        angles = max_angle * torch.rand((n, 1), device=self.device)
        
        sin_half_angle = torch.sin(angles / 2)
        cos_half_angle = torch.cos(angles / 2)
        
        q = torch.cat([sin_half_angle * axis, cos_half_angle], dim=1)
        return q
    
    def _reset_dofs(self, env_ids):
        """重置自由度"""
        motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times
        offset = self.env_origins if hasattr(self, 'env_origins') else None
        motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)
        
        dof_pos_noise = self.config.init_noise_scale.dof_pos * self.config.noise_to_initial_level
        dof_vel_noise = self.config.init_noise_scale.dof_vel * self.config.noise_to_initial_level
        dof_pos = motion_res['dof_pos'][env_ids]
        dof_vel = motion_res['dof_vel'][env_ids]
        self.simulator.dof_pos[env_ids] = dof_pos + torch.randn_like(dof_pos) * dof_pos_noise
        self.simulator.dof_vel[env_ids] = dof_vel + torch.randn_like(dof_vel) * dof_vel_noise
    
    def _draw_debug_vis(self):
        """绘制调试可视化"""
        self.simulator.clear_lines()
        self._refresh_sim_tensors()
        
        for env_id in range(self.num_envs):
            if not self.config.use_teleop_control:
                for pos_id, pos_joint in enumerate(self.marker_coords[env_id]):
                    if self.config.robot.motion.visualization.customize_color:
                        color_inner = self.config.robot.motion.visualization.marker_joint_colors[pos_id % len(self.config.robot.motion.visualization.marker_joint_colors)]
                    else:
                        color_inner = (0.3, 0.3, 0.3)
                    
                    color_inner = tuple(color_inner)
                    self.simulator.draw_sphere(pos_joint, 0.04, color_inner, env_id, pos_id)
            else:
                for pos_id, pos_joint in enumerate(self.teleop_marker_coords[env_id]):
                    self.simulator.draw_sphere(pos_joint, 0.04, (0.851, 0.144, 0.07), env_id, pos_id)
    
    def _post_physics_step(self):
        """物理步骤后处理"""
        super()._post_physics_step()
        
        if self.save_motion:
            motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times
            
            if (len(self.motions_for_saving['dof'])) > self.config.save_total_steps:
                for k, v in self.motions_for_saving.items():
                    self.motions_for_saving[k] = torch.stack(v[3:]).transpose(0,1).numpy()
                
                self.motions_for_saving['motion_times'] = torch.stack(self.motion_times_buf[3:]).transpose(0,1).numpy()
                
                dump_data = {}
                num_motions = self.num_envs
                keys_to_save = self.motions_for_saving.keys()
                
                for i in range(num_motions):
                    motion_key = f"motion{i}"
                    dump_data[motion_key] = {
                        key: self.motions_for_saving[key][i] for key in keys_to_save
                    }
                    dump_data[motion_key]['fps'] = 1 / self.dt
                
                joblib.dump(dump_data, f'{self.save_motion_dir}.pkl')
                print(colored(f"保存动作数据到 {self.save_motion_dir}.pkl", 'green'))
                import sys
                sys.exit()
            
            root_trans = self.simulator.robot_root_states[:, 0:3].cpu()
            if self.config.simulator.config.name == "isaacgym":
                root_rot = self.simulator.robot_root_states[:, 3:7].cpu()
            elif self.config.simulator.config.name == "isaacsim":
                root_rot = self.simulator.robot_root_states[:, [4, 5, 6, 3]].cpu()
            elif self.config.simulator.config.name == "genesis":
                root_rot = self.simulator.robot_root_states[:, 3:7].cpu()
            
            root_rot_vec = torch.from_numpy(sRot.from_quat(root_rot.numpy()).as_rotvec()).float()
            dof = self.simulator.dof_pos.cpu()
            
            pose_aa = torch.cat([root_rot_vec[:, None, :], self._motion_lib.mesh_parsers.dof_axis * dof[:, :, None], torch.zeros((self.num_envs, self.num_augment_joint, 3))], axis=1)
            self.motions_for_saving['root_trans_offset'].append(root_trans)
            self.motions_for_saving['root_rot'].append(root_rot)
            self.motions_for_saving['dof'].append(dof)
            self.motions_for_saving['pose_aa'].append(pose_aa)
            self.motions_for_saving['action'].append(self.actions.cpu())
            self.motions_for_saving['actor_obs'].append(self.obs_buf_dict['actor_obs'].cpu())
            self.motions_for_saving['terminate'].append(self.reset_buf.cpu())
            self.motions_for_saving['dof_vel'].append(self.simulator.dof_vel.cpu())
            self.motions_for_saving['root_lin_vel'].append(self.simulator.robot_root_states[:, 7:10].cpu())
            self.motions_for_saving['root_ang_vel'].append(self.simulator.robot_root_states[:, 10:13].cpu())
            
            self.motion_times_buf.append(motion_times.cpu())
            self.start_save = True
    
    # 观察函数
    def _get_obs_dif_local_rigid_body_pos(self):
        return self._obs_dif_local_rigid_body_pos
    
    def _get_obs_local_ref_rigid_body_pos(self):
        return self._obs_local_ref_rigid_body_pos
    
    def _get_obs_ref_motion_phase(self):
        return self._ref_motion_phase
    
    def _get_obs_amp_observations(self):
        """获取AMP观察"""
        return self.amp_obs_buf
    
    def _get_obs_style_rewards(self):
        """获取风格奖励"""
        return self.style_rewards.unsqueeze(-1)
    
    # 历史观察函数
    def _get_obs_history_actor(self):
        assert "history_actor" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['history_actor']
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)
    
    def _get_obs_history_critic(self):
        assert "history_critic" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['history_critic']
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)
    
    # 奖励函数
    def _reward_teleop_body_position_extend(self):
        upper_body_diff = self.dif_global_body_pos[:, self.upper_body_id, :]
        lower_body_diff = self.dif_global_body_pos[:, self.lower_body_id, :]
        
        diff_body_pos_dist_upper = (upper_body_diff**2).mean(dim=-1).mean(dim=-1)
        diff_body_pos_dist_lower = (lower_body_diff**2).mean(dim=-1).mean(dim=-1)
        
        r_body_pos_upper = torch.exp(-diff_body_pos_dist_upper / self.config.rewards.reward_tracking_sigma.teleop_upper_body_pos)
        r_body_pos_lower = torch.exp(-diff_body_pos_dist_lower / self.config.rewards.reward_tracking_sigma.teleop_lower_body_pos)
        r_body_pos = r_body_pos_lower * self.config.rewards.teleop_body_pos_lowerbody_weight + r_body_pos_upper * self.config.rewards.teleop_body_pos_upperbody_weight
        
        return r_body_pos
    
    def _reward_teleop_vr_3point(self):
        vr_3point_diff = self.dif_global_body_pos[:, self.motion_tracking_id, :]
        vr_3point_dist = (vr_3point_diff**2).mean(dim=-1).mean(dim=-1)
        r_vr_3point = torch.exp(-vr_3point_dist / self.config.rewards.reward_tracking_sigma.teleop_vr_3point_pos)
        return r_vr_3point
    
    def _reward_teleop_body_position_feet(self):
        feet_diff = self.dif_global_body_pos[:, self.feet_indices, :]
        feet_dist = (feet_diff**2).mean(dim=-1).mean(dim=-1)
        r_feet = torch.exp(-feet_dist / self.config.rewards.reward_tracking_sigma.teleop_feet_pos)
        return r_feet
    
    def _reward_teleop_body_rotation_extend(self):
        rotation_diff = quat_to_angle_axis(self.dif_global_body_rot)[0]
        diff_body_rot_dist = (rotation_diff**2).mean(dim=-1)
        r_body_rot = torch.exp(-diff_body_rot_dist / self.config.rewards.reward_tracking_sigma.teleop_body_rot)
        return r_body_rot
    
    def _reward_teleop_body_velocity_extend(self):
        velocity_diff = self.dif_global_body_vel
        diff_body_vel_dist = (velocity_diff**2).mean(dim=-1).mean(dim=-1)
        r_body_vel = torch.exp(-diff_body_vel_dist / self.config.rewards.reward_tracking_sigma.teleop_body_vel)
        return r_body_vel
    
    def _reward_teleop_body_ang_velocity_extend(self):
        ang_velocity_diff = self.dif_global_body_ang_vel
        diff_body_ang_vel_dist = (ang_velocity_diff**2).mean(dim=-1).mean(dim=-1)
        r_body_ang_vel = torch.exp(-diff_body_ang_vel_dist / self.config.rewards.reward_tracking_sigma.teleop_body_ang_vel)
        return r_body_ang_vel
    
    def _reward_teleop_joint_position(self):
        joint_pos_diff = self.dif_joint_angles
        diff_joint_pos_dist = (joint_pos_diff**2).mean(dim=-1)
        r_joint_pos = torch.exp(-diff_joint_pos_dist / self.config.rewards.reward_tracking_sigma.teleop_joint_pos)
        return r_joint_pos
    
    def _reward_teleop_joint_velocity(self):
        joint_vel_diff = self.dif_joint_velocities
        diff_joint_vel_dist = (joint_vel_diff**2).mean(dim=-1)
        r_joint_vel = torch.exp(-diff_joint_vel_dist / self.config.rewards.reward_tracking_sigma.teleop_joint_vel)
        return r_joint_vel
    
    def setup_visualize_entities(self):
        """设置可视化实体"""
        if self.debug_viz and self.config.simulator.config.name == "genesis":
            num_visualize_markers = len(self.config.robot.motion.visualization.marker_joint_colors)
            self.simulator.add_visualize_entities(num_visualize_markers)
        elif self.debug_viz and self.config.simulator.config.name == "mujoco":
            num_visualize_markers = len(self.config.robot.motion.visualization.marker_joint_colors)
            self.simulator.add_visualize_entities(num_visualize_markers)
        else:
            pass
    
    def get_amp_metrics(self):
        """获取AMP指标"""
        metrics = {}
        
        if self.amp_loss_history:
            metrics['amp_discriminator_loss'] = np.mean(list(self.amp_loss_history))
            metrics['amp_discriminator_loss_std'] = np.std(list(self.amp_loss_history))
        
        if self.style_reward_history:
            metrics['amp_style_reward'] = np.mean(list(self.style_reward_history))
            metrics['amp_style_reward_std'] = np.std(list(self.style_reward_history))
        
        metrics['amp_buffer_size'] = self.amp_replay_buffer.size()
        metrics['amp_reference_buffer_size'] = self.reference_motion_buffer.size()
        metrics['amp_discriminator_updates'] = self.discriminator_update_count
        
        return metrics
    
    def save_amp_model(self, path):
        """保存AMP模型"""
        save_dict = {
            'discriminator_state_dict': self.discriminator.state_dict(),
            'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
            'amp_config': self.amp_config,
            'amp_obs_dim': self.amp_obs_dim,
            'discriminator_update_count': self.discriminator_update_count,
        }
        torch.save(save_dict, path)
        logger.info(f"AMP模型已保存到: {path}")
    
    def load_amp_model(self, path):
        """加载AMP模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        self.discriminator_update_count = checkpoint.get('discriminator_update_count', 0)
        logger.info(f"AMP模型已从 {path} 加载")


# 配置示例
def create_amp_config():
    """创建AMP配置示例"""
    amp_config = {
        # 判别器网络配置
        'discriminator_lr': 3e-4,
        'discriminator_weight_decay': 1e-4,
        'discriminator_batch_size': 512,
        'discriminator_replay_buffer_size': 100000,
        'discriminator_update_freq': 2,
        'discriminator_hidden_dims': [1024, 512, 256],
        'discriminator_activation': 'relu',
        'discriminator_loss_type': 'bce',  # 'bce' or 'least_square'
        
        # 奖励权重
        'task_reward_weight': 0.5,
        'style_reward_weight': 0.5,
        'discriminator_reward_scale': 2.0,
        
        # 训练配置
        'amp_replay_buffer_size': 100000,
        'reference_motion_buffer_size': 100000,
    }
    return amp_config

# 训练脚本示例
def train_amp_agent():
    """AMP训练脚本示例"""
    import yaml
    from omegaconf import OmegaConf
    
    # 加载配置
    config_path = "path/to/your/amp_config.yaml"
    config = OmegaConf.load(config_path)
    
    # 添加AMP配置
    config.amp = create_amp_config()
    
    # 创建环境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = LeggedRobotAMP(config, device)
    
    # 训练循环
    max_episodes = 10000
    save_interval = 1000
    
    for episode in range(max_episodes):
        # 重置环境
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # 获取动作（这里需要你的策略网络）
            # action = policy_network(obs)
            action = torch.randn(env.num_envs, env.num_actions, device=device)  # 随机动作示例
            
            # 执行动作
            obs, reward, done, info = env.step(action)
            episode_reward += reward.mean().item()
        
        # 记录训练信息
        if episode % 100 == 0:
            amp_metrics = env.get_amp_metrics()
            logger.info(f"回合 {episode}: 平均奖励 = {episode_reward:.4f}")
            logger.info(f"AMP指标: {amp_metrics}")
        
        # 保存模型
        if episode % save_interval == 0:
            env.save_amp_model(f"amp_model_episode_{episode}.pth")
    
    logger.info("训练完成！")

# 评估脚本示例
def evaluate_amp_agent():
    """AMP评估脚本示例"""
    import yaml
    from omegaconf import OmegaConf
    
    # 加载配置
    config_path = "path/to/your/amp_config.yaml"
    config = OmegaConf.load(config_path)
    config.amp = create_amp_config()
    
    # 创建环境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = LeggedRobotAMP(config, device)
    env.set_is_evaluating()
    
    # 加载训练好的模型
    env.load_amp_model("amp_model_final.pth")
    
    # 评估循环
    num_eval_episodes = 100
    total_rewards = []
    
    for episode in range(num_eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # 使用训练好的策略（这里需要你的策略网络）
            # action = policy_network(obs)
            action = torch.randn(env.num_envs, env.num_actions, device=device)
            
            obs, reward, done, info = env.step(action)
            episode_reward += reward.mean().item()
        
        total_rewards.append(episode_reward)
        
        # 切换到下一个动作
        env.next_task()
    
    # 打印评估结果
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    logger.info(f"评估完成! 平均奖励: {avg_reward:.4f} ± {std_reward:.4f}")

# 完整的AMP配置文件示例
def create_full_amp_config():
    """创建完整的AMP配置文件"""
    config = {
        'env_name': 'LeggedRobotAMP',
        'num_envs': 4096,
        'device': 'cuda',
        
        # 机器人配置
        'robot': {
            'motion': {
                'step_dt': 0.0167,  # 60fps
                'motion_file': 'path/to/motion/library.pkl',
                'motion_tracking_link': ['left_hand', 'right_hand', 'head'],
                'lower_body_link': ['pelvis', 'left_foot', 'right_foot'],
                'upper_body_link': ['torso', 'left_hand', 'right_hand', 'head'],
                'extend_config': [],
                'visualization': {
                    'customize_color': True,
                    'marker_joint_colors': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                }
            }
        },
        
        # AMP配置
        'amp': create_amp_config(),
        
        # 奖励配置
        'rewards': {
            'reward_tracking_sigma': {
                'teleop_upper_body_pos': 0.1,
                'teleop_lower_body_pos': 0.1,
                'teleop_vr_3point_pos': 0.1,
                'teleop_body_rot': 0.1,
                'teleop_body_vel': 0.1,
                'teleop_body_ang_vel': 0.1,
                'teleop_joint_pos': 0.1,
                'teleop_joint_vel': 0.1,
                'teleop_feet_pos': 0.1
            },
            'teleop_body_pos_lowerbody_weight': 0.6,
            'teleop_body_pos_upperbody_weight': 0.4
        },
        
        # 终止条件
        'termination': {
            'terminate_when_motion_far': True,
            'terminate_when_motion_end': True
        },
        
        # 课程学习
        'termination_curriculum': {
            'terminate_when_motion_far_curriculum': True,
            'terminate_when_motion_far_curriculum_level_down_threshold': 100,
            'terminate_when_motion_far_curriculum_level_up_threshold': 200,
            'terminate_when_motion_far_curriculum_degree': 0.1,
            'terminate_when_motion_far_threshold_min': 0.5,
            'terminate_when_motion_far_threshold_max': 2.0
        },
        
        # 训练配置
        'resample_motion_when_training': True,
        'resample_time_interval_s': 30.0,
        'enforce_randomize_motion_start_eval': False,
        
        # 噪声配置
        'init_noise_scale': {
            'root_pos': 0.1,
            'root_rot': 0.1,
            'root_vel': 0.1,
            'root_ang_vel': 0.1,
            'dof_pos': 0.1,
            'dof_vel': 0.1
        },
        'noise_to_initial_level': 1.0,
        
        # 观察配置
        'obs': {
            'obs_auxiliary': {
                'history_actor': {
                    'dof_pos': 3,
                    'dof_vel': 3,
                    'actions': 3
                },
                'history_critic': {
                    'dof_pos': 3,
                    'dof_vel': 3,
                    'actions': 3
                }
            }
        },
        
        # 保存配置
        'save_motion': False,
        'save_total_steps': 10000,
        'ckpt_dir': 'checkpoints',
        'save_note': 'amp_training'
    }
    
    return config

# 使用示例
if __name__ == "__main__":
    # 创建配置
    config = create_full_amp_config()
    
    # 保存配置到文件
    import yaml
    with open('amp_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("AMP配置已保存到 amp_config.yaml")
    print("现在可以使用以下命令训练AMP模型:")
    print("python train_amp.py --config amp_config.yaml")
    
    # 如果你想直接运行训练，取消下面的注释
    train_amp_agent()
    
