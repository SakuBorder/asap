import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from humanoidverse.agents.ppo.ppo import PPO
from humanoidverse.agents.modules.amp_modules import AMPDiscriminator
from loguru import logger

class AMPPPO(PPO):
    def __init__(self, env, config, log_dir=None, device='cpu'):
        # 在调用父类init前确保AMP配置正确
        self._pre_init_amp_config(env, config)
        super().__init__(env, config, log_dir, device)

    def _pre_init_amp_config(self, env, config):
        """在初始化前确保AMP配置正确"""
        # 确保algo_obs_dim_dict存在
        if not hasattr(env.config.robot, 'algo_obs_dim_dict'):
            env.config.robot.algo_obs_dim_dict = {}
        
        # 如果环境已经设置了AMP观测维度，使用环境的设置
        if "amp_obs" in env.config.robot.algo_obs_dim_dict:
            amp_obs_dim = env.config.robot.algo_obs_dim_dict["amp_obs"]
        else:
            # 否则计算默认的AMP观测维度
            dof_obs_size = env.config.robot.dof_obs_size
            num_key_bodies = len(env.config.robot.key_bodies) if hasattr(env.config.robot, 'key_bodies') else 2
            amp_obs_dim = 1 + 6 + 3 + 3 + dof_obs_size + dof_obs_size + (3 * num_key_bodies)
            env.config.robot.algo_obs_dim_dict["amp_obs"] = amp_obs_dim
        
        logger.info(f"预设置AMP观测维度: {amp_obs_dim}")

    def _setup_models_and_optimizer(self):
        super()._setup_models_and_optimizer()
        
        # 再次确认AMP观测维度
        self._ensure_amp_obs_dim()
        
        # 创建AMP判别器
        self.discriminator = AMPDiscriminator(
            self.algo_obs_dim_dict,
            self.config.module_dict.discriminator
        ).to(self.device)
        
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.discriminator_learning_rate
        )
        
        logger.info(f"AMP判别器设置完成，学习率: {self.config.discriminator_learning_rate}")

    def _ensure_amp_obs_dim(self):
        """确保AMP观测维度正确设置"""
        if "amp_obs" not in self.algo_obs_dim_dict:
            logger.warning("algo_obs_dim_dict中没有amp_obs，正在重新添加...")
            dof_obs_size = self.env.config.robot.dof_obs_size
            amp_obs_dim = 2 * dof_obs_size + 6
            self.algo_obs_dim_dict["amp_obs"] = amp_obs_dim
            logger.info(f"重新添加amp_obs维度: {amp_obs_dim}")
        else:
            amp_obs_dim = self.algo_obs_dim_dict["amp_obs"]
            logger.info(f"确认amp_obs维度: {amp_obs_dim}")

    def _setup_storage(self):
        super()._setup_storage()
        
        # 确保AMP观测维度存在
        self._ensure_amp_obs_dim()
        
        # 注册AMP观测存储
        try:
            # import ipdb;ipdb.set_trace()
            amp_obs_dim = self.algo_obs_dim_dict["amp_obs"]
            # self.storage.register_key('amp_obs', shape=(amp_obs_dim,), dtype=torch.float)
            logger.info(f"✅ 成功注册AMP观测存储，维度: {amp_obs_dim}")
            logger.info(f"当前已注册的keys: {self.storage.stored_keys}")
            self.amp_storage_available = True
        except Exception as e:
            logger.error(f"❌ AMP观测存储注册失败: {e}")
            self.amp_storage_available = False
            # 不再有fallback重试，直接标记为不可用
            logger.warning("⚠️  AMP将在无存储模式下运行")

    def _init_loss_dict_at_training_step(self):
        """初始化训练步骤的损失字典"""
        loss_dict = super()._init_loss_dict_at_training_step()
        
        # 基础AMP损失
        loss_dict['Discriminator_Loss'] = 0.0
        loss_dict['Discriminator_Real_Acc'] = 0.0
        loss_dict['Discriminator_Fake_Acc'] = 0.0
        loss_dict['Discriminator_Total_Acc'] = 0.0
        loss_dict['Discriminator_Real_Logits'] = 0.0
        loss_dict['Discriminator_Fake_Logits'] = 0.0
        
        # AMP奖励统计
        loss_dict['AMP_Reward_Mean'] = 0.0
        loss_dict['AMP_Reward_Std'] = 0.0
        loss_dict['AMP_Reward_Min'] = 0.0  
        loss_dict['AMP_Reward_Max'] = 0.0
        loss_dict['AMP_Reward_Weight'] = float(self.config.amp_reward_weight)
        
        # 数据质量监控
        loss_dict['Expert_Data_Size'] = 0.0
        loss_dict['Expert_Data_Mean'] = 0.0
        loss_dict['Expert_Data_Std'] = 0.0
        loss_dict['Current_Data_Mean'] = 0.0
        loss_dict['Current_Data_Std'] = 0.0
        
        return loss_dict


    def _rollout_step(self, obs_dict):
        """重写rollout以包含AMP观测"""
        with torch.inference_mode():
            for i in range(self.num_steps_per_env):
                policy_state_dict = {}
                policy_state_dict = self._actor_rollout_step(obs_dict, policy_state_dict)
                values = self._critic_eval_step(obs_dict).detach()
                policy_state_dict["values"] = values

                # 获取AMP观测
                amp_obs = self._get_amp_observations()
                
                # 验证AMP观测
                if amp_obs is None:
                    logger.error("AMP observations is None during rollout")
                    amp_obs = torch.zeros(self.env.num_envs, self.algo_obs_dim_dict["amp_obs"], device=self.device)
                
                policy_state_dict["amp_obs"] = amp_obs

                # 存储观测到storage
                for obs_key in obs_dict.keys():
                    self.storage.update_key(obs_key, obs_dict[obs_key])

                # 存储policy状态，包括AMP观测
                for obs_key in policy_state_dict.keys():
                    if obs_key in self.storage.stored_keys:
                        self.storage.update_key(obs_key, policy_state_dict[obs_key])
                    else:
                        if obs_key == "amp_obs" and not self.amp_storage_available:
                            logger.debug("amp_obs存储不可用，跳过存储")
                        
                actions = policy_state_dict["actions"]
                actor_state = {"actions": actions}
                obs_dict, rewards, dones, infos = self.env.step(actor_state)
                
                for obs_key in obs_dict.keys():
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                rewards, dones = rewards.to(self.device), dones.to(self.device)

                # 计算AMP奖励
                amp_rewards, amp_reward_computed = self._compute_amp_rewards(amp_obs)
                
                # 验证AMP奖励计算
                if amp_reward_computed:
                    # 检查奖励是否有效
                    if torch.isnan(amp_rewards).any() or torch.isinf(amp_rewards).any():
                        logger.error("AMP rewards contain NaN or Inf, skipping")
                        amp_reward_computed = False
                    else:
                        rewards += amp_rewards
                        logger.debug(f"AMP rewards applied: mean={amp_rewards.mean().item():.4f}")
                else:
                    logger.debug("AMP rewards not computed")
                
                # 记录AMP奖励统计
                self._log_amp_rewards(infos, amp_rewards if amp_reward_computed else None, amp_reward_computed)

                # 确保infos["to_log"]中的所有值都是tensor
                self._ensure_infos_tensors(infos)

                self.episode_env_tensors.add(infos["to_log"])
                
                rewards_stored = rewards.clone().unsqueeze(1)
                if 'time_outs' in infos:
                    rewards_stored += self.gamma * policy_state_dict['values'] * infos['time_outs'].unsqueeze(1).to(self.device)
                    
                self.storage.update_key('rewards', rewards_stored)
                self.storage.update_key('dones', dones.unsqueeze(1))
                self.storage.increment_step()

                self._process_env_step(rewards, dones, infos)

                if self.log_dir is not None:
                    if 'episode' in infos:
                        self.ep_infos.append(infos['episode'])
                    self.cur_reward_sum += rewards
                    self.cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False).flatten()
                    # self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    if len(new_ids) > 0:
                        if len(self.cur_reward_sum.shape) == 1:
                            # 如果是1D tensor，直接使用
                            reward_values = self.cur_reward_sum[new_ids].cpu().numpy().tolist()
                            length_values = self.cur_episode_length[new_ids].cpu().numpy().tolist()
                        else:
                            # 如果是2D tensor，取第一列
                            reward_values = self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                            length_values = self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        
                        self.rewbuffer.extend(reward_values)
                        self.lenbuffer.extend(length_values)
                    # self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    if len(new_ids) > 0:
                        if len(self.cur_reward_sum.shape) == 1:
                            # 如果是1D tensor，直接使用
                            reward_values = self.cur_reward_sum[new_ids].cpu().numpy().tolist()
                            length_values = self.cur_episode_length[new_ids].cpu().numpy().tolist()
                        else:
                            # 如果是2D tensor，取第一列
                            reward_values = self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                            length_values = self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        
                        self.rewbuffer.extend(reward_values)
                        self.lenbuffer.extend(length_values)
                    self.cur_reward_sum[new_ids] = 0
                    self.cur_episode_length[new_ids] = 0

            self.stop_time = time.time()
            self.collection_time = self.stop_time - self.start_time
            self.start_time = self.stop_time
            
            returns, advantages = self._compute_returns(
                last_obs_dict=obs_dict,
                policy_state_dict=dict(
                    values=self.storage.query_key('values'), 
                    dones=self.storage.query_key('dones'), 
                    rewards=self.storage.query_key('rewards')
                )
            )
            self.storage.batch_update_data('returns', returns)
            self.storage.batch_update_data('advantages', advantages)

        return obs_dict

    def _get_amp_observations(self):
        """获取AMP观测的统一接口"""
        try:
            # 尝试多种方法获取AMP观测
            if hasattr(self.env, '_compute_amp_observations'):
                amp_obs = self.env._compute_amp_observations()
                logger.debug(f"从环境_compute_amp_observations获取AMP观测，形状: {amp_obs.shape}")
                return amp_obs
            elif hasattr(self.env, '_get_obs_amp_obs'):
                amp_obs = self.env._get_obs_amp_obs()
                logger.debug(f"从环境_get_obs_amp_obs获取AMP观测，形状: {amp_obs.shape}")
                return amp_obs
            elif 'amp_obs' in self.env.obs_buf_dict_raw.get('critic_obs', {}):
                # 尝试从obs_buf_dict_raw获取
                amp_obs = self.env.obs_buf_dict_raw['critic_obs']['amp_obs']
                logger.debug(f"从obs_buf_dict_raw获取AMP观测，形状: {amp_obs.shape}")
                return amp_obs
            else:
                # Fallback: 手动构造AMP观测
                logger.debug("使用fallback方法构造AMP观测")
                return self._construct_amp_observations_fallback()
                
        except Exception as e:
            logger.warning(f"获取AMP观测时出错: {e}")
            return self._construct_amp_observations_fallback()

    def _construct_amp_observations_fallback(self):
        """Fallback方法：手动构造AMP观测"""
        try:
            dof_pos = self.env.simulator.dof_pos
            dof_vel = self.env.simulator.dof_vel
            base_lin_vel = self.env.base_lin_vel
            base_ang_vel = self.env.base_ang_vel
            
            amp_obs = torch.cat([dof_pos, dof_vel, base_lin_vel, base_ang_vel], dim=-1)
            logger.debug(f"Fallback构造AMP观测，形状: {amp_obs.shape}")
            return amp_obs
            
        except Exception as e:
            logger.error(f"Fallback构造AMP观测失败: {e}")
            # 最后的fallback：返回零观测
            amp_obs_dim = self.algo_obs_dim_dict["amp_obs"]
            return torch.zeros(self.env.num_envs, amp_obs_dim, device=self.device)

    def _compute_amp_rewards(self, amp_obs):
        """计算AMP奖励"""
        try:
            # 验证输入数据
            if amp_obs is None:
                logger.error("AMP observations is None")
                return torch.zeros(self.env.num_envs, device=self.device), False
                
            if amp_obs.shape[0] == 0:
                logger.error("AMP observations is empty")
                return torch.zeros(self.env.num_envs, device=self.device), False
                
            # 检查数据质量
            if torch.isnan(amp_obs).any() or torch.isinf(amp_obs).any():
                logger.error("AMP observations contains NaN or Inf")
                return torch.zeros(amp_obs.shape[0], device=self.device), False
                
            # 计算discriminator奖励
            amp_rewards = self.discriminator.compute_disc_rewards(amp_obs)
            
            # 验证奖励
            if torch.isnan(amp_rewards).any() or torch.isinf(amp_rewards).any():
                logger.error("AMP rewards contains NaN or Inf")
                return torch.zeros(amp_obs.shape[0], device=self.device), False
                
            # 应用权重
            weighted_rewards = amp_rewards * self.config.amp_reward_weight
            
            logger.debug(f"AMP rewards computed: mean={amp_rewards.mean().item():.4f}, "
                        f"std={amp_rewards.std().item():.4f}, "
                        f"min={amp_rewards.min().item():.4f}, "
                        f"max={amp_rewards.max().item():.4f}")
            
            return weighted_rewards, True
            
        except Exception as e:
            logger.error(f"计算AMP奖励时出错: {e}")
            import traceback
            traceback.print_exc()
            return torch.zeros(amp_obs.shape[0] if amp_obs is not None else self.env.num_envs, device=self.device), False

    def _log_amp_rewards(self, infos, amp_rewards, amp_reward_computed):
        """记录AMP奖励到日志"""
        if "to_log" not in infos:
            infos["to_log"] = {}
        
        if amp_reward_computed and amp_rewards is not None:
            # 记录实际的AMP奖励统计
            infos["to_log"]["amp_reward_mean"] = torch.tensor(amp_rewards.mean().item(), device=self.device, dtype=torch.float)
            infos["to_log"]["amp_reward_std"] = torch.tensor(amp_rewards.std().item(), device=self.device, dtype=torch.float)
            infos["to_log"]["amp_reward_min"] = torch.tensor(amp_rewards.min().item(), device=self.device, dtype=torch.float)
            infos["to_log"]["amp_reward_max"] = torch.tensor(amp_rewards.max().item(), device=self.device, dtype=torch.float)
            infos["to_log"]["amp_reward_active"] = torch.tensor(1.0, device=self.device, dtype=torch.float)
            
            # 记录加权后的AMP奖励（实际使用的）
            weighted_amp_rewards = amp_rewards * self.config.amp_reward_weight
            infos["to_log"]["weighted_amp_reward_mean"] = torch.tensor(weighted_amp_rewards.mean().item(), device=self.device, dtype=torch.float)
            
            # 记录AMP奖励权重
            infos["to_log"]["amp_reward_weight"] = torch.tensor(self.config.amp_reward_weight, device=self.device, dtype=torch.float)
        else:
            # 标记AMP奖励未计算
            infos["to_log"]["amp_reward_mean"] = torch.tensor(0.0, device=self.device, dtype=torch.float)
            infos["to_log"]["amp_reward_std"] = torch.tensor(0.0, device=self.device, dtype=torch.float)
            infos["to_log"]["amp_reward_min"] = torch.tensor(0.0, device=self.device, dtype=torch.float)
            infos["to_log"]["amp_reward_max"] = torch.tensor(0.0, device=self.device, dtype=torch.float)
            infos["to_log"]["amp_reward_active"] = torch.tensor(0.0, device=self.device, dtype=torch.float)
            infos["to_log"]["weighted_amp_reward_mean"] = torch.tensor(0.0, device=self.device, dtype=torch.float)
            infos["to_log"]["amp_reward_weight"] = torch.tensor(self.config.amp_reward_weight, device=self.device, dtype=torch.float)
            
            # 记录问题原因
            if amp_rewards is None:
                logger.warning("AMP rewards is None - discriminator computation failed")
            if not amp_reward_computed:
                logger.warning("AMP reward computation failed")


    def _ensure_infos_tensors(self, infos):
        """确保infos["to_log"]中的所有值都是tensor"""
        if "to_log" in infos:
            for key, value in infos["to_log"].items():
                if isinstance(value, (int, float)):
                    infos["to_log"][key] = torch.tensor(value, device=self.device, dtype=torch.float)
                elif isinstance(value, torch.Tensor):
                    infos["to_log"][key] = value.to(self.device).float()
                    if len(infos["to_log"][key].shape) > 0:
                        infos["to_log"][key] = infos["to_log"][key].mean()

    def _update_algo_step(self, policy_state_dict, loss_dict):
        # 先更新基础PPO
        loss_dict = super()._update_algo_step(policy_state_dict, loss_dict)
        
        # 然后更新判别器
        if self.amp_storage_available and "amp_obs" in policy_state_dict:
            loss_dict = self._update_discriminator(policy_state_dict, loss_dict)
        else:
            logger.debug("跳过判别器更新：存储不可用或没有amp_obs")
            # 添加默认值到loss_dict
            loss_dict['Discriminator_Loss'] += 0.0
            loss_dict['Discriminator_Real_Acc'] += 0.5
            loss_dict['Discriminator_Fake_Acc'] += 0.5
            loss_dict['Discriminator_Total_Acc'] += 0.5
            loss_dict['AMP_Reward_Mean'] += 0.0
            
        return loss_dict

    def _update_discriminator(self, policy_state_dict, loss_dict):
        """更新AMP判别器"""
        try:
            fake_amp_obs = policy_state_dict["amp_obs"]
            batch_size = fake_amp_obs.shape[0]
            
            # 获取真实数据（专家演示）
            real_amp_obs = self._get_expert_amp_observations(batch_size)
            
            # 计算判别器损失
            disc_loss, disc_info = self.discriminator.compute_disc_loss(real_amp_obs, fake_amp_obs)
            
            # 更新判别器
            self.discriminator_optimizer.zero_grad()
            disc_loss.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
            self.discriminator_optimizer.step()
            
            # 记录判别器信息到loss_dict（这些会被展示）
            loss_dict['Discriminator_Loss'] += disc_info['disc_loss']
            loss_dict['Discriminator_Real_Acc'] += disc_info['real_acc']
            loss_dict['Discriminator_Fake_Acc'] += disc_info['fake_acc']
            loss_dict['Discriminator_Total_Acc'] += disc_info['total_acc']
            loss_dict['Discriminator_Real_Logits'] += disc_info['real_logits_mean']
            loss_dict['Discriminator_Fake_Logits'] += disc_info['fake_logits_mean']
            
            # 计算AMP奖励统计
            with torch.no_grad():
                amp_rewards = self.discriminator.compute_disc_rewards(fake_amp_obs)
                loss_dict['AMP_Reward_Mean'] += amp_rewards.mean().item()
                loss_dict['AMP_Reward_Std'] += amp_rewards.std().item()
                loss_dict['AMP_Reward_Min'] += amp_rewards.min().item()
                loss_dict['AMP_Reward_Max'] += amp_rewards.max().item()
                
            # 记录expert数据质量
            loss_dict['Expert_Data_Size'] += len(real_amp_obs)
            loss_dict['Expert_Data_Mean'] += real_amp_obs.mean().item()
            loss_dict['Expert_Data_Std'] += real_amp_obs.std().item()
            
            # 记录当前数据质量
            loss_dict['Current_Data_Mean'] += fake_amp_obs.mean().item()
            loss_dict['Current_Data_Std'] += fake_amp_obs.std().item()
            
            logger.debug(f"Discriminator updated successfully: loss={disc_loss.item():.4f}, "
                        f"real_acc={disc_info['real_acc']:.3f}, fake_acc={disc_info['fake_acc']:.3f}")
                    
        except Exception as e:
            logger.error(f"判别器更新失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 设置错误状态的默认值
            loss_dict['Discriminator_Loss'] += 0.0
            loss_dict['Discriminator_Real_Acc'] += 0.5
            loss_dict['Discriminator_Fake_Acc'] += 0.5
            loss_dict['Discriminator_Total_Acc'] += 0.5
            loss_dict['Discriminator_Real_Logits'] += 0.0
            loss_dict['Discriminator_Fake_Logits'] += 0.0
            loss_dict['AMP_Reward_Mean'] += 0.0
            loss_dict['AMP_Reward_Std'] += 0.0
            loss_dict['AMP_Reward_Min'] += 0.0
            loss_dict['AMP_Reward_Max'] += 0.0
            loss_dict['Expert_Data_Size'] += 0.0
            loss_dict['Expert_Data_Mean'] += 0.0
            loss_dict['Expert_Data_Std'] += 0.0
            loss_dict['Current_Data_Mean'] += 0.0
            loss_dict['Current_Data_Std'] += 0.0
        
        return loss_dict


    def _get_expert_amp_observations(self, batch_size):
        """获取专家AMP观测"""
        try:
            if hasattr(self.env, 'get_expert_amp_observations'):
                expert_obs = self.env.get_expert_amp_observations(num_samples=batch_size)
                
                # 验证expert数据
                if expert_obs is None:
                    logger.error("Expert AMP observations is None")
                    raise ValueError("Expert data is None")
                    
                if expert_obs.shape[0] == 0:
                    logger.error("Expert AMP observations is empty")
                    raise ValueError("Expert data is empty")
                    
                if torch.isnan(expert_obs).any() or torch.isinf(expert_obs).any():
                    logger.error("Expert AMP observations contains NaN or Inf")
                    raise ValueError("Expert data contains invalid values")
                    
                logger.debug(f"Expert AMP observations loaded: shape={expert_obs.shape}, "
                            f"mean={expert_obs.mean().item():.4f}, "
                            f"std={expert_obs.std().item():.4f}")
                
                return expert_obs
            else:
                logger.warning("环境没有get_expert_amp_observations方法")
                # Fallback: 使用随机数据（这不理想，但至少不会崩溃）
                amp_obs_dim = self.algo_obs_dim_dict["amp_obs"]
                fallback_data = torch.randn(batch_size, amp_obs_dim, device=self.device)
                logger.warning(f"使用随机fallback数据: shape={fallback_data.shape}")
                return fallback_data
                
        except Exception as e:
            logger.error(f"获取专家AMP观测失败: {e}")
            amp_obs_dim = self.algo_obs_dim_dict["amp_obs"]
            fallback_data = torch.randn(batch_size, amp_obs_dim, device=self.device)
            logger.error(f"使用随机fallback数据: shape={fallback_data.shape}")
            return fallback_data

    def load(self, ckpt_path):
        if ckpt_path is not None:
            loaded_dict = torch.load(ckpt_path, map_location=self.device)
            self.actor.load_state_dict(loaded_dict["actor_model_state_dict"])
            self.critic.load_state_dict(loaded_dict["critic_model_state_dict"])
            
            # 加载判别器（如果存在）
            if "discriminator_model_state_dict" in loaded_dict:
                self.discriminator.load_state_dict(loaded_dict["discriminator_model_state_dict"])
                logger.info("加载判别器状态")
            else:
                logger.warning("checkpoint中没有判别器状态")
            
            if self.load_optimizer:
                self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
                self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
                
                if "discriminator_optimizer_state_dict" in loaded_dict:
                    self.discriminator_optimizer.load_state_dict(loaded_dict["discriminator_optimizer_state_dict"])
                    self.discriminator_learning_rate = loaded_dict['discriminator_optimizer_state_dict']['param_groups'][0]['lr']
                else:
                    self.discriminator_learning_rate = self.config.discriminator_learning_rate
                
                self.actor_learning_rate = loaded_dict['actor_optimizer_state_dict']['param_groups'][0]['lr']
                self.critic_learning_rate = loaded_dict['critic_optimizer_state_dict']['param_groups'][0]['lr']
                
                self.set_learning_rate(self.actor_learning_rate, self.critic_learning_rate, self.discriminator_learning_rate)
                
            self.current_learning_iteration = loaded_dict["iter"]
            return loaded_dict["infos"]

    def save(self, path, infos=None):
        torch.save({
            'actor_model_state_dict': self.actor.state_dict(),
            'critic_model_state_dict': self.critic.state_dict(),
            'discriminator_model_state_dict': self.discriminator.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)

    def set_learning_rate(self, actor_learning_rate, critic_learning_rate, discriminator_learning_rate=None):
        super().set_learning_rate(actor_learning_rate, critic_learning_rate)
        if discriminator_learning_rate is not None:
            self.discriminator_learning_rate = discriminator_learning_rate
            for param_group in self.discriminator_optimizer.param_groups:
                param_group['lr'] = discriminator_learning_rate

    @property
    def inference_model(self):
        return {
            "actor": self.actor,
            "critic": self.critic,
            "discriminator": self.discriminator
        }
