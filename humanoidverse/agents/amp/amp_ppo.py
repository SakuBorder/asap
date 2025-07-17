import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from humanoidverse.agents.ppo.ppo import PPO
from humanoidverse.agents.modules.amp_modules import AMPDiscriminator
from loguru import logger

class AMPPPO(PPO):
    def __init__(self, env, config, log_dir=None, device='cpu'):
        super().__init__(env, config, log_dir, device)

    def _setup_models_and_optimizer(self):
        super()._setup_models_and_optimizer()
        
        # 确保 algo_obs_dim_dict 中包含 amp_obs
        if "amp_obs" not in self.algo_obs_dim_dict:
            logger.warning("algo_obs_dim_dict中没有amp_obs，正在添加...")
            # 计算AMP观测维度: dof_pos + dof_vel + base_lin_vel + base_ang_vel
            dof_obs_size = self.env.config.robot.dof_obs_size
            amp_obs_dim = 2 * dof_obs_size + 6  # dof_pos + dof_vel + 3 + 3
            self.algo_obs_dim_dict["amp_obs"] = amp_obs_dim
            logger.info(f"自动添加amp_obs维度: {amp_obs_dim}")
        
        # 添加AMP判别器
        self.discriminator = AMPDiscriminator(
            self.algo_obs_dim_dict,
            self.config.module_dict.discriminator
        ).to(self.device)
        
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.discriminator_learning_rate
        )
        
        logger.info(f"AMP判别器设置完成，学习率: {self.config.discriminator_learning_rate}")

    def _setup_storage(self):
        super()._setup_storage()
        
        # 注册 amp_obs 存储
        try:
            # 确保维度存在
            if "amp_obs" not in self.algo_obs_dim_dict:
                dof_obs_size = self.env.config.robot.dof_obs_size
                amp_obs_dim = 2 * dof_obs_size + 6
                self.algo_obs_dim_dict["amp_obs"] = amp_obs_dim
                logger.info(f"在存储设置阶段添加amp_obs维度: {amp_obs_dim}")
            
            amp_obs_dim = self.algo_obs_dim_dict["amp_obs"]
            self.storage.register_key('amp_obs', shape=(amp_obs_dim,), dtype=torch.float)
            logger.info(f"成功注册AMP观测存储，维度: {amp_obs_dim}")
            
        except Exception as e:
            logger.error(f"AMP观测存储注册失败: {e}")
            # 强制添加维度并重试
            dof_obs_size = self.env.config.robot.dof_obs_size
            amp_obs_dim = 2 * dof_obs_size + 6
            self.algo_obs_dim_dict["amp_obs"] = amp_obs_dim
            try:
                self.storage.register_key('amp_obs', shape=(amp_obs_dim,), dtype=torch.float)
                logger.info(f"重试成功，注册AMP观测存储，维度: {amp_obs_dim}")
            except Exception as e2:
                logger.error(f"重试也失败: {e2}")

    def _init_loss_dict_at_training_step(self):
        loss_dict = super()._init_loss_dict_at_training_step()
        loss_dict['Discriminator_Loss'] = 0.0
        loss_dict['Discriminator_Real_Acc'] = 0.0
        loss_dict['Discriminator_Fake_Acc'] = 0.0
        loss_dict['Discriminator_Total_Acc'] = 0.0
        loss_dict['AMP_Reward_Mean'] = 0.0
        loss_dict['AMP_Reward_Weight'] = float(self.config.amp_reward_weight)
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
                if hasattr(self.env, '_compute_amp_observations'):
                    amp_obs = self.env._compute_amp_observations()
                    logger.debug(f"从环境获取AMP观测，形状: {amp_obs.shape}")
                elif hasattr(self.env, '_get_obs_amp_obs'):
                    amp_obs = self.env._get_obs_amp_obs()
                    logger.debug(f"使用_get_obs_amp_obs获取AMP观测，形状: {amp_obs.shape}")
                elif 'amp_obs' in obs_dict:
                    amp_obs = obs_dict['amp_obs']
                    logger.debug(f"从obs_dict获取AMP观测，形状: {amp_obs.shape}")
                else:
                    # Fallback: 构造临时AMP观测
                    logger.warning("无法获取AMP观测，使用临时构造的观测")
                    dof_pos = self.env.simulator.dof_pos
                    dof_vel = self.env.simulator.dof_vel
                    base_lin_vel = self.env.base_lin_vel
                    base_ang_vel = self.env.base_ang_vel
                    amp_obs = torch.cat([dof_pos, dof_vel, base_lin_vel, base_ang_vel], dim=-1)
                    logger.debug(f"构造的AMP观测形状: {amp_obs.shape}")
                
                policy_state_dict["amp_obs"] = amp_obs

                ## Append states to storage
                for obs_key in obs_dict.keys():
                    self.storage.update_key(obs_key, obs_dict[obs_key])

                for obs_ in policy_state_dict.keys():
                    if obs_ in self.storage.stored_keys:  # 只存储已注册的键
                        self.storage.update_key(obs_, policy_state_dict[obs_])
                    else:
                        if obs_ == "amp_obs":
                            logger.warning(f"amp_obs未在存储中注册，跳过存储")
                    
                actions = policy_state_dict["actions"]
                actor_state = {"actions": actions}
                obs_dict, rewards, dones, infos = self.env.step(actor_state)
                
                for obs_key in obs_dict.keys():
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                rewards, dones = rewards.to(self.device), dones.to(self.device)

                # 计算AMP奖励
                amp_reward_computed = False
                try:
                    amp_rewards = self.discriminator.compute_disc_rewards(amp_obs)
                    amp_rewards = amp_rewards * self.config.amp_reward_weight
                    
                    # 添加AMP奖励到总奖励
                    rewards += amp_rewards
                    amp_reward_computed = True
                    
                    # 记录AMP奖励到日志 - 确保所有值都是tensor
                    if "to_log" not in infos:
                        infos["to_log"] = {}
                    infos["to_log"]["amp_reward_mean"] = torch.tensor(amp_rewards.mean().item(), device=self.device, dtype=torch.float)
                    infos["to_log"]["amp_reward_active"] = torch.tensor(1.0, device=self.device, dtype=torch.float)
                    
                except Exception as e:
                    logger.warning(f"计算AMP奖励时出错: {e}")
                    # 设置默认值
                    if "to_log" not in infos:
                        infos["to_log"] = {}
                    infos["to_log"]["amp_reward_mean"] = torch.tensor(0.0, device=self.device, dtype=torch.float)
                    infos["to_log"]["amp_reward_active"] = torch.tensor(0.0, device=self.device, dtype=torch.float)

                # 确保infos["to_log"]中的所有值都是tensor
                if "to_log" in infos:
                    for key, value in infos["to_log"].items():
                        if isinstance(value, (int, float)):
                            infos["to_log"][key] = torch.tensor(value, device=self.device, dtype=torch.float)
                        elif isinstance(value, torch.Tensor):
                            # 确保tensor在正确的设备上
                            infos["to_log"][key] = value.to(self.device).float()
                            # 如果是标量tensor，保持标量形状
                            if len(infos["to_log"][key].shape) == 0:
                                infos["to_log"][key] = infos["to_log"][key]
                            elif len(infos["to_log"][key].shape) > 0:
                                # 如果是多维tensor，取平均值变为标量
                                infos["to_log"][key] = infos["to_log"][key].mean()

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
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
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

    def _update_algo_step(self, policy_state_dict, loss_dict):
        # 先更新基础PPO
        loss_dict = super()._update_algo_step(policy_state_dict, loss_dict)
        
        # 然后更新判别器（如果amp_obs存在）
        if "amp_obs" in policy_state_dict:
            loss_dict = self._update_discriminator(policy_state_dict, loss_dict)
        else:
            logger.warning("policy_state_dict中没有amp_obs，跳过判别器更新")
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
            # 获取真实数据（专家演示）
            batch_size = policy_state_dict["amp_obs"].shape[0]
            
            # 检查环境是否有get_expert_amp_observations方法
            if hasattr(self.env, 'get_expert_amp_observations'):
                real_amp_obs = self.env.get_expert_amp_observations(num_samples=batch_size)
                logger.debug(f"获取专家AMP观测，形状: {real_amp_obs.shape}")
            else:
                logger.warning("环境没有get_expert_amp_observations方法，使用随机数据")
                amp_obs_dim = self.algo_obs_dim_dict["amp_obs"]
                real_amp_obs = torch.randn(batch_size, amp_obs_dim, device=self.device)
            
            fake_amp_obs = policy_state_dict["amp_obs"]
            
            # 计算判别器损失
            disc_loss, disc_info = self.discriminator.compute_disc_loss(real_amp_obs, fake_amp_obs)
            
            # 更新判别器
            self.discriminator_optimizer.zero_grad()
            disc_loss.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
            self.discriminator_optimizer.step()
            
            # 记录判别器信息 - 累积到loss_dict
            loss_dict['Discriminator_Loss'] += disc_info['disc_loss']
            loss_dict['Discriminator_Real_Acc'] += disc_info['real_acc']
            loss_dict['Discriminator_Fake_Acc'] += disc_info['fake_acc']
            loss_dict['Discriminator_Total_Acc'] += disc_info['total_acc']
            
            # 计算AMP奖励统计
            with torch.no_grad():
                amp_rewards = self.discriminator.compute_disc_rewards(fake_amp_obs)
                loss_dict['AMP_Reward_Mean'] += amp_rewards.mean().item()
                
        except Exception as e:
            logger.error(f"判别器更新失败: {e}")
            # 设置默认值避免键错误
            loss_dict['Discriminator_Loss'] += 0.0
            loss_dict['Discriminator_Real_Acc'] += 0.5
            loss_dict['Discriminator_Fake_Acc'] += 0.5
            loss_dict['Discriminator_Total_Acc'] += 0.5
            loss_dict['AMP_Reward_Mean'] += 0.0
        
        return loss_dict

    def load(self, ckpt_path):
        if ckpt_path is not None:
            loaded_dict = torch.load(ckpt_path, map_location=self.device)
            self.actor.load_state_dict(loaded_dict["actor_model_state_dict"])
            self.critic.load_state_dict(loaded_dict["critic_model_state_dict"])
            
            # 加载判别器（如果存在）
            if "discriminator_model_state_dict" in loaded_dict:
                self.discriminator.load_state_dict(loaded_dict["discriminator_model_state_dict"])
            
            if self.load_optimizer:
                self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
                self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
                
                if "discriminator_optimizer_state_dict" in loaded_dict:
                    self.discriminator_optimizer.load_state_dict(loaded_dict["discriminator_optimizer_state_dict"])
                    self.discriminator_learning_rate = loaded_dict['discriminator_optimizer_state_dict']['param_groups'][0]['lr']
                
                self.actor_learning_rate = loaded_dict['actor_optimizer_state_dict']['param_groups'][0]['lr']
                self.critic_learning_rate = loaded_dict['critic_optimizer_state_dict']['param_groups'][0]['lr']
                
                self.set_learning_rate(self.actor_learning_rate, self.critic_learning_rate, 
                                     getattr(self, 'discriminator_learning_rate', self.config.discriminator_learning_rate))
                
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
