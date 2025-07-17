import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import BaseModule

class AMPDiscriminator(nn.Module):
    """AMP判别器，用于区分专家和策略生成的动作"""
    
    def __init__(self, obs_dim_dict, module_config_dict):
        super(AMPDiscriminator, self).__init__()
        self.module = BaseModule(obs_dim_dict, module_config_dict)
        
        # 添加梯度惩罚相关参数
        self.gradient_penalty_weight = module_config_dict.get('gradient_penalty_weight', 10.0)
        self.use_spectral_norm = module_config_dict.get('use_spectral_norm', False)
        
        # 如果使用谱归一化
        if self.use_spectral_norm:
            self._apply_spectral_norm()
            
    def _apply_spectral_norm(self):
        """应用谱归一化到所有线性层"""
        for module in self.module.modules():
            if isinstance(module, nn.Linear):
                nn.utils.spectral_norm(module)
        
    def forward(self, amp_obs):
        """
        Args:
            amp_obs: AMP observation tensor [batch_size, amp_obs_dim]
        Returns:
            logits: Discriminator logits [batch_size, 1]
        """
        return self.module(amp_obs)
    
    def compute_disc_logits(self, amp_obs):
        """计算判别器logits"""
        return self.forward(amp_obs)
    
    def compute_disc_rewards(self, amp_obs):
        """从判别器logits计算AMP奖励"""
        with torch.no_grad():
            disc_logits = self.forward(amp_obs)
            # 使用 -log(1 - sigmoid(logits)) 作为奖励
            disc_probs = torch.sigmoid(disc_logits)
            # 添加数值稳定性
            disc_rewards = -torch.log(torch.clamp(1.0 - disc_probs, min=1e-8, max=1.0 - 1e-8))
            return disc_rewards.squeeze(-1)
    
    def compute_gradient_penalty(self, real_data, fake_data):
        """计算梯度惩罚（WGAN-GP风格）"""
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, device=real_data.device)
        
        # 插值
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        # 计算判别器输出
        disc_interpolated = self.forward(interpolated)
        
        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=disc_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(disc_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # 梯度惩罚
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def compute_disc_loss(self, real_amp_obs, fake_amp_obs, use_wgan=False):
        """计算判别器损失"""
        real_logits = self.forward(real_amp_obs)
        fake_logits = self.forward(fake_amp_obs)
        
        if use_wgan:
            # WGAN损失
            disc_loss = -torch.mean(real_logits) + torch.mean(fake_logits)
            
            # 添加梯度惩罚
            gp = self.compute_gradient_penalty(real_amp_obs, fake_amp_obs)
            disc_loss += self.gradient_penalty_weight * gp
            
        else:
            # 标准GAN损失
            real_loss = F.binary_cross_entropy_with_logits(
                real_logits, torch.ones_like(real_logits)
            )
            fake_loss = F.binary_cross_entropy_with_logits(
                fake_logits, torch.zeros_like(fake_logits)
            )
            disc_loss = 0.5 * (real_loss + fake_loss)
        
        # 计算准确率用于监控
        with torch.no_grad():
            if use_wgan:
                real_acc = (real_logits > 0).float().mean()
                fake_acc = (fake_logits < 0).float().mean()
            else:
                real_acc = (torch.sigmoid(real_logits) > 0.5).float().mean()
                fake_acc = (torch.sigmoid(fake_logits) < 0.5).float().mean()
            total_acc = 0.5 * (real_acc + fake_acc)
        
        disc_info = {
            'disc_loss': disc_loss.item(),
            'real_acc': real_acc.item(),
            'fake_acc': fake_acc.item(), 
            'total_acc': total_acc.item(),
            'real_logits_mean': real_logits.mean().item(),
            'fake_logits_mean': fake_logits.mean().item()
        }
        
        if use_wgan:
            disc_info['gradient_penalty'] = gp.item()
        
        return disc_loss, disc_info