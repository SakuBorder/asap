class AMPTrainer:
    """AMP训练策略管理器"""
    
    def __init__(self, config):
        self.config = config
        self.discriminator_update_freq = config.get('discriminator_update_freq', 1)
        self.generator_update_freq = config.get('generator_update_freq', 1)
        self.use_adaptive_weights = config.get('use_adaptive_weights', False)
        
        # 自适应权重
        self.amp_reward_weight = config.amp_reward_weight
        self.min_amp_weight = config.get('min_amp_weight', 0.1)
        self.max_amp_weight = config.get('max_amp_weight', 2.0)
        
    def should_update_discriminator(self, iteration):
        """判断是否应该更新判别器"""
        return iteration % self.discriminator_update_freq == 0
    
    def should_update_generator(self, iteration):
        """判断是否应该更新生成器（策略）"""
        return iteration % self.generator_update_freq == 0
    
    def adapt_amp_weight(self, disc_acc, target_acc=0.7):
        """根据判别器准确率自适应调整AMP权重"""
        if not self.use_adaptive_weights:
            return self.amp_reward_weight
            
        if disc_acc > target_acc + 0.1:
            # 判别器太强，降低AMP权重
            self.amp_reward_weight = max(self.min_amp_weight, self.amp_reward_weight * 0.99)
        elif disc_acc < target_acc - 0.1:
            # 判别器太弱，增加AMP权重
            self.amp_reward_weight = min(self.max_amp_weight, self.amp_reward_weight * 1.01)
            
        return self.amp_reward_weight