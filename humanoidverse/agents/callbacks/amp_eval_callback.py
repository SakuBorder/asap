# 创建 humanoidverse/agents/callbacks/amp_eval_callback.py
class AMPEvalCallback:
    def __init__(self, config):
        self.config = config
        
    def on_post_eval_env_step(self, actor_state):
        # 记录AMP奖励分布
        if hasattr(actor_state, 'amp_reward'):
            # 记录奖励统计
            pass
        return actor_state