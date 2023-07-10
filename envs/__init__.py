from gym.envs.registration import register

register(id='AttackTimingTF-v0',
         entry_point='envs.attacktiming_tf:AttackTimingTF')

register(id='ToyMDP1-v0', entry_point='envs.toymdp:ToyMDP1')
register(id='ToyMDP2-v0', entry_point='envs.toymdp:ToyMDP2')
register(id='ToyMDP3-v0', entry_point='envs.toymdp:ToyMDP3')
