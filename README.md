# Keystep

Python codes for "Learning Key Steps to Attack Deep Reinforcement Learning
Agent". We take pretrained DQN models from `Dopamine`, and use DQN to train
attack policies that identifies the key steps to attack. The adversarial
examples are computed using `Foolbox`.

## Setup

The codes are tested with Python 3.6 and CUDA 10.0

```
# Install dependencies
cd keystep
pip install -r requirements.txt

# Download Dopamine pretrained models from
# https://github.com/google/dopamine/tree/master/docs
# The link is at the bottom of the page (DQN checkpoints). After the download
# is finished, unpack it into the `pretrained` directory.
tar zxvf dqn_checkpoints.tar.gz -C pretrained
```

## Usage

### Train an RL attack policy

```
python main.py {CONFIG_FILE} -l -s {SAVED_MODEL}
```

For example, if you want to train an attack policy in Pong, execute the
following:
```
python main.py config/DQN_attack_pong_0.1_wh.yaml -l -s save/DQN_attack_pong_0.1_wh.pth
```
The configuration file `DQN_attack_pong_0.1_wh.yaml` specifies the target agent
and all training configurations ($\lambda=0.1$, white-box attacks). The
configuration file for other settings and for other environments are all in the
`config` directory.

### Test an RL attack policy

```
python attack.py tf rl {CONFIG_FILE} --attacker_path {SAVED_MODEL} --test_num 10
```

For example, if you want to test an attack policy in Pong after the training is
finished, execute the following:
```
python attack.py tf rl config/DQN_attack_pong_0.1_wh.yaml --attacker_path save/DQN_attack_pong_0.1_wh.pth --test_num 10
```

### Test other attack policies

We compare our RL attack policy to three baselines: `random`, `large_value`,
and `large_prob_gap`. You can test it through
```
python attack.py tf {BASELINE} {ENV_CONFIG_FILE} --agent_seed 1 --test_num 10 --realistic {BLACKBOX_ARGUMENTS}
```

For example, if you want to test the `large_prob_gap` policy in Pong with
white-box attacks, execute the following:
```
python attack.py tf large_prob_gap config/DQN_pong_v0.yaml --agent_seed 1 --test_num 10 --realistic
```
If you want to test the `large_prob_gap` policy in Pong with black-box attacks,
execute the following:
```
python attack.py tf large_prob_gap config/DQN_pong_v0.yaml --agent_seed 1 --test_num 10 --realistic --source_agent_seed 5 --blackbox --decreasing_epsilons
```

### Toy example
The toy example is handled by configuration files `DQN_toymdp_{#}.yaml`, where
each # corresponds to a different penalty parameter. To train an attack policy
for this toy example, execute
```
python main.py config/DQN_toymdp_1.yaml -l -s save/DQN_toymdp.pth
```


## Structure
- `config`: Use `yaml` files to specify all configurations
- `envs`: Customized environments (including the environment for training
attacker and the environment for toy example)
- `algorithm`: RL training algorithm and loss function
- `model`: Network model
- `layer`: Common network layers
- `wrapper`: Wrappers for environments
- `dopamine`: Codes from `dopamine`
- `pretrained`: Store pretrained agents from `dopamine`
- `logs`: Store logging files
- `save`: Store models
- `baseline_result`: Results for `random`, `large_value`, and `large_prob_gap`
- `RL_result`: Results for `RL` attack policy


## References
- [google/dopamine](https://github.com/google/dopamine)
- [bethgelab/foolbox](https://github.com/bethgelab/foolbox)
- [higgsfield/RL-adventure](https://github.com/higgsfield/RL-Adventure)
