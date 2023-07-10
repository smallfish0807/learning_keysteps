def config(env, eta, mode):
    upper_env = {
        'pong': 'Pong',
        'spaceinvaders': 'SpaceInvaders',
        'seaquest': 'Seaquest',
        'riverraid': 'Riverraid'
    }
    output = (
        'env: AttackTimingTF-v0\n' + 'GPU: 0\n' + 'algorithm: DQN\n' +
        'config_path: config/DQN_{}_v0.yaml\n'.format(env) +
        'load_path: pretrained/dqn/{}/1/tf_checkpoints/tf_ckpt-199\n'.format(
            upper_env[env]) + 'attack:\n' + '    eta: {}\n'.format(eta) +
        '    max_epsilon: 0.01\n' + '    epsilons: 100\n' +
        '    targetted: False\n' + '    realistic: True\n' + 'model:\n' +
        '    atari: True\n' + '    exploration:\n' +
        '        name: epsilon_greedy\n' + '        epsilon_end: 0.01\n' +
        '        epsilon_decay: 250000\n')
    if mode == 'bl':
        output += (
            '        decreasing_epsilons: True\n' +
            '        source_agent_path: pretrained/dqn/{}/5/tf_checkpoints/tf_ckpt-199\n'
            .format(upper_env[env]))
    output += ('        epsilon_test: 0.001\n' + 'common:\n' +
               '    seed: 1126\n' + '    batch_size: 32\n' +
               '    gamma: 0.99\n' + '    lr: 0.0001\n' +
               '    training_steps: 10000000\n' +
               '    target_update_freq: 1000\n' + '    learn_start: 20000\n' +
               '    learn_freq: 4\n' + '    replay_buffer: 100000\n' +
               '    test_freq: 100000\n' + '    test_num: 10\n' +
               '    max_steps_per_episode: 27000\n')

    return output


if __name__ == '__main__':
    for env in ['pong', 'spaceinvaders', 'seaquest', 'riverraid']:
        for eta in [10, 1, 0.1, 0.01, 0.001]:
            for mode in ['wh', 'bl']:
                with open(
                        'config/DQN_attack_{}_{}_{}.yaml'.format(
                            env, eta, mode), 'w') as fd:
                    fd.write(config(env, eta, mode))
