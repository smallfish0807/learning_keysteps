from argparse import ArgumentParser
from argparse import RawTextHelpFormatter


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('config', help='location for configuration file')
    parser.add_argument('-l',
                        '--logging',
                        action='store_true',
                        help='create logging file in logs')
    parser.add_argument('-s',
                        '--save_path',
                        default=None,
                        help='PATH to save model')
    return parser


def get_attack_parser():
    parser = ArgumentParser(
        description=
        """Attack the agent according to attack_policy. Here are some example usages.

        For attack_policy other than 'rl':
        - If agent_type is 'torch':
            python attack.py torch attack_large_prob_gap config/DQN_pong.yaml --agent_path save/DQN_pong.pth
        - If agent_type is 'tf':
            python attack.py tf attack_large_prob_gap config/DQN_pong_v0.yaml --agent_seed 1

        For attack_policy 'rl', for now we support only attacker_type of 'torch'.
        - If agent_type is 'torch':
            python attack.py torch rl config/DQN_attack_pong.yaml --attacker_path save/DQN_attack_pong.pth
        - If agent_type is 'tf':
            python attack.py tf rl config/DQN_attack_pong_tf_1.yaml --attacker_path save/DQN_attack_pong_tf_1.pth""",
        formatter_class=RawTextHelpFormatter)
    parser.add_argument('agent_type',
                        type=str,
                        choices=['torch', 'tf'],
                        help='type of the agent under attack')
    parser.add_argument('attacker_policy',
                        type=str,
                        choices=[
                            'random', 'attack_large_value',
                            'attack_large_value_gap', 'attack_large_prob_gap',
                            'rl'
                        ],
                        help='types of attacker policy')
    parser.add_argument('config',
                        type=str,
                        help='location for configuration file')

    # Agent and attacker
    parser.add_argument('--agent_path',
                        type=str,
                        help='path to load the agent')
    parser.add_argument('--agent_seed',
                        type=int,
                        choices=[1, 2, 3, 4, 5],
                        default=0,
                        help='random seed during training if agent_type=tf')
    parser.add_argument('--attacker_type',
                        type=str,
                        choices=['torch', 'tf'],
                        default='torch')
    parser.add_argument('--attacker_path',
                        type=str,
                        help='path to load the attacker')
    parser.add_argument('--source_agent_seed',
                        type=int,
                        choices=[1, 2, 3, 4, 5],
                        default=0,
                        help='source agent seed for tf agent')

    # Additional attack parameters
    parser.add_argument('--test', action='store_true', help='perform testing')
    parser.add_argument('--test_num',
                        type=int,
                        default=1,
                        help='number of tests and attack_tests '
                        '(default: %(default)s)')
    parser.add_argument('--targetted',
                        action='store_true',
                        help='switch to targetted attack')
    parser.add_argument(
        '--attack_proportion',
        type=float,
        default=-1,
        help='attack proportion for attacker_policy other than rl, '
        '-1 for grid search (default: %(default)f)')
    parser.add_argument('--blackbox',
                        action='store_true',
                        help='switch to blackbox attack')
    parser.add_argument('--decreasing_epsilons',
                        action='store_true',
                        help='set epsilons in decreasing order')
    parser.add_argument('--realistic',
                        action='store_true',
                        help='switch to realistic attack')

    # For visualization
    parser.add_argument('--visualize',
                        action='store_true',
                        help='generate image and gif for visualization')
    parser.add_argument(
        '--directory',
        type=str,
        default='visualize/',
        help='directory to save perturbed images (default: %(default)s)')
    parser.add_argument(
        '--filename',
        type=str,
        default='pong.gif',
        help='filename for the generated gif (default: %(default)s)')

    return parser
