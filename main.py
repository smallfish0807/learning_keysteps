from importlib import import_module
import yaml

import random
import torch
import numpy as np

from parser import get_parser
from logger import Logger
import envs  # NOQA
from utils import create_new_env

if __name__ == '__main__':

    args = get_parser().parse_args()
    with open(args.config) as fd:
        spec = yaml.load(fd, Loader=yaml.FullLoader)

    ################
    # Global Setup #
    ################
    logger = Logger(args.config) if args.logging else None
    device = torch.device('cuda:{}'.format(spec.get('GPU', 0)) if torch.cuda.
                          is_available() else 'cpu')

    ######################
    # Deterministic Runs #
    ######################
    seed = spec['common'].get('seed', 1126)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #############
    # Env Setup #
    #############
    if 'AttackTiming' in spec['env']:
        with open(spec['config_path']) as fd:
            spec['spec'] = yaml.load(fd, Loader=yaml.FullLoader)

    env = create_new_env(spec)
    env.seed(seed)
    env_test = create_new_env(spec)
    env_test.seed(seed)

    #############
    # Training  #
    #############
    assert 'algorithm' in spec
    algo_module = import_module('algorithm')
    algo = getattr(algo_module, spec['algorithm'])
    algo(env, env_test, logger, args.save_path, spec['model'], spec['common'],
         device).train()

    if logger is not None:
        logger.close()
