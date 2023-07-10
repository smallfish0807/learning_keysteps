from importlib import import_module

import numpy as np
import torch

import gym
from wrapper import AtariPreprocessing, ImageToPyTorch


def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):
    """Returns the current epsilon for the agent's epsilon-greedy policy.
    Begin at 1. until warmup_steps steps have been taken; then
    Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
    Use epsilon from there on.
    """
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon) * steps_left / decay_period
    bonus = np.clip(bonus, 0., 1. - epsilon)
    return epsilon + bonus


def load_model(load_path, spec, env):
    device = torch.device('cuda:{}'.format(spec.get('GPU', 0)) if torch.cuda.
                          is_available() else 'cpu')
    algo_to_model = {
        'DQN': 'DeepQN',
    }
    algo_module = import_module('model')
    modelclass = getattr(algo_module, algo_to_model[spec['algorithm']])
    model_kwargs = {
        'device': device,
        'input_shape': env.observation_space.shape,
        'num_actions': env.action_space.n,
        'exploration': spec['model']['exploration'],
        'use_cnn': spec['model']['atari']
    }
    model_special_params = spec['model'].copy()

    # Remove unneeded arguments
    model_special_params.pop('atari', None)
    model_special_params.pop('exploration', None)

    # Create model and load parameters
    model = modelclass(**model_kwargs, **model_special_params)

    model.load_state_dict(torch.load(load_path))
    model = model.to(device)
    return model


def create_new_env(spec):

    env = gym.make(spec['env'])
    if 'AttackTiming' in spec['env']:
        env.my_init(spec['spec'], spec['load_path'], spec['attack'])
    elif spec['model']['atari']:
        # Strip out the TimeLimit wrapper from Gym, which caps us at 100k
        # frames. We handle this time limit internally instead, which lets us
        # cap at 108k frames (30 minutes). The TimeLimit wrapper also plays
        # poorly with saving and restoring states.
        env = env.env
        env = AtariPreprocessing(env)
        env = ImageToPyTorch(env)
    return env
