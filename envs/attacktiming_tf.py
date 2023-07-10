import numpy as np
import random

import gym
from gym import spaces
import foolbox
from foolbox.criteria import TargetClass
from gradient_attack import FGSM

from utils import create_new_env
from utils_tf import load_model_tf


class AttackTimingTF(gym.Env):
    def __init__(self):
        pass

    def my_init(self, spec, load_path_list, attack_params):

        # Initialize inner env
        self.env = create_new_env(spec)

        # Load model
        agent_name = 'dqn'
        game_name = spec['env'][:spec['env'].find('NoFrameskip')]
        if not isinstance(load_path_list, list):
            load_path_list = [load_path_list]
        self.model_list = []
        for load_path in load_path_list:
            model = load_model_tf(agent_name, game_name, load_path,
                                  'result_tf')
            model.eval_mode = True
            self.model_list.append(model)

        # Setup attack
        self.source_agent_path = attack_params.get('source_agent_path', None)
        if self.source_agent_path is None:
            self.fmodel_list, self.attack_list = [], []
            for model in self.model_list:
                with model._sess.as_default():
                    fmodel = foolbox.models.TensorFlowModel(
                        model.state_ph,
                        model._net_outputs.q_values,
                        bounds=(0, 255))
                    attack = FGSM(fmodel)
                    self.fmodel_list.append(fmodel)
                    self.attack_list.append(attack)
        else:
            # Blackbox attack
            self.source_model = load_model_tf(agent_name, game_name,
                                              self.source_agent_path,
                                              'result_tf')
            self.source_model.eval_mode = True
            with self.source_model._sess.as_default():
                self.fmodel = foolbox.models.TensorFlowModel(
                    self.source_model.state_ph,
                    self.source_model._net_outputs.q_values,
                    bounds=(0, 255))
                self.attack = FGSM(self.fmodel)

        # Valid action:
        # 0: don't attack
        # 1: perform attack
        self.action_space = spaces.Discrete(2)

        # Observation space
        self.observation_space = self.env.observation_space

        # Agent parameters
        self.exploration = spec['model']['exploration']

        # Attacker parameters
        self.eta = attack_params['eta']
        self.max_epsilon = attack_params['max_epsilon']
        self.epsilons = attack_params['epsilons']
        self.targetted = attack_params['targetted']
        if attack_params.get('decreasing_epsilons', False):
            self.epsilons = np.linspace(self.max_epsilon,
                                        0,
                                        num=self.epsilons + 1)[:-1]
        self.realistic = attack_params.get('realistic', False)

        self.model_index = -1

    def step(self, action_attacker):
        # self.num_step += 1
        self.model._record_observation(self.state)

        if action_attacker == 1:
            # Add perturbation to state
            values = self.source_model._sess.run(
                self.source_model._net_outputs.q_values,
                {self.source_model.state_ph: self.model.state})
            action = values.argmax()
            # self.num_step_attacked += 0
            if self.targetted:
                least_preferred_action = values.argmin()
                self.attack = FGSM(
                    self.fmodel, criterion=TargetClass(least_preferred_action))
            state_attacked = self.attack(self.model.state.squeeze(0),
                                         action,
                                         epsilons=self.epsilons,
                                         max_epsilon=self.max_epsilon,
                                         realistic=self.realistic)
            if state_attacked is None:
                # Attack fails
                # self.num_step_attack_fail += 1
                state_attacked = self.model.state
            else:
                state_attacked = state_attacked[np.newaxis]
        else:
            state_attacked = self.model.state

        # Agent perform action and interacts with environment
        if random.random() <= 0.001:
            action_attacked = random.randint(0, self.model.num_actions - 1)
        else:
            action_attacked = self.model._sess.run(
                self.model._q_argmax, {self.model.state_ph: state_attacked})

        # Realistic attack
        if self.realistic:
            self.model.state = state_attacked

        self.state, reward, done, _ = self.env.step(action_attacked)

        # Reward function for attacker
        # Negative instant reward
        reward = -reward
        if action_attacker == 1:
            reward -= self.eta

        return self.state, reward, done, {}

    def reset(self):
        # Update model_index
        # self.model_index = random.randrange(len(self.model_list))
        self.model_index = (self.model_index + 1) % len(self.model_list)
        self.model = self.model_list[self.model_index]
        self.model._reset_state()

        if self.source_agent_path is None:
            self.source_model = self.model
            self.fmodel = self.fmodel_list[self.model_index]
            self.attack = self.attack_list[self.model_index]

        self.state = self.env.reset()

        # self.num_step = 0
        # self.num_step_attacked = 0
        # self.num_step_attack_fail = 0
        return self.state

    def render(self):
        pass

    def seed(self, seed=None):
        self.env.seed(seed)
