import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from replay_buffer import ReplayBuffer
from model import DeepQN
from utils import update_target, linearly_decaying_epsilon


class DQN:
    def __init__(self, env, env_test, logger, save_path, mp, cp, device):
        self.env = env
        self.env_test = env_test
        self.logger = logger
        self.save_path = save_path

        self.exploration = mp['exploration']
        if self.exploration['name'] == 'epsilon_greedy':
            self.epsilon_end = mp['exploration']['epsilon_end']
            self.epsilon_decay = mp['exploration']['epsilon_decay']
            self.epsilon_test = mp['exploration']['epsilon_test']
        else:
            raise ValueError('Exploration strategy not implemented')

        use_cnn = mp.get('atari', False)
        dropout = mp.get('dropout', 0.)
        self.current_model = DeepQN(device, env.observation_space.shape,
                                    env.action_space.n, mp['exploration'],
                                    use_cnn, dropout)
        self.target_model = DeepQN(device, env.observation_space.shape,
                                   env.action_space.n, mp['exploration'],
                                   use_cnn, dropout)

        self.device = device
        self.current_model = self.current_model.to(device)
        self.target_model = self.target_model.to(device)
        self.target_model.eval()

        self.batch_size = cp['batch_size']
        self.gamma = cp['gamma']
        self.optimizer = optim.Adam(self.current_model.parameters(),
                                    lr=cp['lr'])

        self.training_steps = cp['training_steps']
        self.target_update_freq = cp['target_update_freq']
        self.learn_start = cp['learn_start']
        self.learn_freq = cp['learn_freq']
        self.replay_buffer = ReplayBuffer(cp['replay_buffer'])
        self.test_freq = cp['test_freq']
        self.test_num = cp['test_num']
        self.max_steps_per_episode = cp['max_steps_per_episode']

    def compute_td_loss(self, batch_size):
        state, action, reward, next_state, done = self.replay_buffer.sample(
            batch_size)

        # By default, requires_grad = False
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        next_state = torch.tensor(next_state,
                                  dtype=torch.float,
                                  device=self.device)
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)
        done = torch.tensor(np.float32(done), device=self.device)

        q_values = self.current_model(state)
        next_q_values = self.target_model(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.current_model.parameters(), 0.5)
        self.optimizer.step()

        return loss

    def train(self):
        best_score = float('-inf')
        losses = []
        all_rewards = []
        episode_reward = 0
        state = self.env.reset()
        step_count = 0

        # Timing
        start_time = time.time()
        start_step = 1

        for step_idx in range(1, self.training_steps + 1):

            # Training
            self.current_model.train()

            if self.exploration['name'] == 'epsilon_greedy':
                action = self.current_model.act(
                    state,
                    linearly_decaying_epsilon(self.epsilon_decay, step_idx,
                                              self.learn_start,
                                              self.epsilon_end))

            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            step_count += 1
            episode_reward += reward

            if done or step_count == self.max_steps_per_episode:
                all_rewards.append(episode_reward)
                episode_reward = 0
                state = self.env.reset()
                step_count = 0

            if step_idx >= self.learn_start:
                if step_idx % self.learn_freq == 0:
                    loss = self.compute_td_loss(self.batch_size)
                    losses.append(loss.item())

                if step_idx % self.target_update_freq == 0:
                    update_target(self.current_model, self.target_model)

            # Testing
            if step_idx % self.test_freq == 0:
                self.current_model.eval()

                # Timing
                average_steps_per_second = float(step_idx - start_step) / (
                    time.time() - start_time)

                episode_reward_test_list = []
                actions_test = [0] * self.env_test.action_space.n
                for _ in range(self.test_num):
                    state_test = self.env_test.reset()
                    episode_reward_test = 0
                    done_test = False

                    step_count_test = 0
                    while (not done
                           and step_count_test != self.max_steps_per_episode):
                        if self.exploration['name'] == 'epsilon_greedy':
                            action_test = self.current_model.act(
                                state_test, self.epsilon_test)
                        actions_test[action_test] += 1
                        state_test, reward_test, done_test, _ = self.env_test.step(
                            action_test)
                        episode_reward_test += reward_test
                        step_count_test += 1

                    episode_reward_test_list.append(episode_reward_test)

                score = np.mean(episode_reward_test_list)
                print('{:8d} {:8.2f} {:8.4f} {:8.2f} steps per sec {}'.format(
                    step_idx, score, np.mean(losses), average_steps_per_second,
                    actions_test))
                if self.logger is not None:
                    self.logger.write(step_idx, score, np.mean(losses),
                                      actions_test)
                losses = []
                if self.save_path is not None and score > best_score:
                    print('Save model with score {}'.format(score))
                    torch.save(self.current_model.state_dict(), self.save_path)
                    best_score = score

                # Timing
                start_time = time.time()
                start_step = step_idx
