import yaml
import matplotlib.pyplot as plt
import shutil
import os
import sys

import random
import torch
import torch.nn.functional as F
import numpy as np
import foolbox
from foolbox.criteria import TargetClass
from gradient_attack import FGSM

from parser import get_attack_parser
from utils import load_model, create_new_env
from utils_tf import load_model_tf
import envs  # NOQA


def plot_comparison(state_plot, state_attacked_plot, filename='attack.png'):
    # state_plot = state[:, ::-1, :]
    # state_attacked_plot = state_attacked[:, ::-1, :]

    plt.figure()
    plt.tight_layout()
    plt.subplots_adjust(left=0.05,
                        right=0.95,
                        bottom=0.05,
                        top=0.95,
                        wspace=0.1,
                        hspace=0)

    plt.subplot(1, 3, 1)
    plt.title('Original')
    # division by 255 to convert [0, 255] to [0, 1]
    plt.imshow(state_plot.squeeze() / 255, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Adversarial')
    plt.imshow(state_attacked_plot.squeeze() / 255, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Difference')
    difference = (state_attacked_plot - state_plot).squeeze()
    # plt.imshow(difference / abs(difference).max() * 0.2 + 0.5, cmap='gray')
    plt.imshow(difference / 255.0 * 0.5 + 0.5, cmap='gray')
    plt.axis('off')

    plt.suptitle(
        'Untargeted FGSM on Pong-v4, epsilon = 0.01\n'
        'Attacker Policy: attack_large_prob_gap, attack_proportion = 0.1')
    plt.savefig(filename)
    plt.close()


def attack_by_compare_to_history(value, sorted_history, attack_proportion):
    # 1     -> len-1
    # 2     -> len-2
    # 2.5   -> len-3
    # len   ->  0
    if attack_proportion >= 1:
        return True
    elif attack_proportion <= 0:
        return False
    if sorted_history and value > sorted_history[len(sorted_history) - int(
            np.ceil(len(sorted_history) * attack_proportion))]:
        return True
    else:
        return False


def generate_gif(num_step, directory, filename='pong.gif'):
    import imageio
    images = []
    for i in range(1, num_step):
        images.append(imageio.imread(directory + '{}.png'.format(i)))
    imageio.mimsave(filename, images)


if __name__ == '__main__':

    args = get_attack_parser().parse_args()
    with open(args.config) as fd:
        spec = yaml.load(fd, Loader=yaml.FullLoader)

    if args.attacker_policy == 'rl':
        assert 'AttackTiming' in spec['env']
        assert args.attacker_type == 'torch', \
            "Only attacker_type of 'torch' is supported now."
        assert args.attacker_path is not None, \
            "Please specify the path to load the attacker."

        # Setup attacker
        with open(spec['config_path']) as fd:
            spec['spec'] = yaml.load(fd, Loader=yaml.FullLoader)

        # Setup attacker_model
        env = create_new_env(spec['spec'])
        env.action_space.n = 2  # This env is only for init of attacker_model
        attacker_model = load_model(args.attacker_path, spec, env)
        attacker_model.eval()

        args.decreasing_epsilons = spec['attack'].get('decreasing_epsilons',
                                                      False)
        args.realistic = spec['attack'].get('realistic', False)
        source_agent_path = spec['attack'].get('source_agent_path', None)
        if source_agent_path is not None:
            args.source_agent_seed = int(
                source_agent_path[source_agent_path.find('/tf_checkpoints') -
                                  1])
            args.blackbox = True

        # Replace args.agent_path and spec
        if args.agent_type == 'torch':
            args.agent_path = spec['load_path']
        else:
            lp = spec['load_path']
            args.agent_seed = int(lp[lp.find('/tf_checkpoints') - 1])
        attacker_exploration_name = spec['model']['exploration']['name']
        spec = spec['spec']

    assert 'AttackTiming' not in spec['env']
    # Setup env
    env = create_new_env(spec)
    max_steps_per_episode = spec['common']['max_steps_per_episode']

    # Setup model
    if args.agent_type == 'torch':
        model = load_model(args.agent_path, spec, env)
        model.eval()
    elif args.agent_type == 'tf':
        agent_name = 'dqn'
        game_name = spec['env'][:spec['env'].find('NoFrameskip')]
        ckpt_formatter = 'pretrained/{}/{}/{}/tf_checkpoints/tf_ckpt-199'
        restore_ckpt = ckpt_formatter.format(agent_name, game_name,
                                             args.agent_seed)
        model = load_model_tf(agent_name, game_name, restore_ckpt, 'result_tf')
        model.eval_mode = True
        if args.source_agent_seed > 0:
            # assert args.attacker_policy != 'rl', \
            #    'attacker_policy should not be rl'
            restore_ckpt = ckpt_formatter.format(agent_name, game_name,
                                                 args.source_agent_seed)
            source_model = load_model_tf(agent_name, game_name, restore_ckpt,
                                         'result_tf')
            source_model.eval_mode = True
    else:
        raise ValueError('agent_type {} not supported'.format(args.agent_type))

    # Deterministic Runs
    seed = spec['common'].get('seed', 1126)
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    score_list = []
    score_std_list = []
    ratio_list = []
    ratio_std_list = []

    # Testing
    if args.test:
        episode_reward_list = []
        for _ in range(args.test_num):
            if args.agent_type == 'tf':
                model._reset_state()
            state = env.reset()
            episode_reward = 0
            done = False
            actions = [0] * env.action_space.n

            step_count = 0
            while (not done and step_count != max_steps_per_episode):
                if args.agent_type == 'torch':
                    if spec['model']['exploration'][
                            'name'] == 'epsilon_greedy':
                        action = model.act(
                            state,
                            spec['model']['exploration']['epsilon_test'])
                    elif spec['model']['exploration']['name'] in [
                            'noisy_linear', 'entropy'
                    ]:
                        action = model.act(state)
                    else:
                        raise ValueError(
                            'exploration strategy not implemented')
                else:
                    model._record_observation(state)
                    action = model._select_action()
                actions[action] += 1
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                step_count += 1

            print('{} {}'.format(episode_reward, actions))
            episode_reward_list.append(episode_reward)

        score = np.mean(episode_reward_list)
        std = np.std(episode_reward_list)
        print(score, std)
        sys.exit(0)

    # Attack and test
    if args.agent_type == 'torch':
        fmodel = foolbox.models.PyTorchModel(model,
                                             bounds=(0, 255),
                                             num_classes=env.action_space.n)
        # Reload the model to avoid a weird behavior of fmodel
        # that changes the model device setting
        model = load_model(args.agent_path, spec, env)
        model.eval()
    else:
        if args.blackbox:
            assert args.source_agent_seed > 0, \
                'Source agent is required for blockbox attack'
            print('Performing blockbox attack (perturbation is computed by '
                  'source model)')
            with source_model._sess.as_default():
                fmodel = foolbox.models.TensorFlowModel(
                    source_model.state_ph,
                    source_model._net_outputs.q_values,
                    bounds=(0, 255))
        else:
            with model._sess.as_default():
                fmodel = foolbox.models.TensorFlowModel(
                    model.state_ph,
                    model._net_outputs.q_values,
                    bounds=(0, 255))
    attack = FGSM(fmodel)

    # Attack Settings
    # max_epsilons = np.linspace(0., 0.005, 11)[1:]
    max_epsilons = [0.01]
    epsilons = 100

    if args.attack_proportion < 0:
        attack_proportions = np.linspace(0, 1, 11)[1:]
    else:
        attack_proportions = [args.attack_proportion]

    if args.visualize:
        if os.path.exists(args.directory):
            shutil.rmtree(args.directory)
        os.makedirs(args.directory)

    print('attacker_policy: {}'.format(args.attacker_policy))
    print('targetted: {} visualize: {} decreasing_epsilons: {}'.format(
        args.targetted, args.visualize, args.decreasing_epsilons))
    print('epsilons: {} max_epsilons: {} attack_proportions: {}'.format(
        epsilons, max_epsilons, attack_proportions))

    for max_epsilon in max_epsilons:
        for attack_proportion in attack_proportions:
            print('epsilons: {} max_epsilon: {} attack_proportion: {}'.format(
                epsilons, max_epsilon, attack_proportion))
            if args.decreasing_epsilons:
                epsilon_list = np.linspace(max_epsilon, 0,
                                           num=epsilons + 1)[:-1]
            else:
                epsilon_list = np.linspace(0, max_epsilon,
                                           num=epsilons + 1)[1:]
            episode_reward_list, episode_ratio_list = [], []
            for _ in range(args.test_num):
                if args.agent_type == 'tf':
                    model._reset_state()

                state = env.reset()
                episode_reward = 0
                done = False
                actions = [0] * env.action_space.n

                num_step = 0
                num_step_attacked = 0
                num_step_attack_fail = 0

                if args.attacker_policy in [
                        'attack_large_value', 'attack_large_value_gap',
                        'attack_large_prob_gap'
                ]:
                    sorted_history = []

                while (not done and num_step != max_steps_per_episode):
                    num_step += 1

                    if args.agent_type == 'torch':
                        values = model.values(state)
                        value_max = values.max(1)[0][0].item()
                        value_min = values.min(1)[0][0].item()
                        value_gap = value_max - value_min
                        probs = F.softmax(values[0], dim=0)
                        prob_gap = (probs.max() - probs.min()).item()
                        action = values.max(1)[1][0].item()
                        least_preferred_action = values.min(1)[1][0].item()
                    else:
                        model._record_observation(state)
                        if args.source_agent_seed > 0:
                            values = source_model._sess.run(
                                source_model._net_outputs.q_values,
                                {source_model.state_ph: model.state})
                        else:
                            values = model._sess.run(
                                model._net_outputs.q_values,
                                {model.state_ph: model.state})
                        value_max = values.max()
                        value_min = values.min()
                        value_gap = value_max - value_min
                        probs = np.exp(values) / np.sum(np.exp(values))
                        prob_gap = probs.max() - probs.min()
                        action = values.argmax()
                        least_preferred_action = values.argmin()

                    if args.attacker_policy == 'random':
                        perform_attack = (random.random() < attack_proportion)
                    elif args.attacker_policy == 'attack_large_value':
                        perform_attack = attack_by_compare_to_history(
                            value_max, sorted_history, attack_proportion)
                    elif args.attacker_policy == 'attack_large_value_gap':
                        perform_attack = attack_by_compare_to_history(
                            value_gap, sorted_history, attack_proportion)
                    elif args.attacker_policy == 'attack_large_prob_gap':
                        perform_attack = attack_by_compare_to_history(
                            prob_gap, sorted_history, attack_proportion)
                    elif args.attacker_policy == 'rl':
                        if args.attacker_type == 'torch':
                            if attacker_exploration_name == 'epsilon_greedy':
                                attacker_action = attacker_model.act(state, 0.)
                            elif attacker_exploration_name in [
                                    'noisy_linear', 'entropy'
                            ]:
                                attacker_action = attacker_model.act(state)
                            else:
                                raise ValueError(
                                    'exploration strategy not implemented')
                        else:
                            raise ValueError("Only attacker_type of 'torch' "
                                             "is supported now.")
                        perform_attack = (attacker_action == 1)
                    else:
                        raise ValueError(
                            '{} not supported for attacker_policy'.format(
                                args.attacker_policy))

                    if args.agent_type == 'torch':
                        if perform_attack:
                            num_step_attacked += 1
                            if args.targetted:
                                attack = FGSM(fmodel,
                                              criterion=TargetClass(
                                                  least_preferred_action))
                            state_attacked = attack(state.astype(np.float32),
                                                    action,
                                                    epsilons=epsilon_list,
                                                    max_epsilon=max_epsilon)
                            if state_attacked is None:
                                # Attack fails
                                num_step_attack_fail += 1
                                state_attacked = state
                        else:
                            state_attacked = state
                        if spec['model']['exploration'][
                                'name'] == 'epsilon_greedy':
                            action_attacked = model.act(
                                state_attacked,
                                spec['model']['exploration']['epsilon_test'])
                        elif spec['model']['exploration']['name'] in [
                                'noisy_linear', 'entropy'
                        ]:
                            action_attacked = model.act(state_attacked)
                        else:
                            raise ValueError(
                                'exploration strategy not implemented')
                    else:
                        if args.source_agent_seed > 0 and not args.blackbox:
                            # Set action and least_preferred_action
                            values = model._sess.run(
                                model._net_outputs.q_values,
                                {model.state_ph: model.state})
                            action = values.argmax()
                            least_preferred_action = values.argmin()
                        if perform_attack:
                            num_step_attacked += 1
                            if args.targetted:
                                attack = FGSM(fmodel,
                                              criterion=TargetClass(
                                                  least_preferred_action))
                            state_attacked = attack(model.state.squeeze(0),
                                                    action,
                                                    epsilons=epsilon_list,
                                                    max_epsilon=max_epsilon,
                                                    realistic=args.realistic)
                            if state_attacked is None:
                                # Attack fails
                                # num_step_attack_fail += 1
                                state_attacked = model.state
                            else:
                                state_attacked = state_attacked[np.newaxis]
                        else:
                            state_attacked = model.state

                        if random.random() <= 0.001:
                            action_attacked = random.randint(
                                0, model.num_actions - 1)
                        else:
                            action_attacked = model._sess.run(
                                model._q_argmax,
                                {model.state_ph: state_attacked})
                            if perform_attack and \
                                action_attacked == model._sess.run(
                                    model._q_argmax,
                                    {model.state_ph: model.state}):  # NOQA
                                num_step_attack_fail += 1

                    if args.attacker_policy == 'attack_large_value':
                        sorted_history.append(value_max)
                        sorted_history.sort()
                    elif args.attacker_policy == 'attack_large_value_gap':
                        sorted_history.append(value_gap)
                        sorted_history.sort()
                    elif args.attacker_policy == 'attack_large_prob_gap':
                        sorted_history.append(prob_gap)
                        sorted_history.sort()

                    # Plot each step
                    if args.visualize:
                        if args.agent_type == 'torch':
                            plot_comparison(
                                state, state_attacked,
                                args.directory + '{}.png'.format(num_step))
                        else:
                            plot_comparison(
                                model.state[0, :, :, 3],
                                state_attacked[0, :, :, 3],
                                args.directory + '{}.png'.format(num_step))

                    # Realistic attack
                    if args.realistic:
                        model.state = state_attacked
                    actions[action_attacked] += 1
                    state, reward, done, _ = env.step(action_attacked)
                    episode_reward += reward

                try:
                    print('{} {} {} {}'.format(
                        episode_reward,
                        float(num_step_attacked) / num_step,
                        float(num_step_attack_fail) / num_step_attacked,
                        actions))
                except ZeroDivisionError:
                    print('{} {} {}'.format(
                        episode_reward,
                        float(num_step_attacked) / num_step, actions))
                episode_reward_list.append(episode_reward)
                episode_ratio_list.append(float(num_step_attacked) / num_step)

            score = np.mean(episode_reward_list)
            score_std = np.std(episode_reward_list)
            ratio = np.mean(episode_ratio_list)
            ratio_std = np.std(episode_ratio_list)

            print(score, score_std, ratio, ratio_std)
            score_list.append(score)
            score_std_list.append(score_std)
            ratio_list.append(ratio)
            ratio_std_list.append(ratio_std)

    print(score_list)
    print(score_std_list)
    print(ratio_list)
    print(ratio_std_list)

    # Generate gif
    if args.visualize:
        generate_gif(num_step, args.directory, filename=args.filename)
