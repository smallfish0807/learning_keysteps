import matplotlib.pyplot as plt
import numpy as np
import os


def read_baseline_data(name, bl=''):
    baseline_data = []
    with open('baseline_result/DQN_{}_1'.format(name)) as f:
        for line in f:
            line_list = line.rstrip().lstrip().split()
            try:
                float_list = [float(x) for x in line_list]
            except ValueError:
                continue
            if float_list:
                float_list.append(0.)
                float_list.append(0.)
                baseline_data.append(float_list)
    for heur in ['random', 'attack_large_value', 'attack_large_prob_gap']:
        with open('baseline_result/DQN_{}_1_{}{}'.format(name, heur, bl)) as f:
            for line in f:
                line_list = line.rstrip().lstrip().split()
                try:
                    float_list = [float(x) for x in line_list]
                except ValueError:
                    continue
                if float_list:
                    baseline_data.append(float_list)
    return baseline_data


def read_method_data(name, bl='', verbose=False):
    method_data = []
    method_data_mean = []
    for eta in ['eta10_', 'eta1_', 'eta0.1_', 'eta0.01_', 'eta0.001_']:
        filename = 'RL_result/result_{}_tf_1{}_{}real'.format(name, bl, eta)
        try:
            with open(filename) as f:
                for line in f:
                    line_list = line.rstrip().lstrip().split()
                    try:
                        float_list = [float(x) for x in line_list[:2]]
                    except ValueError:
                        continue
                    if float_list:
                        method_data.append(float_list)
                    try:
                        float_list = [float(x) for x in line_list]
                    except ValueError:
                        continue
                    if float_list:
                        method_data_mean.append(float_list)
            method_data = method_data[:-1]
        except FileNotFoundError:
            if verbose:
                print('{} not finished'.format(filename))
    return method_data, method_data_mean


def plot_baseline(baseline_data):
    attack_proportions = np.linspace(0, 1, 11)
    random = [baseline_data[0]] + [baseline_data[i] for i in range(1, 11)]
    value = [baseline_data[0]] + [baseline_data[i] for i in range(11, 21)]
    prob = [baseline_data[0]] + [baseline_data[i] for i in range(21, 31)]

    coef = 1
    plt.plot(attack_proportions, [x[0] for x in random],
             '.-',
             label='random',
             color='blue')
    plt.fill_between(attack_proportions, [x[0] + coef * x[1] for x in random],
                     [x[0] - coef * x[1] for x in random],
                     facecolor='blue',
                     alpha=0.33)
    plt.plot(attack_proportions, [x[0] for x in value],
             '.-',
             label='large-value',
             color='orange')
    plt.fill_between(attack_proportions, [x[0] + coef * x[1] for x in value],
                     [x[0] - coef * x[1] for x in value],
                     facecolor='orange',
                     alpha=0.33)
    plt.plot(attack_proportions, [x[0] for x in prob],
             '.-',
             label='large-prob gap',
             color='green')
    plt.fill_between(attack_proportions, [x[0] + coef * x[1] for x in prob],
                     [x[0] - coef * x[1] for x in prob],
                     facecolor='green',
                     alpha=0.33)


def plot_method(method_data, method_data_mean, bdata):
    coef = 1
    method_data_mean.sort(key=lambda x: x[2])
    while method_data_mean[0][2] < 1e-4:
        method_data_mean = method_data_mean[1:]
    method_data_mean = [[bdata[0][0], bdata[0][1], 0., 0.]] + method_data_mean
    plt.plot([x[2] for x in method_data_mean],
             [x[0] for x in method_data_mean],
             '.-',
             label='RL',
             color='red')
    plt.fill_between([x[2] for x in method_data_mean],
                     [x[0] + coef * x[1] for x in method_data_mean],
                     [x[0] - coef * x[1] for x in method_data_mean],
                     facecolor='red',
                     alpha=0.33)
    plt.scatter([x[1] for x in method_data if x[1] > 1e-4],
                [x[0] for x in method_data if x[1] > 1e-4],
                color='red',
                alpha=0.5,
                marker='.')


def pf(x, n=0, small=False):
    ans = ''
    y = x
    for _ in range(n):
        if y < np.power(10, n):
            y *= 10.
            ans += '\\,\\,\\,'
        else:
            break
    if small:
        return '{{\\scriptsize{}({:.1f})}}'.format(ans, x)
    else:
        return '{}{:.1f}'.format(ans, x)


def print_method_data_mean(name, mdm, n=2):
    print('\\midrule')
    print('\\multirow{{2}}{{*}}{{{}}} & return & '.format(name[0].upper() +
                                                          name[1:]),
          end='')
    mdm_print = ['{} {}'.format(pf(x[0]), pf(x[1], n, True)) for x in mdm]
    print(*mdm_print, sep=' & ', end='')
    print('\\\\')
    print('& attack ratio (\\%) & ', end='')
    mdm_print = [
        '{} {}'.format(pf(x[2] * 100), pf(x[3] * 100, n, True)) for x in mdm
    ]
    print(*mdm_print, sep=' & ', end='')
    print('\\\\')


def plot_training_curve(directory, bl, name, eta, max_step):
    x, y = [], []
    with open('logs/DQN_attack_{}_tf_1{}{}_real.yaml/logger'.format(
            name, bl, eta)) as fp:
        if eta.find('_step200') >= 0:
            eta_float = float(eta[4:-8])
        else:
            eta_float = float(eta[4:])
        for line in fp:
            linelist = line.rstrip().lstrip().split()
            x.append(int(linelist[0]))
            y.append(-float(linelist[1]) -
                     float(linelist[-1][:-1]) / 10 * eta_float)
    plt.plot(x, y)
    if bl == '':
        bl = '_wh'
    else:
        bl = '_bl'
    plt.xlim(0, max_step)
    plt.savefig(directory + '{}{}{}.pdf'.format(name, bl, eta))
    plt.close()


def plot_training_curve_toy(directory):
    plt.plot(list(range(1000)), [-1] * 1000,
             dashes=[6, 2],
             label=r'heuristics ($B=2$)',
             color='blue',
             alpha=0.75)
    etas = [10, 1, 0.1]
    colors = ['orange', 'green', 'red']

    # plot target agent's return
    for idx in range(3):
        x, y = [], []
        with open('logs/DQN_toymdp{}.yaml/logger'.format(idx + 1)) as fp:
            for line in fp:
                linelist = line.rstrip().lstrip().split()
                x.append(int(linelist[0]))
                y.append(-float(linelist[1]) -
                         float(linelist[-1][:-1]) / 10 * etas[idx])
            plt.plot(x[::2],
                     y[::2],
                     '.-',
                     label=r'RL ($\lambda=10^{{{}}}$)'.format(1 - idx),
                     color=colors[idx],
                     alpha=0.5)
    plt.xlim(0, 1000)
    plt.legend(loc='upper right')
    plt.xlabel('training steps')
    plt.ylabel('return of target agent')
    plt.tight_layout()
    plt.savefig(directory + 'toy.pdf')
    plt.close()


if __name__ == '__main__':

    plt.rcParams.update({'font.size': 16})
    directory = 'figs/result/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    axes = {
        'pong': [0, 1, -21, 19],
        'spaceinvaders': [0, 1, 0, 2100],
        'seaquest': [0, 1, 0, 2500],
        'riverraid': [0, 1, 0, 20000]
    }

    # Plot comparison and print table
    for bl in [('', '', '_wh', 2), ('_bl', '_source_5', '_bl', 3)]:
        for name in ['pong', 'spaceinvaders', 'seaquest', 'riverraid']:
            # Plot comparison
            baseline_data = read_baseline_data(name, bl[0])
            plot_baseline(baseline_data)
            method_data, method_data_mean = read_method_data(name,
                                                             bl[1],
                                                             verbose=True)
            # Make the zero attack cases consistent
            method_data_mean = [
                baseline_data[0] if x[2] < 1e-4 else x
                for x in method_data_mean
            ]
            plot_method(method_data, method_data_mean, baseline_data)
            plt.legend(loc='upper right')
            plt.axis(axes[name])
            plt.savefig(directory + name + '{}.pdf'.format(bl[2]))
            plt.close()

            # Print table
            print_method_data_mean(name, method_data_mean, bl[3])
    '''
    # Plot training curve
    # directory = 'figs/training/'
    # if not os.path.exists(directory):
        # os.makedirs(directory)
    # for bl in ['', '_source_5']:
        # for name in ['pong', 'spaceinvaders', 'seaquest', 'riverraid']:
            # for eta in ['_eta10', '_eta1', '_eta0.1', '_eta0.01', '_eta0.001']:
                # plot_training_curve(directory, bl, name, eta, 10000000)

    # Plot training curve for step200
    # directory = 'figs/training_step200/'
    # if not os.path.exists(directory):
        # os.makedirs(directory)
    # for name, eta in [('pong', '_eta0.1_step200'),
                      # ('spaceinvaders', '_eta1_step200'),
                      # ('seaquest', '_eta1_step200'),
                      # ('riverraid', '_eta10_step200')]:
        # plot_training_curve(directory, '', name, eta, 200000000)

    # Plot training curve for toymdp
    # directory = 'figs/'
    # if not os.path.exists(directory):
        # os.makedirs(directory)
    # plot_training_curve_toy(directory)'''
