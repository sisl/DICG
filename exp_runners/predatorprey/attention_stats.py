import numpy as np
import collections
import tikzplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'
rcParams['mathtext.fontset'] = 'cm'
rcParams['figure.figsize'] = [4, 4]
rcParams['figure.dpi'] = 200

def plot_attn_stats(distance_vs_weight, exp_dir, penalty):
    fig, axs = plt.subplots(1, 3)
    plt.subplots_adjust(wspace=0.4)
    scatter_plt, avg_weight_plt, n_pts_plt = axs[0], axs[1], axs[2]
    distance_vs_avgweight = {}
    distance_vs_stdweight = {}
    distance_vs_n = {}
    for d in distance_vs_weight.keys():
        weights = distance_vs_weight[d]
        distance_vs_avgweight[d] = np.mean(weights)
        distance_vs_stdweight[d] = np.std(weights)
        distance_vs_n[d] = len(weights)
        scatter_plt.scatter([d] * len(weights), weights,
                    color='red', s=0.1,)
        scatter_plt.scatter(d, distance_vs_avgweight[d], 
                    color='black', marker='_')
    scatter_plt.set_xlabel('distance')
    scatter_plt.set_ylabel('attention weight')
    
    distance_vs_avgweight = \
        collections.OrderedDict(sorted(distance_vs_avgweight.items()))
    distance_vs_stdweight = \
        collections.OrderedDict(sorted(distance_vs_stdweight.items()))
    distance_vs_n = \
        collections.OrderedDict(sorted(distance_vs_n.items()))

    _x, _y, _y_err = [], [], []
    for d in distance_vs_avgweight.keys():
        _x.append(d)
        _y.append(distance_vs_avgweight[d])
        _y_err.append(distance_vs_stdweight[d])
    _x = np.array(_x)
    _y = np.array(_y)
    _y_err = np.array(_y_err)
    avg_weight_plt.plot(_x, _y, color='blue')
    avg_weight_plt.fill_between(_x, 
                                _y - _y_err / 2, 
                                _y + _y_err / 2,
                                color='blue', 
                                alpha=0.2)
    avg_weight_plt.set_xlabel('Distance')
    avg_weight_plt.set_ylabel('Avg. Attention Weight')
    # avg_weight_plt.set_title('predatorprey, grid={}, n_agents={}, n_preys={}'.
    #     format(args.grid_size, args.n_agents, args.n_preys))

    _x, _y = [], []
    for d in distance_vs_n.keys():
        _x.append(d)
        _y.append(distance_vs_n[d])
    _y = np.array(_y[1:]) / np.sum(_y[1:])
    n_pts_plt.plot(_x[1:], _y) # discard self attention
    n_pts_plt.set_xlabel('Distance')
    n_pts_plt.set_ylabel('Frequency')
    fig.tight_layout()
    fig.savefig('./plots/penalty={}_AttnWeights.png'.format(penalty))


def plot_attn_stats_simple(distance_vs_weight, exp_dir, penalty, n_samples, de):
    fig, axs = plt.subplots()
    avg_weight_plt = axs
    distance_vs_avgweight = {}
    distance_vs_stdweight = {}
    distance_vs_n = {}
    for d in distance_vs_weight.keys():
        weights = distance_vs_weight[d]
        distance_vs_avgweight[d] = np.mean(weights)
        distance_vs_stdweight[d] = np.std(weights)
        distance_vs_n[d] = len(weights)

    distance_vs_avgweight = \
        collections.OrderedDict(sorted(distance_vs_avgweight.items()))
    distance_vs_stdweight = \
        collections.OrderedDict(sorted(distance_vs_stdweight.items()))
    distance_vs_n = \
        collections.OrderedDict(sorted(distance_vs_n.items()))

    _x, _y, _y_err, _n = [], [], [], []
    for d in distance_vs_avgweight.keys():
        _x.append(d)
        _y.append(distance_vs_avgweight[d])
        _y_err.append(distance_vs_stdweight[d])
        _n.append(distance_vs_n[d])
    _x = np.array(_x)
    _y = np.array(_y)
    _y_err = np.array(_y_err)
    _n = np.array(_n)
    avg_weight_plt.plot(_x, _y, color='blue')
    avg_weight_plt.fill_between(_x, 
                                _y - _y_err / 2 / np.sqrt(n_samples), 
                                _y + _y_err / 2 / np.sqrt(n_samples), 
                                color='blue', 
                                alpha=0.2)
    avg_weight_plt.set_xlabel('Distance')
    avg_weight_plt.set_ylabel('Avg. Attention Weight')
    avg_weight_plt.set_ylim(0, 0.5)
    avg_weight_plt.set_xlim(0, None)
    fig.tight_layout()
    if de:
        fig.savefig('./plots/penalty={}_DE_AttnWeights.png'.format(penalty))
        tikzplotlib.save(figure=fig, filepath='./plots/penalty={}_DE_AttnWeights.tex'.format(penalty))
    else:
        fig.savefig('./plots/penalty={}_CE_AttnWeights.png'.format(penalty))
        tikzplotlib.save(figure=fig, filepath='./plots/penalty={}_CE_AttnWeights.tex'.format(penalty))


def attn_analysis(unseeded_exp_dir, args, seeds, de=False):
    import joblib
    distance_vs_weight = {}

    for seed in seeds:
        exp_dir = unseeded_exp_dir + ('_seed=' + str(seed))
        data = joblib.load(exp_dir + '/params.pkl')
        env = data['env']
        algo = data['algo']
        
        for i_eps in range(args.n_eval_episodes):
            print('Seed = {}, Eval episode: {}/{}'.format(seed, i_eps + 1, args.n_eval_episodes))
            obses = env.reset()
            algo.policy.reset([True])
            for i_step in range(args.max_env_steps):

                actions, agent_infos = algo.policy.get_actions(obses,
                    env.get_avail_actions(), greedy=args.eval_greedy)
                if de:
                    attention_weights = algo.baseline.get_attention_weights(obses)
                    attention_weights_0 = attention_weights[0].detach().numpy()
                else:
                    attention_weights_0 = agent_infos['attention_weights'][0]

                for i_agent in range(env.n_agents):
                    d = np.sqrt(
                        (env.agent_pos[0][0] - env.agent_pos[i_agent][0]) ** 2 
                        + (env.agent_pos[0][1] - env.agent_pos[i_agent][1]) ** 2)
                    
                    if d not in distance_vs_weight.keys():
                        distance_vs_weight[d] = [attention_weights_0[i_agent]]
                    else:
                        distance_vs_weight[d].append(attention_weights_0[i_agent])
                
                obses, _, agent_dones, _ = env.step(actions)

                if agent_dones:
                    if i_step < args.max_env_steps - 1:
                        print('eps {} captured all preys in {} steps'.
                            format(i_eps + 1, i_step + 1))
                    break
            env.close()

    plot_attn_stats_simple(distance_vs_weight, exp_dir, args.penalty, len(seeds), de)

