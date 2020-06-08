import sys
import os

current_file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_path + '/../../')

import socket
import collections
import numpy as np
import argparse
import joblib
import time
from types import SimpleNamespace
import torch
from torch.nn import functional as F

import akro
import garage
from garage import wrap_experiment
from garage.envs import GarageEnv
from garage.experiment.deterministic import set_seed

from envs import SMACWrapper
from dicg.torch.algos import CentralizedMAPPO
from dicg.torch.baselines import DICGCritic
from dicg.torch.policies import DecCategoricalLSTMPolicy
from dicg.experiment.local_runner_wrapper import LocalRunnerWrapper
from dicg.sampler import CentralizedMAOnPolicyVectorizedSampler

def run(args):

    if args.exp_name is None:
        exp_layout = collections.OrderedDict([
            ('dicg{}_de_ppo', args.n_gcn_layers),
            ('incact={}', bool(args.state_include_actions)),
            ('atype={}', args.attention_type),
            ('res={}', bool(args.residual)),
            ('entcoeff={}', args.ent),
            ('map={}', args.map),
            ('difficulty={}', args.difficulty),
            ('bs={:0.0e}', args.bs),
            ('nenvs={}', args.n_envs),
            ('splits={}', args.opt_n_minibatches),
            ('miniepoch={}', args.opt_mini_epochs),
            ('seed={}', args.seed)
        ])

        exp_name = '_'.join(
            [key.format(val) for key, val in exp_layout.items()]
        )

        if args.comment != '':
            exp_name = exp_name + '_' + args.comment
    else:
        exp_name = args.exp_name

    if args.loc is None:
        loc = 'local' if socket.gethostname() in ['Mac', 'cave'] else 'remote'
    else:
        loc = args.loc

    prefix = 'smac'
    id_suffix = ('_' + str(args.run_id)) if args.run_id != 0 else ''
    exp_dir = './data/' + args.loc +'/' + exp_name + id_suffix

    # Enforce
    args.center_adv = False if args.entropy_method == 'max' else args.center_adv
    set_seed(args.seed)

    if args.mode == 'train':
        # making sequential log dir if name already exists
        @wrap_experiment(name=exp_name,
                         prefix=prefix,
                         log_dir=exp_dir,
                         snapshot_mode='last', 
                         snapshot_gap=1)
        
        def train_smac(ctxt=None, args_dict=vars(args)):
            args = SimpleNamespace(**args_dict)
            
            env = SMACWrapper(
                centralized=True,
                map_name=args.map,
                difficulty=args.difficulty,
            )
            env = GarageEnv(env)

            runner = LocalRunnerWrapper(
                ctxt,
                eval=args.eval_during_training,
                n_eval_episodes=args.n_eval_episodes,
                eval_greedy=args.eval_greedy,
                eval_epoch_freq=args.eval_epoch_freq,
                save_env=env.pickleable
            )

            hidden_nonlinearity = F.relu if args.hidden_nonlinearity == 'relu' \
                                    else torch.tanh

            policy = DecCategoricalLSTMPolicy(
                env.spec,
                n_agents=env.n_agents,
                encoder_hidden_sizes=args.policy_encoder_hidden_sizes,
                embedding_dim=args.policy_embedding_dim,
                lstm_hidden_size=args.policy_lstm_hidden_size,
                state_include_actions=args.state_include_actions,
                name='dec_categorical_lstm_policy'
            )


            baseline = DICGCritic(
                env.spec,
                env.n_agents,
                encoder_hidden_sizes=args.dicg_encoder_hidden_sizes,
                embedding_dim=args.dicg_embedding_dim,
                attention_type=args.attention_type,
                n_gcn_layers=args.n_gcn_layers,
                residual=args.residual,
                gcn_bias=args.gcn_bias,
                name='dicg_critic'
            )
            
            # Set max_path_length <= max_steps
            # If max_path_length > max_steps, algo will pad obs
            # obs.shape = torch.Size([n_paths, algo.max_path_length, feat_dim])
            algo = CentralizedMAPPO(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                max_path_length=env.episode_limit, # Notice
                discount=args.discount,
                center_adv=bool(args.center_adv),
                positive_adv=bool(args.positive_adv),
                gae_lambda=args.gae_lambda,
                policy_ent_coeff=args.ent,
                entropy_method=args.entropy_method,
                stop_entropy_gradient=True \
                   if args.entropy_method == 'max' else False,
                clip_grad_norm=args.clip_grad_norm,
                optimization_n_minibatches=args.opt_n_minibatches,
                optimization_mini_epochs=args.opt_mini_epochs,
            )
            
            runner.setup(algo, env,
                sampler_cls=CentralizedMAOnPolicyVectorizedSampler, 
                sampler_args={'n_envs': args.n_envs})
            runner.train(n_epochs=args.n_epochs, 
                         batch_size=args.bs)

        train_smac(args_dict=vars(args))

    elif args.mode in ['restore', 'eval']:
        env = SMACWrapper(
            centralized=True,
            map_name=args.map,
            difficulty=args.difficulty,
            replay_dir=exp_dir,
            replay_prefix='dicg_de_lstm',
        )
        if args.mode == 'restore':
            from dicg.experiment.runner_utils import restore_training
            env = GarageEnv(env)
            restore_training(exp_dir, exp_name, args,
                             env_saved=False, env=env)

        elif args.mode == 'eval':
            data = joblib.load(exp_dir + '/params.pkl')
            algo = data['algo']
            env.eval(algo.policy, n_episodes=args.n_eval_episodes, 
                greedy=args.eval_greedy, load_from_file=True, 
                save_replay=args.save_replay)
            env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Meta
    parser.add_argument('--mode', '-m', type=str, default='train')
    parser.add_argument('--loc', type=str, default='local')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--comment', type=str, default='')
    # Train
    parser.add_argument('--seed', '-s', type=int, default=1)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--bs', type=int, default=20000)
    parser.add_argument('--n_envs', type=int, default=1)
    # Eval
    parser.add_argument('--run_id', type=int, default=0) # sequential naming
    parser.add_argument('--n_eval_episodes', type=int, default=100)
    parser.add_argument('--render', type=int, default=1)
    parser.add_argument('--save_replay', type=int, default=0)
    parser.add_argument('--inspect_steps', type=int, default=0)
    parser.add_argument('--eval_during_training', type=int, default=1)
    parser.add_argument('--eval_greedy', type=int, default=1)
    parser.add_argument('--eval_epoch_freq', type=int, default=20)
    # Env
    parser.add_argument('--map', type=str, default='8m')
    parser.add_argument('--difficulty', type=str, default='7')
    # Algo
    # parser.add_argument('--max_algo_path_length', type=int, default=n_steps)
    parser.add_argument('--hidden_nonlinearity', type=str, default='tanh')
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--center_adv', type=int, default=1)
    parser.add_argument('--positive_adv', type=int, default=0)
    parser.add_argument('--gae_lambda', type=float, default=0.97)
    parser.add_argument('--ent', type=float, default=0.02) # 0.01 is too small
    parser.add_argument('--entropy_method', type=str, default='regularized')
    parser.add_argument('--clip_grad_norm', type=float, default=7)
    parser.add_argument('--opt_n_minibatches', type=int, default=4,
        help='The number of splits of a batch of trajectories for optimization.')
    parser.add_argument('--opt_mini_epochs', type=int, default=10,
        help='The number of epochs the optimizer runs for each batch of trajectories.')
    # Critic
    # Example: --encoder_hidden_sizes 12 123 1234 
    parser.add_argument('--dicg_encoder_hidden_sizes', nargs='+', type=int)
    parser.add_argument('--dicg_embedding_dim', type=int, default=64)
    parser.add_argument('--attention_type', type=str, default='general')
    parser.add_argument('--n_gcn_layers', type=int, default=2)
    parser.add_argument('--gcn_bias', type=int, default=1)
    parser.add_argument('--residual', type=int, default=1)
    # Policy
    # Example: --encoder_hidden_sizes 12 123 1234 
    parser.add_argument('--policy_encoder_hidden_sizes', nargs='+', type=int)
    parser.add_argument('--policy_embedding_dim', type=int, default=64)
    parser.add_argument('--policy_lstm_hidden_size', type=int, default=64)
    parser.add_argument('--state_include_actions', type=int, default=1)

    args = parser.parse_args()

    # single agent action_space = Discrete(14)
    # single agent observation_space = Box(80, )

    if args.policy_encoder_hidden_sizes is None:
        args.policy_encoder_hidden_sizes = [128, ] # Default hidden sizes

    if args.dicg_encoder_hidden_sizes is None:
        args.dicg_encoder_hidden_sizes = [128, ] # Default hidden sizes

    if args.map == '8m_vs_9m':
        args.ent = 0.025
        args.bs = 80000
    elif args.map == '3s_vs_5z':
        args.ent = 0.1
        args.bs = 60000
    elif args.map == '6h_vs_8z':
        args.ent = 0.025
        args.bs = 60000

    run(args)



