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
import matplotlib.pyplot as plt
from types import SimpleNamespace
import torch
from torch.nn import functional as F

import akro
import garage
from garage import wrap_experiment
from garage.envs import GarageEnv
from garage.experiment.deterministic import set_seed

from envs import TrafficJunctionWrapper
from dicg.torch.baselines import AttentionMLPCritic
from dicg.torch.algos import CentralizedMAPPO
from dicg.torch.policies import DecCategoricalMLPPolicy
from dicg.experiment.local_runner_wrapper import LocalRunnerWrapper
from dicg.sampler import CentralizedMAOnPolicyVectorizedSampler

def run(args):

    if args.exp_name is None:
        exp_layout = collections.OrderedDict([
            ('attn_mlp_ac_ppo', ''),
            ('atype={}', args.attention_type),
            ('entcoeff={}', args.ent),
            ('dim={}', args.dim),
            ('nagents={}', args.n_agents),
            ('difficulty={}', args.difficulty),
            ('curr={}', bool(args.curriculum)),
            ('steps={}', args.max_env_steps),
            ('nenvs={}', args.n_envs),
            ('bs={:0.0e}', args.bs),
            ('splits={}', args.opt_n_minibatches),
            ('miniepoch={}', args.opt_mini_epochs),
            ('seed={}', args.seed)
        ])

        exp_name = '_'.join(
            [key.format(val) for key, val in exp_layout.items()]
        )

    else:
        exp_name = args.exp_name


    prefix = 'traffic'
    id_suffix = ('_' + str(args.run_id)) if args.run_id != 0 else ''
    unseeded_exp_dir = './data/' + args.loc +'/' + exp_name[:-7]
    exp_dir = './data/' + args.loc +'/' + exp_name + id_suffix

    # Enforce
    args.center_adv = False if args.entropy_method == 'max' else args.center_adv

    if args.mode == 'train':
        # making sequential log dir if name already exists
        @wrap_experiment(name=exp_name,
                         prefix=prefix,
                         log_dir=exp_dir,
                         snapshot_mode='last', 
                         snapshot_gap=1)
        
        def train_traffic(ctxt=None, args_dict=vars(args)):
            args = SimpleNamespace(**args_dict)
            
            set_seed(args.seed)

            if args.curriculum:
                curr_start = int(0.125 * args.n_epochs)
                curr_end = int(0.625 * args.n_epochs)
            else:
                curr_start = 0
                curr_end = 0
                args.add_rate_min = args.add_rate_max
            
            env = TrafficJunctionWrapper(
                centralized=True,
                dim=args.dim,
                vision=1,
                add_rate_min=args.add_rate_min,
                add_rate_max=args.add_rate_max,
                curr_start=curr_start,
                curr_end=curr_end,
                difficulty=args.difficulty,
                n_agents=args.n_agents,
                max_steps=args.max_env_steps
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

            policy = DecCategoricalMLPPolicy(
                env.spec,
                env.n_agents,
                hidden_nonlinearity=hidden_nonlinearity,
                hidden_sizes=args.policy_hidden_sizes,
                name='dec_categorical_mlp_policy'
            )

            baseline = AttentionMLPCritic(
                env.spec,
                env.n_agents,
                encoder_hidden_sizes=args.encoder_hidden_sizes,
                embedding_dim=args.embedding_dim,
                attention_type=args.attention_type,
                name='attention_mlp_critic'
            )
            
            # Set max_path_length <= max_steps
            # If max_path_length > max_steps, algo will pad obs
            # obs.shape = torch.Size([n_paths, algo.max_path_length, feat_dim])
            algo = CentralizedMAPPO(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                max_path_length=args.max_env_steps, # Notice
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

        train_traffic(args_dict=vars(args))

    elif args.mode in ['restore', 'eval']:
        data = joblib.load(exp_dir + '/params.pkl')
        env = data['env']
        algo = data['algo']

        if args.mode == 'restore':
            from dicg.experiment.runner_utils import restore_training
            restore_training(exp_dir, exp_name, args,
                             env_saved=env.pickleable, env=env)

        elif args.mode == 'eval':
            env.eval(algo.policy, n_episodes=args.n_eval_episodes, greedy=args.eval_greedy, 
                load_from_file=True, max_steps=args.max_env_steps, render=args.render)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Meta
    parser.add_argument('--mode', '-m', type=str, default='train')
    parser.add_argument('--loc', type=str, default='local')
    parser.add_argument('--exp_name', type=str, default=None)
    # Train
    parser.add_argument('--seed', '-s', type=int, default=1)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--bs', type=int, default=80000)
    parser.add_argument('--n_envs', type=int, default=1)
    # Eval
    parser.add_argument('--run_id', type=int, default=0) # sequential naming
    parser.add_argument('--n_eval_episodes', type=int, default=100)
    parser.add_argument('--render', type=int, default=0)
    parser.add_argument('--inspect_steps', type=int, default=0)
    parser.add_argument('--eval_during_training', type=int, default=1)
    parser.add_argument('--eval_greedy', type=int, default=1)
    parser.add_argument('--eval_epoch_freq', type=int, default=5)
    # Env
    parser.add_argument('--max_env_steps', type=int, default=20)
    parser.add_argument('--dim', type=int, default=8)
    parser.add_argument('--n_agents', '-n', type=int, default=5)
    parser.add_argument('--difficulty', type=str, default='easy')
    parser.add_argument('--add_rate_max', type=float, default=0.3)
    parser.add_argument('--add_rate_min', type=float, default=0.1)
    parser.add_argument('--curriculum', type=int, default=0)
    # Algo
    # parser.add_argument('--max_algo_path_length', type=int, default=n_steps)
    parser.add_argument('--hidden_nonlinearity', type=str, default='tanh')
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--center_adv', type=int, default=1)
    parser.add_argument('--positive_adv', type=int, default=0)
    parser.add_argument('--gae_lambda', type=float, default=0.97)
    parser.add_argument('--ent', type=float, default=0.02)
    parser.add_argument('--entropy_method', type=str, default='regularized')
    parser.add_argument('--clip_grad_norm', type=float, default=7)
    parser.add_argument('--opt_n_minibatches', type=int, default=4,
        help='The number of splits of a batch of trajectories for optimization.')
    parser.add_argument('--opt_mini_epochs', type=int, default=10,
        help='The number of epochs the optimizer runs for each batch of trajectories.')
    # Critic
    parser.add_argument('--encoder_hidden_sizes', nargs='+', type=int)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--attention_type', type=str, default='general')
    # Policy
    # Example: --encoder_hidden_sizes 12 123 1234 
    parser.add_argument('--policy_hidden_sizes', nargs='+', type=int)

    args = parser.parse_args()

    # Enforce values
    if args.difficulty == 'hard':
        args.max_env_steps = 60
        args.dim = 18
        args.n_agents = 20
        args.add_rate_min = 0.02
        args.add_rate_max = 0.05
        
    elif args.difficulty == 'medium':
        args.max_env_steps = 40
        args.dim = 14
        args.n_agents = 10
        args.add_rate_min = 0.05
        args.add_rate_max = 0.2
        
    elif args.difficulty == 'easy':
        args.max_env_steps = 20
        args.dim = 8
        args.n_agents = 5
        args.add_rate_min = 0.1
        args.add_rate_max = 0.3
        args.embedding_dim = 64

    if args.policy_hidden_sizes is None:
        if args.difficulty == 'easy':
            args.policy_hidden_sizes = [128, 64, 64]
            args.encoder_hidden_sizes = [128, 128]
        else:
            args.policy_hidden_sizes = [265, 128, 64]
            args.encoder_hidden_sizes = [256, 128]

    run(args)



