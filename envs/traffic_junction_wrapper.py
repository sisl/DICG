# Using local gym
import sys
import os
current_file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_path + '/../../')


from envs.traffic_junction.traffic_junction import TrafficJunctionEnv
import gym
from gym.utils import seeding
import numpy as np

from envs.utils import standard_eval
import time
import copy
from types import SimpleNamespace
import torch

def flatdim(space):
    """Return the number of dimensions a flattened equivalent of this space
    would have.
    Accepts a space and returns an integer. Raises ``NotImplementedError`` if
    the space is not defined in ``gym.spaces``.
    """
    if isinstance(space, gym.spaces.Box):
        return int(np.prod(space.shape))
    elif isinstance(space, gym.spaces.Discrete):
        return int(space.n)
    elif isinstance(space, gym.spaces.Tuple):
        return int(sum([flatdim(s) for s in space.spaces]))
    elif isinstance(space, gym.spaces.Dict):
        return int(sum([flatdim(s) for s in space.spaces.values()]))
    elif isinstance(space, gym.spaces.MultiBinary):
        return int(np.prod(space.n)) # customized
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return int(np.prod(space.shape))
    else:
        raise NotImplementedError

def flatten(space, x):
    """Flatten a data point from a space.
    This is useful when e.g. points from spaces must be passed to a neural
    network, which only understands flat arrays of floats.
    Accepts a space and a point from that space. Always returns a 1D array.
    Raises ``NotImplementedError`` if the space is not defined in
    ``gym.spaces``.
    """
    if isinstance(space, gym.spaces.Box):
        return np.asarray(x, dtype=np.float32).flatten()
    elif isinstance(space, gym.spaces.Discrete):
        onehot = np.zeros(space.n, dtype=np.float32)
        onehot[np.array(x, dtype=int)] = 1.0
        return onehot
    elif isinstance(space, gym.spaces.Tuple):
        return np.concatenate(
            [flatten(s, x_part) for x_part, s in zip(x, space.spaces)])
    elif isinstance(space, gym.spaces.Dict):
        return np.concatenate(
            [flatten(s, x[key]) for key, s in space.spaces.items()])
    elif isinstance(space, gym.spaces.MultiBinary):
        return np.asarray(x).flatten()
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return np.asarray(x).flatten()
    else:
        raise NotImplementedError

class TrafficJunctionWrapper(TrafficJunctionEnv):

    def __init__(self, 
                 centralized,
                 dim=18,
                 vision=1,
                 add_rate_min=0.02,
                 add_rate_max=0.05,
                 curr_start=0,
                 curr_end=0,
                 difficulty='hard',
                 vocab_type='bool',
                 n_agents=20,
                 max_steps=100):

        super().__init__()

        env_args = SimpleNamespace(**{
            'dim': dim,
            'vision': vision,
            'add_rate_min': add_rate_min,
            'add_rate_max': add_rate_max,
            'curr_start': curr_start,
            'curr_end': curr_end,
            'difficulty': difficulty,
            'vocab_type': vocab_type,
            'nagents': n_agents,
        })

        self.multi_agent_init(env_args)
        self.n_agents = self.ncar
        self.max_steps = max_steps

        self.curriculum_learning = True

        self.centralized = centralized
        self.original_obs_space = copy.deepcopy(self.observation_space)
        single_agent_obs_flatdim = flatdim(self.observation_space)

        if centralized:
            self.observation_space = gym.spaces.Box(
                low=np.array(single_agent_obs_flatdim * [-1] * self.n_agents, dtype=np.float32),
                high=np.array(single_agent_obs_flatdim * [1] * self.n_agents, dtype=np.float32)
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=np.array(single_agent_obs_flatdim * [-1], dtype=np.float32),
                high=np.array(single_agent_obs_flatdim * [1], dtype=np.float32)
            )
        self.pickleable = True

    def get_avail_actions(self):
        avail_actions = [[1] * self.action_space.n for _ in range(self.n_agents)]
        if not self.centralized:
            return avail_actions
        else:
            return np.concatenate(avail_actions)

    def step(self, actions):
        obses_raw, rewards, dones, infos = super().step(actions)
        obses = []
        for i in range(self.n_agents):
            obses.append(list(flatten(self.original_obs_space, obses_raw[i])))
        if not self.centralized:
            return obses, rewards, [bool(dones)] * self.n_agents, infos
        else:
            return np.concatenate(obses), np.mean(rewards), np.all(dones), infos

    def reset(self, epoch=None):
        obses_raw = super().reset(epoch)
        obses = []
        for i in range(self.n_agents):
            obses.append(list(flatten(self.original_obs_space, obses_raw[i])))
        if not self.centralized:
            return obses
        else:
            return np.concatenate(obses)

    def eval(self, policy, n_episodes=20, greedy=True, load_from_file=False, 
             max_steps=60):
        import dowel
        from dowel import logger, tabular
        from garage.misc.prog_bar_counter import ProgBarCounter

        if load_from_file:
            logger.add_output(dowel.StdOutput())
        logger.log('Evaluating policy, {} episodes, greedy = {} ...'.format(
            n_episodes, greedy))
        episode_rewards = []
        success = 0
        pbar = ProgBarCounter(n_episodes)
        for e in range(n_episodes):
            obs = self.reset()
            policy.reset([True])
            terminated = False
            t = 0
            episode_rewards.append(0)
            while not terminated:
                if not self.centralized:
                    # obs.shape = (n_agents, n_envs, obs_dim)
                    obs = torch.Tensor(obs).unsqueeze(1) # add n_envs dim
                    avail_actions = torch.Tensor(self.get_avail_actions()).unsqueeze(1)
                    actions, agent_infos = policy.get_actions(obs, 
                        avail_actions, greedy=greedy)
                    if len(actions.shape) == 3: # n-d action
                        actions = actions[:, 0, :]
                    elif len(actions.shape) == 2: # 1-d action
                        actions = actions[:, 0]
                    obs, reward, terminated, info = self.step(actions) # n_env = 1
                    terminated = all(terminated) 
                else:
                    # obs.shape = (n_envs, n_agents * obs_dim)
                    obs = np.array([obs])
                    avail_actions = np.array([self.get_avail_actions()])
                    actions, agent_infos = policy.get_actions(obs, 
                        avail_actions, greedy=greedy)
                    obs, reward, terminated, info = self.step(actions[0]) # n_env = 1
                t += 1
                if t >= max_steps:
                    terminated = True
                episode_rewards[-1] += np.mean(reward)
            # episode end
            success += self.stat['success']
            pbar.inc(1)
        pbar.stop()
        policy.reset([True])
        avg_return = np.mean(episode_rewards)
        success = success / n_episodes
        logger.log('EvalAvgReturn: {}'.format(avg_return))
        logger.log('EvalSucessRate: {}'.format(success))
        if not load_from_file:
            tabular.record('EvalAvgReturn', avg_return)
            tabular.record('EvalSucessRate', success)

        # return eval metric
        return success

    def seed(self, n):
        self.np_random, seed1 = seeding.np_random(n)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]


if __name__ == '__main__':
    env = TrafficJunctionWrapper(centralized=True, vocab_type='bool')
    print('Env test, centralized = {}'.format(env.centralized))
    print('n_agents:', env.n_agents)
    print('single agent action_space:', env.action_space)
    print('all agent observation_space:', env.observation_space)
    obs = env.reset()
    # print('all agent obs:', obs)
    print('flat full obs len:', len(obs))

    for t in range(env.max_steps):
        actions = [env.action_space.sample() for _ in range(env.n_agents)]
        obs, rew, done, info = env.step(actions)
        print(t)
        print(env.car_loc[0])
        print(env.alive_mask[0])
        print(np.reshape(obs, [env.n_agents, -1])[0, :])