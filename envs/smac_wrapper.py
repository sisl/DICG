# Using local gym
import sys
import os
current_file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_path + '/../../')

import dowel
from dowel import logger, tabular
from garage.misc.prog_bar_counter import ProgBarCounter
from smac.env import StarCraft2Env
import gym
from gym.utils import seeding
from envs.ma_gym.envs.utils.observation_space import MultiAgentObservationSpace
import numpy as np

class SMACWrapper(StarCraft2Env):

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, centralized, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_range = (-np.inf, np.inf)
        self.viewer = None
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.agent_obs_dim = np.sum([np.prod(self.get_obs_move_feats_size()),
                                     np.prod(self.get_obs_enemy_feats_size()),
                                     np.prod(self.get_obs_ally_feats_size()),
                                     np.prod(self.get_obs_own_feats_size())])
        self.observation_space_low = [0] * self.agent_obs_dim
        self.observation_space_high = [1] * self.agent_obs_dim
        self.observation_space = gym.spaces.Box(
                low=np.array(self.observation_space_low),
                high=np.array(self.observation_space_high))
        self.centralized = centralized
        if centralized:
            self.observation_space = gym.spaces.Box(
                low=np.array(self.observation_space_low * self.n_agents),
                high=np.array(self.observation_space_high * self.n_agents)
            )
        self.pickleable = False

    def get_avail_actions(self):
        avail_actions = super().get_avail_actions()
        if not self.centralized:
            return avail_actions
        else:
            return np.concatenate(avail_actions)

    def reset(self):
        obses = super().reset()[0]
        if not self.centralized:
            return obses
        else:
            return np.concatenate(obses)

    def step(self, actions):
        reward, terminated, info = super().step(actions)
        if not self.centralized:
            return super().get_obs(), [reward] * self.n_agents, \
                    [terminated] * self.n_agents, info 
        else:
            return np.concatenate(super().get_obs()), reward, terminated, info

    def seed(self, n):
        self.np_random, seed1 = seeding.np_random(n)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        super().close()

    def eval(self, policy, n_episodes=20, greedy=True, load_from_file=False,
             save_replay=False):
        if load_from_file:
            logger.add_output(dowel.StdOutput())
        logger.log('Evaluating policy, {} episodes, greedy = {} ...'.format(
            n_episodes, greedy))

        n_won = 0
        episode_rewards = []
        pbar = ProgBarCounter(n_episodes)
        for e in range(n_episodes):
            obs = self.reset()
            policy.reset([True])
            info = {'battle_won': False}
            terminated = False
            episode_rewards.append(0)

            while not terminated:
                obs = np.array([obs]) # add [.] for vec_env
                avail_actions = np.array([self.get_avail_actions()])
                actions, agent_infos = policy.get_actions(obs, 
                    avail_actions, greedy=greedy)
                obs, reward, terminated, info = self.step(actions[0])
                if not self.centralized:
                    terminated = all(terminated)
                episode_rewards[-1] += np.mean(reward)
            pbar.inc(1)
            if save_replay:
                self.save_replay()

            # If case SC2 restarts during eval, KeyError: 'battle_won' can happen
            # Take precaution
            if type(info) == dict: 
                if 'battle_won' in info.keys():
                    n_won += 1 if info['battle_won'] else 0

        pbar.stop()
        policy.reset([True])
        win_rate = n_won / n_episodes
        avg_return = np.mean(episode_rewards)

        logger.log('EvalWinRate: {}'.format(win_rate))
        logger.log('EvalAvgReturn: {}'.format(avg_return))
        if not load_from_file:
            tabular.record('EvalWinRate', win_rate)
            tabular.record('EvalAvgReturn', avg_return)



if __name__ == '__main__':
    env = SMACWrapper(centralized=True)
    env.reset()
    print(env.n_agents)
    print(env.agents)
    for a_id in range(env.n_agents):
        print(env.get_avail_agent_actions(a_id))
    print(env.action_space)
    print(env.observation_space)

    env = SMACWrapper(centralized=False)
    print(env.n_agents)
    print(env.action_space)
    print(env.observation_space)