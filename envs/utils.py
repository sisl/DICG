import time
import dowel
from dowel import logger, tabular
from garage.misc.prog_bar_counter import ProgBarCounter
import numpy as np
import torch

def standard_eval(env, policy, n_episodes=20, greedy=True, load_from_file=False, 
                  render=False, recorder=None, max_steps=10000):
    if recorder is not None:
        render = False # force off
    if load_from_file:
        logger.add_output(dowel.StdOutput())
    logger.log('Evaluating policy, {} episodes, greedy = {} ...'.format(
        n_episodes, greedy))
    episode_rewards = []
    pbar = ProgBarCounter(n_episodes)
    for e in range(n_episodes):
        obs = env.reset()
        policy.reset([True])
        terminated = False
        t = 0
        episode_rewards.append(0)
        while not terminated:
            if render:
                env.render()
                # time.sleep(0.05)
            if recorder is not None:
                recorder.capture_frame()
            if not env.centralized:
                # obs.shape = (n_agents, n_envs, obs_dim)
                obs = torch.Tensor(obs).unsqueeze(1) # add n_envs dim
                avail_actions = torch.Tensor(env.get_avail_actions()).unsqueeze(1)
                actions, agent_infos = policy.get_actions(obs, 
                    avail_actions, greedy=greedy)
                if len(actions.shape) == 3: # n-d action
                    actions = actions[:, 0, :]
                elif len(actions.shape) == 2: # 1-d action
                    actions = actions[:, 0]
                obs, reward, terminated, info = env.step(actions) # n_env = 1
                terminated = all(terminated) 
            else:
                # obs.shape = (n_envs, n_agents * obs_dim)
                obs = np.array([obs])
                avail_actions = np.array([env.get_avail_actions()])
                actions, agent_infos = policy.get_actions(obs, 
                    avail_actions, greedy=greedy)
                obs, reward, terminated, info = env.step(actions[0]) # n_env = 1
            t += 1
            if t > max_steps:
                terminated = True
            episode_rewards[-1] += np.mean(reward)
        pbar.inc(1)
    pbar.stop()
    policy.reset([True])
    avg_return = np.mean(episode_rewards)
    logger.log('EvalAvgReturn: {}'.format(avg_return))
    if not load_from_file:
        tabular.record('EvalAvgReturn', avg_return)