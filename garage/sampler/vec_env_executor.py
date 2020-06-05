"""Environment wrapper that runs multiple environments."""

import numpy as np

from garage.misc import tensor_utils


class VecEnvExecutor:
    """Environment wrapper that runs multiple environments."""

    def __init__(self, envs, max_path_length):
        self.envs = envs
        self._action_space = envs[0].action_space
        self._observation_space = envs[0].observation_space
        self.ts = np.zeros(len(self.envs), dtype='int')
        self.max_path_length = max_path_length

    def step(self, action_n, itr=None):
        """Step all environments using the provided actions."""
        all_results = [env.step(a) for (a, env) in zip(action_n, self.envs)]
        # for e in self.envs:
        #     e.render()
        obs, rewards, dones, env_infos = list(
            map(list, list(zip(*all_results))))
        dones = np.asarray(dones)
        rewards = np.asarray(rewards)
        self.ts += 1
        if self.max_path_length is not None:
            dones[self.ts >= self.max_path_length] = True
        for (i, done) in enumerate(dones):
            if done:
                if itr is not None:
                    obs[i] = self.envs[i].reset(epoch=itr)
                else:
                    obs[i] = self.envs[i].reset()
                # obs[i] = self.envs[i].reset()
                self.ts[i] = 0
        return obs, rewards, dones, tensor_utils.stack_tensor_dict_list(
            env_infos)

    def reset(self, itr=None):
        """Reset all environments."""
        if itr is None:
            results = [env.reset() for env in self.envs]
        else:
            results = [env.reset(epoch=itr) for env in self.envs]
        self.ts[:] = 0
        return results

    @property
    def num_envs(self):
        """Read / write the number of environments."""
        return len(self.envs)

    @property
    def action_space(self):
        """Read / write the action space."""
        return self._action_space

    @property
    def observation_space(self):
        """Read / write the observation space."""
        return self._observation_space

    def close(self):
        """Close all environments."""
        pass
