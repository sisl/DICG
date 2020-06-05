from garage.experiment import LocalRunner
from garage.sampler.base import BaseSampler
from dowel import logger

class LocalRunnerWrapper(LocalRunner):

    def __init__(self, 
                 snapshot_config, 
                 eval=False,
                 n_eval_episodes=100,
                 eval_greedy=True,
                 eval_epoch_freq=10,
                 save_env=True):
        super().__init__(snapshot_config)
        
        self.save_env = save_env

        self.eval = eval
        self.n_eval_episodes = n_eval_episodes
        self.eval_greedy = eval_greedy
        self.eval_epoch_freq = eval_epoch_freq

    def obtain_samples(self, itr, batch_size=None):
        """Obtain one batch of samples.

        Args:
            itr (int): Index of iteration (epoch).
            batch_size (int): Number of steps in batch.
                This is a hint that the sampler may or may not respect.

        Returns:
            list[dict]: One batch of samples.

        """
        paths = None
        if isinstance(self._sampler, BaseSampler):
            paths = self._sampler.obtain_samples(
                itr, (batch_size or self._train_args.batch_size))
        else:
            paths = self._sampler.obtain_samples(
                itr, (batch_size or self._train_args.batch_size),
                agent_update=self._algo.policy.get_param_values())
            paths = paths.to_trajectory_list()

        if hasattr(self._policy, 'centralized'): 
            if self._policy.centralized:
                self._stats.total_env_steps += sum([len(p['rewards']) for p in paths])
            else:
                self._stats.total_env_steps += \
                    int(sum([len(p['rewards']) for p in paths]) / self._env.n_agents)
        else:
            self._stats.total_env_steps += \
                    int(sum([len(p['rewards']) for p in paths]) / self._env.n_agents)

        return paths

    def save(self, epoch):
        """Save snapshot of current batch.

        Args:
            epoch (int): Epoch.

        Raises:
            NotSetupError: if save() is called before the runner is set up.

        """
        if not self._has_setup:
            raise NotSetupError('Use setup() to setup runner before saving.')

        logger.log('Saving snapshot...')

        params = dict()
        # Save arguments
        params['setup_args'] = self._setup_args
        params['train_args'] = self._train_args
        params['stats'] = self._stats

        # Save states
        if self.save_env:
            params['env'] = self._env
        params['algo'] = self._algo

        self._snapshotter.save_itr_params(epoch, params)

        logger.log('Saved')