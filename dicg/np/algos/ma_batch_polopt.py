"""A batch-based algorithm interleaves sampling and policy optimization."""
import abc
import collections

from dowel import tabular
import numpy as np

from garage import log_performance, TrajectoryBatch
from garage.misc import tensor_utils
from garage.np.algos.base import RLAlgorithm
from garage.tf.samplers import BatchSampler
from dicg.sampler import CentralizedMAOnPolicyVectorizedSampler

class MABatchPolopt(RLAlgorithm):
    """A batch-based algorithm interleaves sampling and policy optimization.

    In one round of training, the runner will first instruct the sampler to do
    environment rollout and the sampler will collect a given number of samples
    (in terms of environment interactions). The collected paths are then
    absorbed by `RLAlgorithm.train_once()` and an algorithm performs one step
    of policy optimization. The updated policy will then be used in the
    next round of sampling.

    Args:
        env_spec (garage.envs.EnvSpec): Environment specification.
        policy (garage.tf.policies.base.Policy): Policy.
        baseline (garage.tf.baselines.Baseline): The baseline.
        discount (float): Discount.
        max_path_length (int): Maximum length of a single rollout.
        n_samples (int): Number of train_once calls per epoch.

    """

    def __init__(self, env_spec, policy, baseline, discount, max_path_length,
                 n_samples):
        self.env_spec = env_spec
        self.policy = policy
        self.baseline = baseline
        self.discount = discount
        self.max_path_length = max_path_length
        self.n_samples = n_samples # num train_once per epoch

        self.episode_reward_mean = collections.deque(maxlen=100)
        
        self.sampler_cls = CentralizedMAOnPolicyVectorizedSampler


    @abc.abstractmethod
    def train_once(self, itr, paths):
        """Perform one step of policy optimization given one batch of samples.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        """

    def train(self, runner):
        """Obtain samplers and start actual training for each epoch.

        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        last_return = None

        for epoch in runner.step_epochs():
            for _ in range(self.n_samples):
                runner.step_path = runner.obtain_samples(runner.step_itr)
                tabular.record('TotalEnvSteps', runner.total_env_steps)
                last_return = self.train_once(runner.step_itr,
                                              runner.step_path)
                runner.step_itr += 1

                if hasattr(runner, 'eval'):
                    if runner.eval and epoch % runner.eval_epoch_freq == 0:
                            env_id = np.random.randint(runner._sampler._n_envs)
                            runner._sampler._vec_env.envs[env_id].eval(
                                self.policy,
                                n_episodes=runner.n_eval_episodes,
                                greedy=runner.eval_greedy)
        return last_return

        