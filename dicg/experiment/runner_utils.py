import sys
import os

current_file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_path + '/../../')

from garage.experiment.experiment import ExperimentContext
from garage.experiment.deterministic import set_seed
from dicg.experiment.local_runner_wrapper import LocalRunnerWrapper
from tensorboardX import SummaryWriter
import dowel
from dowel import logger
import time
import socket

def restore_training(log_dir, exp_name, args, env_saved=True, env=None):
    tabular_log_file = os.path.join(log_dir, 'progress_restored.{}.{}.csv'.
        format(str(time.time())[:10], socket.gethostname()))
    text_log_file = os.path.join(log_dir, 'debug_restored.{}.{}.log'.
        format(str(time.time())[:10], socket.gethostname()))
    logger.add_output(dowel.TextOutput(text_log_file))
    logger.add_output(dowel.CsvOutput(tabular_log_file))
    logger.add_output(dowel.TensorBoardOutput(log_dir))
    logger.add_output(dowel.StdOutput())
    logger.push_prefix('[%s] ' % exp_name)

    ctxt = ExperimentContext(snapshot_dir=log_dir,
                             snapshot_mode='last',
                             snapshot_gap=1)

    
    runner = LocalRunnerWrapper(
        ctxt,
        eval=args.eval_during_training,
        n_eval_episodes=args.n_eval_episodes,
        eval_greedy=args.eval_greedy,
        eval_epoch_freq=args.eval_epoch_freq,
        save_env=env_saved
    )
    saved = runner._snapshotter.load(log_dir, 'last')
    runner._setup_args = saved['setup_args']
    runner._train_args = saved['train_args']
    runner._stats = saved['stats']

    set_seed(runner._setup_args.seed)
    algo = saved['algo']

    # Compatibility patch
    if not hasattr(algo, '_clip_grad_norm'):
        setattr(algo, '_clip_grad_norm', args.clip_grad_norm)

    if env_saved:
        env = saved['env']

    runner.setup(env=env,
                 algo=algo,
                 sampler_cls=runner._setup_args.sampler_cls,
                 sampler_args=runner._setup_args.sampler_args)
    runner._train_args.start_epoch = runner._stats.total_epoch + 1
    runner._train_args.n_epochs = runner._train_args.start_epoch + args.n_epochs

    print('\nRestored checkpoint from epoch #{}...'.format(runner._train_args.start_epoch))
    print('To be trained for additional {} epochs...'.format(args.n_epochs))
    print('Will be finished at epoch #{}...\n'.format(runner._train_args.n_epochs))

    return runner._algo.train(runner)