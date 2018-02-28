import numpy as np

import argparse

from sac.rllab.envs.normalized_env import normalize
from sac.algos import SACV2
from sac.envs import MultiGoalEnv
from sac.misc.plotter import QFPolicyPlotter
from sac.misc.utils import timestamp
from sac.policies import RealNVPPolicy
from sac.replay_buffers import SimpleReplayBuffer
from sac.value_functions import NNQFunction, NNVFunction
from sac.misc.utils import initialize_logger
import sac.config as config

LOGGER_PARAMS = {
    'exp_name': str(timestamp()),
    'mode': 'local',
    'log_dir': config.LOCAL_LOG_DIR,
    'snapshot_mode': 'gap',
    'snapshot_gap': 100,
}


def parse_args():
    parser = argparse.ArgumentParser()

    for key, value in LOGGER_PARAMS.items():
        parser.add_argument(
            '--{key}'.format(key=key), type=type(value), default=value)

    args = parser.parse_args()

    return args


def run():
    env = normalize(
        MultiGoalEnv(
            actuation_cost_coeff=1,
            distance_cost_coeff=0.1,
            goal_reward=1,
            init_sigma=0.1))

    pool = SimpleReplayBuffer(max_replay_buffer_size=1e6, env_spec=env.spec)

    base_kwargs = dict(
        min_pool_size=30,
        epoch_length=1000,
        n_epochs=1000,
        max_path_length=30,
        batch_size=64,
        n_train_repeat=2,
        eval_render=True,
        eval_n_episodes=10,
        eval_deterministic=False)

    M = 128
    qf = NNQFunction(env_spec=env.spec, hidden_layer_sizes=[M, M])

    vf = NNVFunction(env_spec=env.spec, hidden_layer_sizes=[M, M])

    real_nvp_config = {
        "scale_regularization": 0.0,
        "num_coupling_layers": 2,
        "translation_hidden_sizes": (M, ),
        "scale_hidden_sizes": (M, ),
    }

    policy = RealNVPPolicy(
        env_spec=env.spec,
        mode="train",
        squash=True,
        real_nvp_config=real_nvp_config,
        observations_preprocessor=None)

    plotter = QFPolicyPlotter(
        qf=qf,
        policy=policy,
        obs_lst=np.array([[-2.5, 0.0], [0.0, 0.0], [2.5, 2.5]]),
        default_action=[np.nan, np.nan],
        n_samples=100)

    algorithm = SACV2(
        base_kwargs=base_kwargs,
        env=env,
        policy=policy,
        pool=pool,
        qf=qf,
        vf=vf,
        plotter=plotter,
        lr=3e-4,
        scale_reward=3,
        discount=0.99,
        tau=1e-4,
        save_full_state=True)

    algorithm.train()


def main():
    args = parse_args()
    initialize_logger(**args.__dict__)
    run()


if __name__ == "__main__":
    main()
