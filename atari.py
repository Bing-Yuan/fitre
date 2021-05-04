import sys
from itertools import product
from typing import Dict

from tr_kfac_opt import KFACOptimizer
# %load_ext autoreload
# %autoreload 2

import ray
ray.init(num_cpus=12, num_gpus=1)

import gym
import numpy as np

import stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env, make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

from ppo_fitre import FitrePPO, FitreMlpPolicy, FitreCnnPolicy

import matplotlib.pyplot as plt

from ray import tune
from ray.tune import Analysis


# Source copied from <https://github.com/openai/gym/blob/master/gym/envs/__init__.py>.
ALL_ATARI_GAMES = [
    'adventure', 'air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis',
    'bank_heist', 'battle_zone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout', 'carnival',
    'centipede', 'chopper_command', 'crazy_climber',
    # 'defender',  # somehow defender env takes a long long time to create, and never finish? The rest are fine.
    'demon_attack', 'double_dunk',
    'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar',
    'hero', 'ice_hockey', 'jamesbond', 'journey_escape', 'kangaroo', 'krull', 'kung_fu_master',
    'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', 'pooyan',
    'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing',
    'solaris', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down',
    'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon'
]


def atari_env_name(game: str, obs_type: str, version: str,
                   deterministic: bool = False, no_frame_skip: bool = False) -> str:
    """ e.g., space_invaders should yield SpaceInvaders-v0 or SpaceInvaders-ram-v0
    :param obs_type: input as 'image' or 'ram'
    :param version: 'v0' or 'v4', v0 has 25% chance repeat previous action,
                    while v4 always follows policy action
    :param deterministic: fixed frameskip (usually 4), otherwise sampeld from (2, 5)
    :param no_frame_skip: cannot be both true with @param deterministic
    """
    assert obs_type in ['image', 'ram']
    assert version in ['v0', 'v4']
    assert not (deterministic and no_frame_skip)

    name = ''.join([g.capitalize() for g in game.split('_')])
    if obs_type == 'ram':
        name = f'{name}-ram'

    if deterministic:
        name = f'{name}Deterministic'
    elif no_frame_skip:
        name = f'{name}NoFrameskip'

    name = f'{name}-{version}'
    return name


def test_print_all_envs():
    for game, obs_type, version, deterministic, no_frame_skip in\
            product(ALL_ATARI_GAMES, ['image', 'ram'], ['v0', 'v4'], [True, False], [True, False]):
        if deterministic and no_frame_skip:
            continue
        name = atari_env_name(game, obs_type, version, deterministic, no_frame_skip)
        env = gym.make(name)
        print(env)
    return

# test_print_all_envs()
# exit(0)


def common_hypers() -> Dict:
    return {
        'n_steps': tune.choice([256, 512, 1024, 2048, 4096]),
        'batch_size': tune.choice([32, 64, 128, 256, 512, 1024, 2048]),
        'n_epochs': tune.randint(4, 32),
        'gamma': tune.uniform(0.95, 1),
        'gae_lambda': tune.uniform(0.95, 0.999),
        'clip_range': tune.uniform(0.01, 0.5),
        'clip_range_vf': None,
        'ent_coef': 0.05,
        'vf_coef': tune.uniform(0.2, 0.8),
        'max_grad_norm': 0.5,
        'use_sde': False,
        'sde_sample_freq': -1,
        'target_kl': None,
        'seed': 1,
    }


def ppo_hypers():
    params = common_hypers()
    params.update({
        'learning_rate': tune.loguniform(1e-6, 0.01),
    })
    return params


def fitre_hypers():
    params = common_hypers()
    params['fitre_params'] = {
        'scheme': 'tr',
        'momentum': tune.uniform(0.8, 1),
        'check_grad': 0,
        'stat_decay': tune.uniform(0.9, 1),
        'kl_clip': 0.001,
        'damping': tune.loguniform(1e-6, 1e0),
        'weight_decay': 0,
        'fast_cnn': False,
        'Ts': 1,
        'Tf': tune.choice([1, 2, 5, 10]),
        'max_delta': 10,
        'min_delta': 1e-6,
        'split_bs': False
    }
    return params


class ReportCallback(sb3.common.callbacks.BaseCallback):
    def __init__(self, verbose=0):
        super(ReportCallback, self).__init__(verbose)

    def _on_rollout_end(self) -> None:
        ep_rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
        ep_rew_mean = np.mean(ep_rewards)
        tune.report(ep_rew=ep_rew_mean)
        return

    def _on_step(self) -> bool:
        return True


def get_env():
    env = make_atari_env(atari_env_name('pong', 'image', 'v4', no_frame_skip=True), n_envs=4, seed=0)
    env = VecFrameStack(env, n_stack=4)
    return env


def train_fitre(config):
    print("RUNNING FITRE")
    env = get_env()
    callback = ReportCallback()

    fitre_params = config['fitre_params']
    del config['fitre_params']

    # model = FitrePPO(FitreMlpPolicy, env, verbose=1)
    model = FitrePPO(FitreCnnPolicy, env, verbose=1, **config)
    model.policy.optimizer = KFACOptimizer(model.policy, **fitre_params)
    model.learn(total_timesteps=1000000, callback=callback)
    model.save("fitre_pong")
    return


def train_ppo(config):
    print("RUNNING PPO")
    env = get_env()
    callback = ReportCallback()
    # model = PPO('MlpPolicy', env, verbose=1)
    model = PPO('CnnPolicy', env, verbose=1, **config)
    model.learn(total_timesteps=1000000, callback=callback)
    model.save("ppo_pong")
    return


### Uncomment below to quickly test-run without tuning.
# train = train_fitre
# config = dict(  # randomly picked ones for testing, otherwise ray.tune values won't concretize
#     learning_rate=1e-4,
#     n_steps=256,
#     batch_size=32,
#     n_epochs=4,
#     gamma=0.99,
#     gae_lambda=0.99,
#     clip_range=0.5,
#     clip_range_vf=None,
#     ent_coef=0.05,
#     vf_coef=0.5,
#     max_grad_norm=0.5,
#     use_sde=False,
#     sde_sample_freq=-1,
#     target_kl=None,
#     seed=1,
#     # policy_kwargs = dict(net_arch=[64, {'pi': [64], 'vf': [64]}])
#
#     fitre_params = {
#         'scheme': 'tr',
#         'momentum': 0.9,
#         'check_grad': 0,
#         'stat_decay': 0.95,
#         'kl_clip': 0.001,
#         'damping': 1e-3,
#         'weight_decay': 0,
#         'fast_cnn': False,
#         'Ts': 1,
#         'Tf': 10,
#         'max_delta': 10,
#         'min_delta': 1e-6,
#         'split_bs': False
#     }
# )
#
# train(config)
# exit(0)


args = sys.argv
if len(args) <= 1 or args[1] == 'fitre':
    run_fitre = True
    train = train_fitre
    config = fitre_hypers()
else:
    run_fitre = False
    train = train_ppo
    config = ppo_hypers()

analysis = tune.run(train, config=config, num_samples=50, verbose=1, metric="ep_rew", mode="max",
                    # resources_per_trial={'gpu': 1},
                    raise_on_failed_trial=False)

# analysis = tune.ExperimentAnalysis("~/ray_results/F16_train_2020-12-14_00-36-42/experiment_state-2020-12-14_00-36-42.json")
# analysis = tune.Analysis("~/ray_results/F16_train_2020-12-14_00-36-42")
print(f'loaded analysis: {analysis}')

plt.figure()

ax = None
dfs = analysis.trial_dataframes
for d in dfs.values():
    ax = d.plot(kind='line', x="training_iteration", y="ep_rew", ax=ax, legend=False)

y_axis = plt.axes().yaxis
y_axis.set_label_text('ep_rew')
# plt.show()
title = 'fitre_pong_out.png' if run_fitre else 'ppo_pong_out.png'
plt.savefig(title)

# analysis.get_best_config(metric="ep_rew", mode="max")
# print(analysis)
