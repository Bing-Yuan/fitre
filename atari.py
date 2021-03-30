import sys
from itertools import product

import gym
from stable_baselines3 import PPO

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

from ppo_fitre import FitrePPO, FitreMlpPolicy
from tr_kfac_opt import KFACOptimizer


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


env = make_atari_env(atari_env_name('pong', 'image', 'v4', no_frame_skip=True), n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4)

args = sys.argv
if len(args) <= 1 or args[1] == 'fitre':
    print("RUNNING FITRE")
    # model = FitrePPO(FitreMlpPolicy, env, verbose=1)
    model = FitrePPO('CnnPolicy', env, verbose=1)
    model.policy.optimizer = KFACOptimizer(model.policy)
    model.learn(total_timesteps=1000000)
else:
    print("RUNNING PPO")
    # model = PPO('MlpPolicy', env, verbose=1)
    model = PPO('CnnPolicy', env, verbose=1)
    model.learn(total_timesteps=1000000)
