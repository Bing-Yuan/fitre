""" Experiment on CartPole to incorporate FITRE, comparing with vanilla policy gradient at present.
    Code modifed from https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/1_simple_pg.py
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tr_kfac_opt import KFACOptimizer


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2,
          epochs=50, batch_size=5000, render=False, use_fitre: bool = False):
    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # make core of policy network
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

    # make function to compute action distribution
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        return get_policy(obs).sample().item()

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    # make optimizer
    if use_fitre:
        optimizer = KFACOptimizer(logits_net)
    else:
        optimizer = Adam(logits_net.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            tmp_obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(dim=0)  # tutorial doesn't have unsqueeze()
            act = get_action(tmp_obs)
            obs, rew, done, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        if use_fitre:
            ''' fitre needs two passes, here we use the categorical distribution as 1st pass's objective,
                just like a classification problem.
            '''
            optimizer.zero_grad()
            optimizer.acc_stats = True
            tmp_obs = torch.as_tensor(batch_obs, dtype=torch.float32)
            batch_pol = get_policy(tmp_obs)
            obj = ce(batch_pol.logits, batch_pol.logits.argmax(dim=1))
            obj.backward(retain_graph=True)

            # take a single policy gradient update step
            optimizer.zero_grad()
            tmp_act = torch.as_tensor(batch_acts, dtype=torch.int32)
            tmp_ws = torch.as_tensor(batch_weights, dtype=torch.float32)
            batch_loss = compute_loss(obs=tmp_obs, act=tmp_act, weights=tmp_ws)
            optimizer.acc_stats = False
            batch_loss.backward(create_graph=True)

            def _batch_loss():
                with torch.no_grad():
                    return compute_loss(obs=tmp_obs, act=tmp_act, weights=tmp_ws)

            optimizer.step(closure=_batch_loss)
        else:
            optimizer.zero_grad()
            batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                      act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                      weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                      )
            batch_loss.backward()
            optimizer.step()
        return batch_loss, batch_rets, batch_lens

    # training loop
    method = 'fitre' if use_fitre else 'vpg'
    fpath = Path(f'cartpole-results-{method}.pth')
    history_rets, history_lens = None, None
    if fpath.is_file():
        history_rets, history_lens = torch.load(fpath)

    curr_rets, curr_lens = [], []
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        mean_rets = np.mean(batch_rets)
        mean_lens = np.mean(batch_lens)
        curr_rets.append(mean_rets)
        curr_lens.append(mean_lens)
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, mean_rets, mean_lens))

    curr_rets = torch.tensor(curr_rets).float().unsqueeze(dim=0)
    curr_lens = torch.tensor(curr_lens).float().unsqueeze(dim=0)
    if history_rets is not None:
        curr_rets = torch.cat((history_rets, curr_rets), dim=0)
    if history_lens is not None:
        curr_lens = torch.cat((history_lens, curr_lens), dim=0)
    torch.save((curr_rets, curr_lens), fpath)
    return


def plot_results():
    """ Plot the saved results of training progress, per iteration. """
    vpg_fpath = Path(f'cartpole-results-vpg.pth')
    fitre_fpath = Path(f'cartpole-results-fitre.pth')
    if not vpg_fpath.is_file():
        print(f'{vpg_fpath} is not a valid file')
        return
    if not fitre_fpath.is_file():
        print(f'{fitre_fpath} is not a valid file')
        return

    # this tutorial code somehow has rets/lens being the same, thus just using rets for plotting
    vpg_rets, _ = torch.load(vpg_fpath)
    fitre_rets, _ = torch.load(fitre_fpath)
    print(f'Plotted vpg results from {vpg_rets.shape[0]} training runs, each having {vpg_rets.shape[1]} iterations.')
    print(f'Plotted fitre results from {fitre_rets.shape[0]} training runs, each having {fitre_rets.shape[1]} iterations.')

    plt.clf()
    fig, ax = plt.subplots()

    def _plot(rets: Tensor, color: str, label: str):
        n_iters = rets.shape[1]
        if len(rets) == 1:
            std = np.zeros((n_iters))
            mean = rets.cpu().numpy()
        else:
            std, mean = torch.std_mean(rets, dim=0)

        xs = np.arange(1, n_iters + 1)
        ax.errorbar(xs, mean, yerr=std, color=color, label=label)
        return

    _plot(vpg_rets, 'b', 'vpg')
    _plot(fitre_rets, 'r', 'fitre')

    plt.legend()
    plt.title(f'vpg/fitre mean/std rets per iteration on CartPole')

    # save as svg if need to preserve the details
    plt.savefig(f'cartpole-results.png')
    plt.close(fig)
    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)

    parser.add_argument('--no_vpg', action='store_true', default=False,
                        help='if True, not running vanilla policy gradient')
    parser.add_argument('--no_fitre', action='store_true', default=False,
                        help='if True, not running FITRE optimizer')
    parser.add_argument('--no_train', action='store_true', default=False,
                        help='if True, no training, just plot the figure using current saved data')
    parser.add_argument('--runs', type=int, default=1)

    args = parser.parse_args()
    if not args.no_train:
        assert args.runs > 0
        for i in range(args.runs):
            print(f'--- {i}-th run ---')
            if not args.no_vpg:
                print('\nUsing simplest formulation of policy gradient.\n')
                train(env_name=args.env_name, render=args.render, lr=args.lr, use_fitre=False)
            if not args.no_fitre:
                print('\nUsing FITRE.\n')
                train(env_name=args.env_name, render=args.render, lr=args.lr, use_fitre=True)

    plot_results()
    pass
