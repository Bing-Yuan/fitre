""" Modified PPO to support FITRE solver. """

from functools import partial
from itertools import zip_longest
from typing import List, Union, Dict, Type, Callable

import numpy as np
import torch as th
from torch.nn import functional as F
from gym import spaces

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common import logger
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.distributions import DiagGaussianDistribution, StateDependentNoiseDistribution, \
    CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution
from stable_baselines3.common.policies import create_sde_features_extractor
from stable_baselines3.common.utils import explained_variance, get_device

from tr_kfac_opt import KFACOptimizer


class FitreMlpExtractor(MlpExtractor):
    """ Inherited to disable bias in Linear layers. """

    def __init__(
        self,
        feature_dim: int,
        net_arch: List[Union[int, Dict[str, List[int]]]],
        activation_fn: Type[th.nn.Module],
        device: Union[th.device, str] = "auto",
    ):
        super(MlpExtractor, self).__init__()
        device = get_device(device)
        shared_net, policy_net, value_net = [], [], []
        policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
        value_only_layers = []  # Layer sizes of the network that only belongs to the value network
        last_layer_dim_shared = feature_dim

        # Iterate through the shared layers and build the shared parts of the network
        for idx, layer in enumerate(net_arch):
            if isinstance(layer, int):  # Check that this is a shared layer
                layer_size = layer
                # TODO: give layer a meaningful name
                shared_net.append(th.nn.Linear(last_layer_dim_shared, layer_size, bias=False))  # change bias to False
                shared_net.append(activation_fn())
                last_layer_dim_shared = layer_size
            else:
                assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                if "pi" in layer:
                    assert isinstance(layer["pi"], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                    policy_only_layers = layer["pi"]

                if "vf" in layer:
                    assert isinstance(layer["vf"], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                    value_only_layers = layer["vf"]
                break  # From here on the network splits up in policy and value network

        last_layer_dim_pi = last_layer_dim_shared
        last_layer_dim_vf = last_layer_dim_shared

        # Build the non-shared part of the network
        for idx, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
            if pi_layer_size is not None:
                assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                policy_net.append(th.nn.Linear(last_layer_dim_pi, pi_layer_size, bias=False))  # change bias to False
                policy_net.append(activation_fn())
                last_layer_dim_pi = pi_layer_size

            if vf_layer_size is not None:
                assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                value_net.append(th.nn.Linear(last_layer_dim_vf, vf_layer_size, bias=False))  # change bias to False
                value_net.append(activation_fn())
                last_layer_dim_vf = vf_layer_size

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.shared_net = th.nn.Sequential(*shared_net).to(device)
        self.policy_net = th.nn.Sequential(*policy_net).to(device)
        self.value_net = th.nn.Sequential(*value_net).to(device)
        return
    pass


class FitreMlpPolicy(MlpPolicy):
    """ Inherited to disable the parameter of log_std. """

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = FitreMlpExtractor(
            self.features_dim, net_arch=self.net_arch, activation_fn=self.activation_fn, device=self.device
        )
        return

    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        # Separate features extractor for gSDE
        if self.sde_net_arch is not None:
            self.sde_features_extractor, latent_sde_dim = create_sde_features_extractor(
                self.features_dim, self.sde_net_arch, self.activation_fn
            )

        if isinstance(self.action_dist, DiagGaussianDistribution):
            # self.action_net, self.log_std = self.action_dist.proba_distribution_net(
            #     latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            # )
            self.action_net = th.nn.Linear(latent_dim_pi, self.action_dist.action_dim, bias=False)  # change bias to False
            # TODO: allow action dependent std
            # self.log_std = th.nn.Parameter(th.ones(self.action_dist.action_dim) * self.log_std_init, requires_grad=True)
            # fix log_std, otherwise making FITRE harder, but still needs it to move to GPU automatically
            self.register_buffer('log_std', th.ones(self.action_dist.action_dim) * self.log_std_init)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            latent_sde_dim = latent_dim_pi if self.sde_net_arch is None else latent_sde_dim
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_sde_dim, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, CategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, BernoulliDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = th.nn.Linear(self.mlp_extractor.latent_dim_vf, 1, bias=False)  # TODO change bias to False
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        return
    pass


class FitrePPO(PPO):
    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer.
        """
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, all_kl_divs = [], []
        pg_losses, value_losses = [], []
        clip_fractions = []

        # train for gradient_steps epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                # TODO: investigate why there is no issue with the gradient
                # if that line is commented (as in SAC)
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)

                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                if isinstance(self.policy.optimizer, KFACOptimizer):
                    self.policy.optimizer.zero_grad()
                    self.policy.optimizer.acc_stats = True

                    # experiments show that his coef is better had
                    loss = self.ent_coef * -th.mean(-log_prob) + self.vf_coef * value_loss
                    #loss = -th.mean(-log_prob) + value_loss
                    loss.backward(retain_graph=True)

                    policy = self.policy
                    ent_coef = self.ent_coef
                    vf_coef = self.vf_coef
                    def _closure_fn():
                        with th.no_grad():
                            values, log_prob, entropy = policy.evaluate_actions(rollout_data.observations, actions)
                            values = values.flatten()
                            # Normalize advantage
                            advantages = rollout_data.advantages
                            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                            # ratio between old and new policy, should be one at the first iteration
                            ratio = th.exp(log_prob - rollout_data.old_log_prob)

                            # clipped surrogate loss
                            policy_loss_1 = advantages * ratio
                            policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                            policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                            if self.clip_range_vf is None:
                                # No clipping
                                values_pred = values
                            else:
                                # Clip the different between old and new value
                                # NOTE: this depends on the reward scaling
                                values_pred = rollout_data.old_values + th.clamp(
                                    values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                                )
                            # Value loss using the TD(gae_lambda) target
                            value_loss = F.mse_loss(rollout_data.returns, values_pred)

                            # Entropy loss favor exploration
                            if entropy is None:
                                # Approximate entropy when no analytical form
                                entropy_loss = -th.mean(-log_prob)
                            else:
                                entropy_loss = -th.mean(entropy)
                            return policy_loss + ent_coef * entropy_loss + vf_coef * value_loss

                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                    self.policy.optimizer.zero_grad()
                    self.policy.optimizer.acc_stats = False
                    loss.backward(create_graph=True)
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step(closure=_closure_fn)
                else:
                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                    # Optimization step
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()

                approx_kl_divs.append(th.mean(rollout_data.old_log_prob - log_prob).detach().cpu().numpy())

            all_kl_divs.append(np.mean(approx_kl_divs))

            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                print(f"Early stopping at step {epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}")
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.returns.flatten(), self.rollout_buffer.values.flatten())

        # Logs
        logger.record("train/entropy_loss", np.mean(entropy_losses))
        logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        logger.record("train/value_loss", np.mean(value_losses))
        logger.record("train/approx_kl", np.mean(approx_kl_divs))
        logger.record("train/clip_fraction", np.mean(clip_fractions))
        logger.record("train/loss", loss.item())
        logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            logger.record("train/clip_range_vf", clip_range_vf)
        return
    pass
