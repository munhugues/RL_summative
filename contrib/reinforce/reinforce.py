import torch
import gymnasium as gym
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.base_class import BaseAlgorithm


class REINFORCE(BaseAlgorithm):
    def __init__(
        self,
        policy: type = ActorCriticPolicy,
        env: VecEnv = None,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        verbose: int = 0,
        **kwargs,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            verbose=verbose,
            supported_action_spaces=(gym.spaces.Discrete,),
            **kwargs,
        )
        self.gamma = gamma

    def _setup_model(self) -> None:
        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            lr_schedule=lambda _: self.learning_rate,
        )
        self.policy = self.policy.to(self.device)

    def select_action(self, obs):
        import numpy as np
        obs_tensor = obs_as_tensor(obs[None] if obs.ndim == 1 else obs, self.device)
        dist = self.policy.get_distribution(obs_tensor)
        action = dist.sample()
        return int(action.cpu().numpy()), dist.log_prob(action)

    def _run_episode(self):
        obs = self.env.reset()
        log_probs, rewards = [], []
        for _ in range(self.n_steps):
            obs_tensor = obs_as_tensor(obs, self.device)
            dist = self.policy.get_distribution(obs_tensor)
            actions = dist.sample()
            log_probs.append(dist.log_prob(actions))
            obs, reward, done, _ = self.env.step(actions.cpu().numpy())
            rewards.append(torch.tensor(reward, dtype=torch.float32, device=self.device))
            if done.any():
                break
        return log_probs, rewards

    def _update(self, log_probs, rewards):
        returns, G = [], 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.stack(returns).detach()
        loss = -torch.sum(torch.stack(log_probs) * returns)
        self.policy.optimizer.zero_grad()
        loss.backward()
        self.policy.optimizer.step()
        self._update_learning_rate([self.policy.optimizer])
        self.logger.record("train/loss", loss.item())

    def learn(
        self,
        total_timesteps: int,
        callback=None,
        log_interval=1,
        tb_log_name="REINFORCE",
        reset_num_timesteps=True,
        progress_bar=False,
    ):
        self._setup_model()
        self.n_steps = 2048
        timesteps = 0
        while timesteps < total_timesteps:
            log_probs, rewards = self._run_episode()
            self._update(log_probs, rewards)
            timesteps += self.n_steps
        return self
