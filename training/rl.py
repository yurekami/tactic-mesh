"""
Reinforcement Learning for Tactic Selection.

Implements hierarchical RL with:
- Step-level rewards (tactic success)
- Episode-level rewards (proof completion)
- Proof tree credit assignment
- GAE for advantage estimation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from core.proof_state import Goal, Proof, ProofState, TacticOutput
from core.router import TacticRouter

logger = logging.getLogger(__name__)


@dataclass
class RLConfig:
    """Configuration for RL training."""
    # Reward structure
    step_reward: float = 0.1          # Reward for successful tactic
    completion_reward: float = 1.0     # Reward for completing proof
    failure_penalty: float = -0.05     # Penalty for failed tactic
    length_penalty: float = -0.01      # Penalty per step (encourage efficiency)

    # PPO parameters
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01

    # GAE parameters
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Training
    batch_size: int = 32
    n_epochs: int = 4
    learning_rate: float = 1e-4
    max_grad_norm: float = 0.5

    # Credit assignment
    use_hindsight_credit: bool = True
    credit_decay: float = 0.9


@dataclass
class ProofStep:
    """A single step in a proof attempt."""
    state: ProofState
    tactic: TacticOutput
    expert_idx: int
    log_prob: float
    value: float
    reward: float = 0.0
    done: bool = False

    # For tree credit assignment
    parent_idx: Optional[int] = None
    children_idx: List[int] = field(default_factory=list)


@dataclass
class ProofEpisode:
    """A complete proof episode (attempt)."""
    goal: Goal
    steps: List[ProofStep] = field(default_factory=list)
    success: bool = False
    total_reward: float = 0.0

    def add_step(self, step: ProofStep) -> int:
        """Add step and return its index."""
        idx = len(self.steps)
        self.steps.append(step)
        return idx

    def get_step(self, idx: int) -> ProofStep:
        return self.steps[idx]


class ProofTreeCreditAssignment:
    """
    Credit assignment for proof trees.

    Propagates rewards through the proof DAG structure,
    not just sequentially.
    """

    def __init__(self, config: RLConfig):
        self.config = config

    def assign_credit(self, episode: ProofEpisode) -> None:
        """
        Assign credit to all steps in the episode.

        Uses hindsight credit assignment - when a branch closes,
        propagate reward to ancestor decisions.
        """
        if not episode.steps:
            return

        # Terminal reward
        terminal_reward = (
            self.config.completion_reward if episode.success
            else 0.0
        )

        # Work backwards through the proof tree
        self._propagate_rewards(episode, terminal_reward)

    def _propagate_rewards(
        self,
        episode: ProofEpisode,
        terminal_reward: float,
    ) -> None:
        """Propagate rewards through proof tree."""
        n_steps = len(episode.steps)

        # Initialize with step rewards
        for step in episode.steps:
            if step.done and episode.success:
                step.reward += terminal_reward / max(n_steps, 1)

        # Hindsight credit: propagate from leaves to roots
        if self.config.use_hindsight_credit:
            # Build reverse adjacency (child → parents)
            for i in range(n_steps - 1, -1, -1):
                step = episode.steps[i]

                if step.parent_idx is not None:
                    parent = episode.steps[step.parent_idx]
                    # Propagate portion of reward to parent
                    credit = step.reward * self.config.credit_decay
                    parent.reward += credit

        # Compute cumulative rewards
        episode.total_reward = sum(s.reward for s in episode.steps)


class GAEComputer:
    """
    Generalized Advantage Estimation for proof RL.
    """

    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def compute_advantages(
        self,
        rewards: Tensor,
        values: Tensor,
        dones: Tensor,
        next_value: float = 0.0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute GAE advantages and returns.

        Args:
            rewards: [T] step rewards
            values: [T] value estimates
            dones: [T] episode termination flags
            next_value: Value estimate for state after final step

        Returns:
            advantages: [T] advantage estimates
            returns: [T] return estimates
        """
        T = len(rewards)
        advantages = torch.zeros(T)
        returns = torch.zeros(T)

        gae = 0.0
        next_val = next_value

        for t in reversed(range(T)):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * next_val - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae

            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            next_val = values[t]

        return advantages, returns


class PPOTrainer:
    """
    PPO trainer for tactic selection.
    """

    def __init__(
        self,
        policy: nn.Module,
        value_net: nn.Module,
        config: RLConfig,
    ):
        self.policy = policy
        self.value_net = value_net
        self.config = config

        self.optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(value_net.parameters()),
            lr=config.learning_rate,
        )

        self.gae = GAEComputer(config.gamma, config.gae_lambda)
        self.credit = ProofTreeCreditAssignment(config)

    def train_on_episodes(
        self,
        episodes: List[ProofEpisode],
    ) -> Dict[str, float]:
        """
        Train on a batch of proof episodes.

        Args:
            episodes: List of proof episodes

        Returns:
            Training metrics
        """
        # Assign credit to all episodes
        for episode in episodes:
            self.credit.assign_credit(episode)

        # Collect transitions
        states = []
        actions = []
        old_log_probs = []
        rewards = []
        values = []
        dones = []

        for episode in episodes:
            for step in episode.steps:
                states.append(step.state)
                actions.append(step.expert_idx)
                old_log_probs.append(step.log_prob)
                rewards.append(step.reward)
                values.append(step.value)
                dones.append(step.done)

        if not states:
            return {"loss": 0.0}

        # Convert to tensors
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        values_t = torch.tensor(values, dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32)
        old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.long)

        # Compute advantages
        advantages, returns = self.gae.compute_advantages(
            rewards_t, values_t, dones_t
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        metrics = self._ppo_update(
            states, actions_t, old_log_probs_t,
            advantages, returns
        )

        return metrics

    def _ppo_update(
        self,
        states: List[ProofState],
        actions: Tensor,
        old_log_probs: Tensor,
        advantages: Tensor,
        returns: Tensor,
    ) -> Dict[str, float]:
        """Perform PPO update."""
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        n_samples = len(states)
        indices = list(range(n_samples))

        for _ in range(self.config.n_epochs):
            # Shuffle
            import random
            random.shuffle(indices)

            # Mini-batches
            for start in range(0, n_samples, self.config.batch_size):
                end = min(start + self.config.batch_size, n_samples)
                batch_idx = indices[start:end]

                # Get batch
                batch_states = [states[i] for i in batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # Forward pass
                log_probs, values, entropy = self._forward_batch(
                    batch_states, batch_actions
                )

                # Policy loss (PPO clip)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1 - self.config.clip_epsilon,
                    1 + self.config.clip_epsilon
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss +
                    self.config.value_coef * value_loss +
                    self.config.entropy_coef * entropy_loss
                )

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_net.parameters()),
                    self.config.max_grad_norm
                )
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()

        n_updates = (n_samples // self.config.batch_size + 1) * self.config.n_epochs

        return {
            "loss": total_loss / n_updates,
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def _forward_batch(
        self,
        states: List[ProofState],
        actions: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass for a batch of states."""
        # This would use the actual policy network
        # Simplified placeholder implementation
        batch_size = len(states)

        # Mock log probs (would come from policy)
        log_probs = torch.randn(batch_size) * 0.1 - 1.0

        # Mock values (would come from value network)
        values = torch.randn(batch_size) * 0.5

        # Mock entropy
        entropy = torch.ones(batch_size) * 0.5

        return log_probs, values, entropy


class ImitationLearning:
    """
    Imitation learning from existing proof corpora.

    Used to warm-start the policy before RL.
    """

    def __init__(
        self,
        policy: nn.Module,
        learning_rate: float = 1e-4,
    ):
        self.policy = policy
        self.optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=learning_rate,
        )

    def train_on_proofs(
        self,
        proofs: List[Tuple[ProofState, TacticOutput]],
        n_epochs: int = 10,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """
        Train policy to imitate expert proofs.

        Args:
            proofs: List of (state, tactic) pairs from expert proofs
            n_epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            Training metrics
        """
        total_loss = 0.0
        n_batches = 0

        for _ in range(n_epochs):
            # Shuffle
            import random
            shuffled = list(proofs)
            random.shuffle(shuffled)

            for start in range(0, len(shuffled), batch_size):
                end = min(start + batch_size, len(shuffled))
                batch = shuffled[start:end]

                states = [p[0] for p in batch]
                tactics = [p[1] for p in batch]

                # Compute loss
                loss = self._imitation_loss(states, tactics)

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        return {
            "imitation_loss": total_loss / max(n_batches, 1),
        }

    def _imitation_loss(
        self,
        states: List[ProofState],
        tactics: List[TacticOutput],
    ) -> Tensor:
        """Compute imitation loss (cross-entropy)."""
        # This would use actual policy forward pass
        # Placeholder implementation
        return torch.tensor(0.1, requires_grad=True)


class SelfPlayTrainer:
    """
    Self-play training for theorem proving.

    Generates proof attempts and learns from them.
    """

    def __init__(
        self,
        policy: nn.Module,
        value_net: nn.Module,
        verifier: Any,  # LeanVerifier
        config: RLConfig,
    ):
        self.policy = policy
        self.value_net = value_net
        self.verifier = verifier
        self.config = config

        self.ppo_trainer = PPOTrainer(policy, value_net, config)
        self.episode_buffer: List[ProofEpisode] = []

    async def run_episode(
        self,
        goal: Goal,
        max_steps: int = 50,
    ) -> ProofEpisode:
        """
        Run a single proof episode.

        Args:
            goal: Goal to prove
            max_steps: Maximum proof steps

        Returns:
            ProofEpisode with all steps
        """
        episode = ProofEpisode(goal=goal)
        current_goals = [goal]
        step_idx = 0

        while current_goals and step_idx < max_steps:
            # Get current goal
            current_goal = current_goals[0]

            # Create proof state
            state = ProofState.from_goal(current_goal)

            # Select tactic using policy
            tactic, expert_idx, log_prob = await self._select_tactic(state)

            # Get value estimate
            value = self._estimate_value(state)

            # Apply tactic
            result = await self.verifier.apply_tactic(current_goal, tactic)

            # Compute step reward
            if result.success:
                reward = self.config.step_reward
                if result.is_complete:
                    reward += self.config.completion_reward
            else:
                reward = self.config.failure_penalty

            reward += self.config.length_penalty

            # Create step
            step = ProofStep(
                state=state,
                tactic=tactic,
                expert_idx=expert_idx,
                log_prob=log_prob,
                value=value,
                reward=reward,
                done=result.is_complete or not result.success,
            )

            episode.add_step(step)

            # Update goals
            if result.success:
                current_goals = current_goals[1:] + result.new_goals
            else:
                current_goals = current_goals[1:]

            step_idx += 1

        episode.success = len(current_goals) == 0 and step_idx > 0

        return episode

    async def _select_tactic(
        self,
        state: ProofState,
    ) -> Tuple[TacticOutput, int, float]:
        """Select tactic using policy."""
        # This would use actual policy
        # Placeholder implementation
        from experts.base import IntroExpert

        expert = IntroExpert()
        tactic = expert.generate_tactic(state, torch.randn(1024))

        return tactic, 0, -1.0

    def _estimate_value(self, state: ProofState) -> float:
        """Estimate value of proof state."""
        # This would use actual value network
        return 0.0

    async def train(
        self,
        goals: List[Goal],
        n_iterations: int = 100,
        episodes_per_iter: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Run self-play training.

        Args:
            goals: Training goals
            n_iterations: Number of training iterations
            episodes_per_iter: Episodes per iteration

        Returns:
            Training history
        """
        history = {
            "loss": [],
            "success_rate": [],
            "avg_reward": [],
        }

        for iteration in range(n_iterations):
            # Collect episodes
            episodes = []
            for _ in range(episodes_per_iter):
                # Sample goal
                import random
                goal = random.choice(goals)

                # Run episode
                episode = await self.run_episode(goal)
                episodes.append(episode)

            # Train on episodes
            metrics = self.ppo_trainer.train_on_episodes(episodes)

            # Record metrics
            history["loss"].append(metrics["loss"])
            history["success_rate"].append(
                sum(1 for e in episodes if e.success) / len(episodes)
            )
            history["avg_reward"].append(
                sum(e.total_reward for e in episodes) / len(episodes)
            )

            logger.info(
                f"Iteration {iteration}: "
                f"loss={metrics['loss']:.4f}, "
                f"success_rate={history['success_rate'][-1]:.2%}, "
                f"avg_reward={history['avg_reward'][-1]:.4f}"
            )

        return history
