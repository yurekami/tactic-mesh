"""
Mixture-of-Tactics Router.

Routes proof states to specialized tactic experts using an
auxiliary-loss-free load balancing strategy inspired by DeepSeek-V3.

Key innovations:
- Auxiliary-loss-free balancing (no performance degradation)
- Proof-state-aware routing features
- Multi-expert selection for ensemble tactics
- Dynamic expert capacity adjustment
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from core.proof_state import ProofState, RoutingFeatures


@dataclass
class RouterConfig:
    """Configuration for the tactic router."""
    n_experts: int = 16           # Number of tactic experts
    d_routing: int = 128          # Routing feature dimension
    top_k: int = 2                # Number of experts to route to
    capacity_factor: float = 1.25  # Expert capacity multiplier

    # Auxiliary-loss-free settings
    use_aux_loss: bool = False    # Disable auxiliary loss
    balance_method: str = "bias"  # "bias" or "capacity"

    # Expert groups (for hierarchical routing)
    expert_groups: Optional[Dict[str, List[int]]] = None


class TacticRouter(nn.Module):
    """
    Mixture-of-Tactics Router.

    Routes proof states to the most appropriate tactic experts
    without using auxiliary losses that degrade performance.

    The key insight from DeepSeek-V3 is that auxiliary losses
    for load balancing hurt model performance. Instead, we use
    learned bias terms and dynamic capacity adjustment.
    """

    def __init__(self, config: RouterConfig):
        super().__init__()
        self.config = config

        # Routing network
        self.routing_net = nn.Sequential(
            nn.Linear(config.d_routing, config.d_routing * 2),
            nn.GELU(),
            nn.Linear(config.d_routing * 2, config.d_routing),
            nn.GELU(),
            nn.Linear(config.d_routing, config.n_experts),
        )

        # Learned bias for load balancing (auxiliary-loss-free)
        self.expert_bias = nn.Parameter(torch.zeros(config.n_experts))

        # Expert capacity tracking
        self.register_buffer(
            "expert_counts",
            torch.zeros(config.n_experts, dtype=torch.long)
        )
        self.register_buffer(
            "total_tokens",
            torch.tensor(0, dtype=torch.long)
        )

        # Expert group routing (for hierarchical selection)
        if config.expert_groups:
            self.group_router = GroupRouter(config)
        else:
            self.group_router = None

        # Temperature for routing
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(
        self,
        routing_features: Tensor,
        return_all_scores: bool = False,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Route proof states to experts.

        Args:
            routing_features: [batch_size, d_routing]
            return_all_scores: Whether to return full score matrix

        Returns:
            expert_indices: [batch_size, top_k] - Selected expert IDs
            expert_weights: [batch_size, top_k] - Routing weights
            all_scores: [batch_size, n_experts] - Full scores (optional)
        """
        # Compute routing logits
        logits = self.routing_net(routing_features)  # [batch, n_experts]

        # Add learned bias for load balancing
        logits = logits + self.expert_bias

        # Apply temperature
        logits = logits / self.temperature.clamp(min=0.1)

        # Get top-k experts
        scores = F.softmax(logits, dim=-1)
        top_scores, top_indices = scores.topk(self.config.top_k, dim=-1)

        # Renormalize weights for selected experts
        top_weights = top_scores / top_scores.sum(dim=-1, keepdim=True)

        # Update expert counts (for monitoring)
        if self.training:
            self._update_counts(top_indices)

        all_scores = scores if return_all_scores else None
        return top_indices, top_weights, all_scores

    def _update_counts(self, expert_indices: Tensor) -> None:
        """Update expert selection counts."""
        for idx in expert_indices.flatten():
            self.expert_counts[idx] += 1
        self.total_tokens += expert_indices.numel()

    def get_load_balance_stats(self) -> Dict[str, float]:
        """Get load balancing statistics."""
        if self.total_tokens == 0:
            return {"balance_ratio": 1.0, "max_load": 0.0}

        expert_freq = self.expert_counts.float() / self.total_tokens.float()
        ideal_freq = 1.0 / self.config.n_experts

        # Balance ratio: 1.0 = perfect balance
        balance_ratio = (expert_freq.min() / expert_freq.max()).item()

        # Max load relative to ideal
        max_load = (expert_freq.max() / ideal_freq).item()

        return {
            "balance_ratio": balance_ratio,
            "max_load": max_load,
            "expert_frequencies": expert_freq.tolist(),
        }

    def adjust_bias(self, target_balance: float = 0.9) -> None:
        """
        Adjust expert bias to improve load balancing.

        This is the auxiliary-loss-free balancing mechanism.
        Instead of adding a loss term, we directly adjust biases.
        """
        if self.total_tokens == 0:
            return

        expert_freq = self.expert_counts.float() / self.total_tokens.float()
        ideal_freq = 1.0 / self.config.n_experts

        # Decrease bias for overused experts, increase for underused
        freq_diff = ideal_freq - expert_freq
        adjustment = freq_diff * 0.1  # Small adjustment step

        with torch.no_grad():
            self.expert_bias.add_(adjustment)

        # Reset counts
        self.expert_counts.zero_()
        self.total_tokens.zero_()


class GroupRouter(nn.Module):
    """
    Hierarchical group routing for expert selection.

    First routes to an expert group (e.g., "introduction", "rewriting"),
    then routes within the group to specific experts.
    """

    def __init__(self, config: RouterConfig):
        super().__init__()
        self.config = config
        self.groups = config.expert_groups

        # Group-level router
        n_groups = len(self.groups)
        self.group_router = nn.Linear(config.d_routing, n_groups)

        # Within-group routers
        self.in_group_routers = nn.ModuleDict({
            name: nn.Linear(config.d_routing, len(experts))
            for name, experts in self.groups.items()
        })

    def forward(
        self,
        routing_features: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Hierarchical routing: group → expert.

        Returns:
            expert_indices: [batch, top_k]
            expert_weights: [batch, top_k]
        """
        batch_size = routing_features.shape[0]

        # Route to group
        group_logits = self.group_router(routing_features)
        group_probs = F.softmax(group_logits, dim=-1)
        top_group = group_probs.argmax(dim=-1)

        # Route within group
        expert_indices = []
        expert_weights = []

        group_names = list(self.groups.keys())
        for i in range(batch_size):
            group_name = group_names[top_group[i].item()]
            group_experts = self.groups[group_name]

            # Get within-group scores
            in_group_logits = self.in_group_routers[group_name](
                routing_features[i:i+1]
            )
            in_group_probs = F.softmax(in_group_logits, dim=-1)

            # Get top-k within group
            k = min(self.config.top_k, len(group_experts))
            top_weights, top_in_group = in_group_probs.topk(k, dim=-1)

            # Map back to global expert indices
            global_indices = torch.tensor(
                [group_experts[j] for j in top_in_group[0].tolist()],
                device=routing_features.device
            )

            expert_indices.append(global_indices)
            expert_weights.append(top_weights[0])

        return (
            torch.stack(expert_indices),
            torch.stack(expert_weights),
        )


class RoutingFeatureExtractor(nn.Module):
    """
    Extract routing features from proof states.

    Converts proof state information into compact routing features
    that the router uses for expert selection.
    """

    def __init__(self, d_routing: int = 128):
        super().__init__()
        self.d_routing = d_routing

        # Input dimension from RoutingFeatures (15 features)
        self.feature_encoder = nn.Sequential(
            nn.Linear(15, d_routing),
            nn.GELU(),
            nn.Linear(d_routing, d_routing),
            nn.LayerNorm(d_routing),
        )

        # Optional: encode proof state embedding
        self.use_state_embedding = True
        if self.use_state_embedding:
            self.state_compressor = nn.Sequential(
                nn.Linear(1024, d_routing),  # Assuming d_model=1024
                nn.GELU(),
            )
            self.combine = nn.Linear(d_routing * 2, d_routing)

    def forward(
        self,
        routing_features: RoutingFeatures,
        state_embedding: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Extract routing features.

        Args:
            routing_features: Compact routing features from proof state
            state_embedding: Optional full proof state embedding

        Returns:
            Routing representation [batch, d_routing]
        """
        # Encode compact features
        compact = routing_features.to_tensor()
        if compact.dim() == 1:
            compact = compact.unsqueeze(0)

        encoded = self.feature_encoder(compact)

        # Combine with state embedding if available
        if self.use_state_embedding and state_embedding is not None:
            state_compressed = self.state_compressor(state_embedding)
            encoded = self.combine(
                torch.cat([encoded, state_compressed], dim=-1)
            )

        return encoded


class MixtureOfTacticsLayer(nn.Module):
    """
    Full Mixture-of-Tactics layer.

    Combines routing with expert execution and output aggregation.
    """

    def __init__(
        self,
        router: TacticRouter,
        experts: nn.ModuleList,
        d_model: int = 1024,
    ):
        super().__init__()
        self.router = router
        self.experts = experts
        self.d_model = d_model

        # Output aggregation
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        proof_state_embedding: Tensor,
        routing_features: Tensor,
    ) -> Tuple[Tensor, Dict[str, any]]:
        """
        Forward pass through MoT layer.

        Args:
            proof_state_embedding: [batch, d_model]
            routing_features: [batch, d_routing]

        Returns:
            output: [batch, d_model]
            aux_info: Auxiliary information (expert indices, weights, etc.)
        """
        batch_size = proof_state_embedding.shape[0]

        # Get expert routing
        expert_indices, expert_weights, all_scores = self.router(
            routing_features,
            return_all_scores=True,
        )

        # Execute selected experts and aggregate
        output = torch.zeros(batch_size, self.d_model, device=proof_state_embedding.device)

        for i in range(batch_size):
            for k in range(self.router.config.top_k):
                expert_idx = expert_indices[i, k].item()
                weight = expert_weights[i, k]

                expert_output = self.experts[expert_idx](
                    proof_state_embedding[i:i+1]
                )
                output[i] += weight * expert_output.squeeze(0)

        output = self.output_proj(output)

        aux_info = {
            "expert_indices": expert_indices,
            "expert_weights": expert_weights,
            "all_scores": all_scores,
            "load_balance": self.router.get_load_balance_stats(),
        }

        return output, aux_info
