"""
Base Tactic Expert Implementation.

Defines the interface and common functionality for all tactic experts
in the Mixture-of-Tactics architecture.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from core.proof_state import ProofState, TacticOutput, TermAST


@dataclass
class ExpertConfig:
    """Configuration for tactic experts."""
    expert_id: str                # Unique expert identifier
    tactic_name: str              # Lean tactic this expert handles
    d_model: int = 1024           # Model dimension
    d_hidden: int = 2048          # Hidden dimension
    n_layers: int = 2             # Number of layers
    dropout: float = 0.1         # Dropout rate

    # Expert-specific settings
    max_args: int = 5             # Maximum number of arguments
    use_hypothesis_attention: bool = True
    use_type_encoding: bool = True


class TacticExpert(nn.Module, ABC):
    """
    Base class for tactic experts.

    Each expert specializes in a particular proof tactic (induction,
    rewriting, case analysis, etc.) and learns to generate appropriate
    arguments for that tactic given the proof state.
    """

    def __init__(self, config: ExpertConfig):
        super().__init__()
        self.config = config
        self.expert_id = config.expert_id
        self.tactic_name = config.tactic_name

        # Core network
        self.encoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_hidden, config.d_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_hidden, config.d_model),
        )

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 4),
            nn.GELU(),
            nn.Linear(config.d_model // 4, 1),
            nn.Sigmoid(),
        )

        # Layer norm
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, proof_state_embedding: Tensor) -> Tensor:
        """
        Forward pass through expert.

        Args:
            proof_state_embedding: [batch, d_model]

        Returns:
            Expert output: [batch, d_model]
        """
        encoded = self.encoder(proof_state_embedding)
        return self.norm(encoded + proof_state_embedding)  # Residual

    def get_confidence(self, proof_state_embedding: Tensor) -> Tensor:
        """Get expert's confidence in handling this proof state."""
        return self.confidence_head(proof_state_embedding)

    @abstractmethod
    def generate_tactic(
        self,
        proof_state: ProofState,
        attention_output: Tensor,
    ) -> TacticOutput:
        """
        Generate a tactic application for the proof state.

        Args:
            proof_state: Full proof state
            attention_output: Output from attention mechanism

        Returns:
            TacticOutput with tactic name and arguments
        """
        pass

    @abstractmethod
    def get_specialization_score(self, proof_state: ProofState) -> float:
        """
        Score how well this expert matches the proof state.

        Higher scores indicate this expert is more suitable.
        Used for expert selection during inference.
        """
        pass


class IntroExpert(TacticExpert):
    """
    Expert for introduction tactics.

    Handles: intro, intros, rintro, fun
    Specializes in introducing hypotheses for goals with forall/pi binders.
    """

    def __init__(self, config: Optional[ExpertConfig] = None):
        if config is None:
            config = ExpertConfig(
                expert_id="intro",
                tactic_name="intro",
            )
        super().__init__(config)

        # Name generation for introduced variables
        self.name_generator = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 26),  # a-z
        )

        # Multi-intro detection
        self.multi_intro_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 4),
            nn.GELU(),
            nn.Linear(config.d_model // 4, 10),  # Max 10 intros at once
            nn.Softmax(dim=-1),
        )

    def generate_tactic(
        self,
        proof_state: ProofState,
        attention_output: Tensor,
    ) -> TacticOutput:
        """Generate intro tactic with appropriate names."""
        # Determine number of intros
        multi_probs = self.multi_intro_head(attention_output)
        n_intros = multi_probs.argmax(dim=-1).item() + 1

        # Generate names
        names = []
        for i in range(n_intros):
            name_logits = self.name_generator(attention_output)
            name_idx = name_logits.argmax(dim=-1).item()
            names.append(chr(ord('a') + name_idx))

        # Get confidence
        confidence = self.get_confidence(attention_output).item()

        if n_intros == 1:
            return TacticOutput(
                tactic="intro",
                args={"name": names[0]},
                confidence=confidence,
                expert_id=self.expert_id,
            )
        else:
            return TacticOutput(
                tactic="intro",
                args={"names": " ".join(names)},
                confidence=confidence,
                expert_id=self.expert_id,
            )

    def get_specialization_score(self, proof_state: ProofState) -> float:
        """Score based on whether goal has forall/pi structure."""
        target = proof_state.goal.target
        if target.kind.name == "PI":
            return 0.9
        return 0.1


class RewriteExpert(TacticExpert):
    """
    Expert for rewriting tactics.

    Handles: rw, rewrite, simp_rw, conv
    Specializes in applying equality hypotheses to transform goals.
    """

    def __init__(self, config: Optional[ExpertConfig] = None):
        if config is None:
            config = ExpertConfig(
                expert_id="rewrite",
                tactic_name="rw",
            )
        super().__init__(config)

        # Hypothesis selector (attention over hypotheses)
        self.hyp_attention = nn.MultiheadAttention(
            config.d_model,
            num_heads=8,
            dropout=config.dropout,
            batch_first=True,
        )

        # Direction head (left-to-right vs right-to-left)
        self.direction_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 4),
            nn.GELU(),
            nn.Linear(config.d_model // 4, 2),
            nn.Softmax(dim=-1),
        )

        # Location head (where to rewrite)
        self.location_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 3),  # goal, hyp, everywhere
        )

    def generate_tactic(
        self,
        proof_state: ProofState,
        attention_output: Tensor,
    ) -> TacticOutput:
        """Generate rewrite tactic with lemma selection."""
        # Find equality hypotheses
        eq_hyps = [h for h in proof_state.hypotheses
                   if self._is_equality(h.type)]

        if not eq_hyps:
            # Fall back to simp
            return TacticOutput(
                tactic="simp",
                args={},
                confidence=0.3,
                expert_id=self.expert_id,
            )

        # Select hypothesis using attention
        # (Simplified: in full implementation, encode hypotheses)
        selected_hyp = eq_hyps[0].name  # Select first for now

        # Determine direction
        direction = self.direction_head(attention_output)
        use_reverse = direction[..., 1] > direction[..., 0]

        # Get confidence
        confidence = self.get_confidence(attention_output).item()

        lemma = f"← {selected_hyp}" if use_reverse.item() else selected_hyp

        return TacticOutput(
            tactic="rw",
            args={"lemma": f"[{lemma}]"},
            confidence=confidence,
            expert_id=self.expert_id,
        )

    def _is_equality(self, term: TermAST) -> bool:
        """Check if term is an equality type."""
        return term.head_constant in ["Eq", "HEq", "="]

    def get_specialization_score(self, proof_state: ProofState) -> float:
        """Score based on equality hypotheses availability."""
        eq_hyps = sum(1 for h in proof_state.hypotheses
                      if self._is_equality(h.type))

        if proof_state.goal.target.head_constant in ["Eq", "=", "HEq"]:
            return 0.8 + min(eq_hyps * 0.05, 0.15)

        if eq_hyps > 0:
            return 0.5 + min(eq_hyps * 0.1, 0.3)

        return 0.1


class InductionExpert(TacticExpert):
    """
    Expert for induction tactics.

    Handles: induction, cases, rcases, obtain
    Specializes in structural induction on inductive types.
    """

    def __init__(self, config: Optional[ExpertConfig] = None):
        if config is None:
            config = ExpertConfig(
                expert_id="induction",
                tactic_name="induction",
            )
        super().__init__(config)

        # Induction target selector
        self.target_selector = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 10),  # Max 10 candidates
        )

        # Motive generator
        self.motive_encoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

        # Use cases vs induction
        self.tactic_selector = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 4),
            nn.GELU(),
            nn.Linear(config.d_model // 4, 2),  # induction vs cases
        )

    def generate_tactic(
        self,
        proof_state: ProofState,
        attention_output: Tensor,
    ) -> TacticOutput:
        """Generate induction tactic with target selection."""
        # Find inductive hypotheses
        ind_hyps = [h for h in proof_state.hypotheses
                    if h.type.has_inductive_type()]

        # Also check goal for inductive structure
        target_has_inductive = proof_state.goal.target.has_inductive_type()

        if not ind_hyps and not target_has_inductive:
            # No good induction target
            return TacticOutput(
                tactic="sorry",  # Placeholder
                args={},
                confidence=0.1,
                expert_id=self.expert_id,
            )

        # Select target
        if ind_hyps:
            target = ind_hyps[0].name
        else:
            target = "this"  # Induction on goal structure

        # Choose between induction and cases
        tactic_probs = self.tactic_selector(attention_output)
        use_induction = tactic_probs[..., 0] > tactic_probs[..., 1]

        tactic = "induction" if use_induction.item() else "cases"

        confidence = self.get_confidence(attention_output).item()

        return TacticOutput(
            tactic=tactic,
            args={"target": target},
            confidence=confidence,
            expert_id=self.expert_id,
        )

    def get_specialization_score(self, proof_state: ProofState) -> float:
        """Score based on inductive types in scope."""
        n_inductive = sum(1 for h in proof_state.hypotheses
                          if h.type.has_inductive_type())

        if proof_state.goal.target.has_inductive_type():
            return 0.85 + min(n_inductive * 0.03, 0.1)

        if n_inductive > 0:
            return 0.6 + min(n_inductive * 0.1, 0.25)

        return 0.1


class SimpExpert(TacticExpert):
    """
    Expert for simplification tactics.

    Handles: simp, simp_all, dsimp, norm_num
    Specializes in automated simplification and normalization.
    """

    def __init__(self, config: Optional[ExpertConfig] = None):
        if config is None:
            config = ExpertConfig(
                expert_id="simp",
                tactic_name="simp",
            )
        super().__init__(config)

        # Variant selector
        self.variant_selector = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 4),  # simp, dsimp, norm_num, simp_all
        )

        # Lemma inclusion head
        self.lemma_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 4),
            nn.GELU(),
            nn.Linear(config.d_model // 4, 1),
            nn.Sigmoid(),
        )

    def generate_tactic(
        self,
        proof_state: ProofState,
        attention_output: Tensor,
    ) -> TacticOutput:
        """Generate simplification tactic."""
        variants = ["simp", "dsimp", "norm_num", "simp_all"]

        # Select variant
        variant_probs = self.variant_selector(attention_output)
        variant_idx = variant_probs.argmax(dim=-1).item()
        tactic = variants[variant_idx]

        # Check if we should include specific lemmas
        include_lemmas = self.lemma_head(attention_output).item() > 0.5

        args = {}
        if include_lemmas:
            # In full implementation, would select relevant lemmas
            args["extra"] = "only"

        confidence = self.get_confidence(attention_output).item()

        return TacticOutput(
            tactic=tactic,
            args=args,
            confidence=confidence,
            expert_id=self.expert_id,
        )

    def get_specialization_score(self, proof_state: ProofState) -> float:
        """Simp is a general-purpose fallback with moderate score."""
        # Higher score for numeric goals
        if "Nat" in str(proof_state.goal.target.head_constant):
            return 0.7
        if "Int" in str(proof_state.goal.target.head_constant):
            return 0.7
        return 0.4  # Always somewhat applicable


# Expert registry
EXPERT_REGISTRY: Dict[str, type] = {
    "intro": IntroExpert,
    "rewrite": RewriteExpert,
    "induction": InductionExpert,
    "simp": SimpExpert,
}


def create_expert(expert_type: str, config: Optional[ExpertConfig] = None) -> TacticExpert:
    """Create an expert by type name."""
    if expert_type not in EXPERT_REGISTRY:
        raise ValueError(f"Unknown expert type: {expert_type}")
    return EXPERT_REGISTRY[expert_type](config)


def create_all_experts(d_model: int = 1024) -> nn.ModuleList:
    """Create all registered experts."""
    experts = []
    for name, expert_class in EXPERT_REGISTRY.items():
        config = ExpertConfig(
            expert_id=name,
            tactic_name=name,
            d_model=d_model,
        )
        experts.append(expert_class(config))
    return nn.ModuleList(experts)
