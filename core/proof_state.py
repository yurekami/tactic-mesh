"""
Proof State Representation for Tactic-Mesh.

This module defines the core data structures for representing proof states
in a way that's amenable to neural network processing and attention mechanisms.

Key innovations:
- De Bruijn indexed terms for position-invariance
- Hierarchical encoding (AST → Type Graph → Proof State)
- Compact routing features for fast expert selection
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor


class TermKind(Enum):
    """Kinds of terms in the proof state."""
    VAR = auto()        # Bound variable (de Bruijn index)
    CONST = auto()      # Constant/definition
    APP = auto()        # Application
    LAM = auto()        # Lambda abstraction
    PI = auto()         # Pi/forall type
    SORT = auto()       # Universe/Type
    LET = auto()        # Let binding
    META = auto()       # Metavariable (hole)
    LITERAL = auto()    # Literal values


@dataclass
class TermAST:
    """
    Abstract Syntax Tree for terms.

    Uses de Bruijn indices for bound variables to ensure
    alpha-equivalence and position-invariance.
    """
    kind: TermKind
    data: Any = None
    children: List[TermAST] = field(default_factory=list)
    type_annotation: Optional[TermAST] = None

    # Cached properties
    _hash: Optional[str] = field(default=None, repr=False)
    _complexity: Optional[int] = field(default=None, repr=False)

    @property
    def hash(self) -> str:
        """Content-addressable hash of the term."""
        if self._hash is None:
            content = f"{self.kind.name}:{self.data}:"
            content += ":".join(c.hash for c in self.children)
            self._hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        return self._hash

    @property
    def complexity(self) -> int:
        """Syntactic complexity (node count)."""
        if self._complexity is None:
            self._complexity = 1 + sum(c.complexity for c in self.children)
        return self._complexity

    @property
    def head_constant(self) -> Optional[str]:
        """Extract the head constant (for routing)."""
        if self.kind == TermKind.CONST:
            return self.data
        elif self.kind == TermKind.APP and self.children:
            return self.children[0].head_constant
        return None

    @property
    def n_metavars(self) -> int:
        """Count metavariables (unresolved holes)."""
        count = 1 if self.kind == TermKind.META else 0
        return count + sum(c.n_metavars for c in self.children)

    def has_inductive_type(self) -> bool:
        """Check if term involves inductive types."""
        # In a real implementation, this would check against Lean's environment
        inductive_indicators = ["Nat", "List", "Bool", "Fin", "Vector"]
        if self.kind == TermKind.CONST and any(
            ind in str(self.data) for ind in inductive_indicators
        ):
            return True
        return any(c.has_inductive_type() for c in self.children)

    def to_tokens(self, vocab: Dict[str, int]) -> List[int]:
        """Linearize to token sequence for embedding."""
        tokens = []

        # Add kind token
        tokens.append(vocab.get(f"KIND_{self.kind.name}", vocab["<unk>"]))

        # Add data token if applicable
        if self.data is not None:
            tokens.append(vocab.get(str(self.data), vocab["<unk>"]))

        # Recursively add children with structure tokens
        for i, child in enumerate(self.children):
            tokens.append(vocab["<child>"])
            tokens.extend(child.to_tokens(vocab))
            tokens.append(vocab["</child>"])

        return tokens


@dataclass
class Hypothesis:
    """A hypothesis in the local context."""
    name: str
    type: TermAST
    value: Optional[TermAST] = None  # For let-bindings
    is_instance: bool = False        # Type class instance

    @property
    def hash(self) -> str:
        """Hash based on type (name-independent)."""
        return self.type.hash


@dataclass
class Goal:
    """A proof goal to be solved."""
    target: TermAST
    hypotheses: List[Hypothesis] = field(default_factory=list)
    tag: Optional[str] = None  # User-provided name

    @property
    def hash(self) -> str:
        """Content-addressable hash of the goal."""
        content = self.target.hash + ":" + ":".join(h.hash for h in self.hypotheses)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class ProofNode:
    """Node in the proof DAG."""
    goal: Goal
    tactic: Optional[str] = None
    children: List[ProofNode] = field(default_factory=list)
    status: str = "open"  # open, solved, failed

    @property
    def is_solved(self) -> bool:
        return self.status == "solved" or (
            self.tactic is not None and
            all(c.is_solved for c in self.children)
        )


@dataclass
class ProofDAG:
    """Directed Acyclic Graph representing proof structure."""
    root: ProofNode
    nodes: Dict[str, ProofNode] = field(default_factory=dict)
    edges: List[Tuple[str, str, str]] = field(default_factory=list)  # (from, to, tactic)

    def add_node(self, node: ProofNode) -> None:
        """Add a node to the DAG."""
        self.nodes[node.goal.hash] = node

    def add_edge(self, from_hash: str, to_hash: str, tactic: str) -> None:
        """Add an edge (tactic application) to the DAG."""
        self.edges.append((from_hash, to_hash, tactic))

    @property
    def n_open_goals(self) -> int:
        """Count open (unsolved) goals."""
        return sum(1 for n in self.nodes.values() if n.status == "open")

    @property
    def depth(self) -> int:
        """Maximum depth of the proof tree."""
        def _depth(node: ProofNode) -> int:
            if not node.children:
                return 1
            return 1 + max(_depth(c) for c in node.children)
        return _depth(self.root)


@dataclass
class RoutingFeatures:
    """
    Compact features for fast expert routing.

    These are extracted from the proof state and used by the
    MoE router to select appropriate tactic experts.
    """
    # Goal characteristics
    goal_head_id: int           # Head constant vocabulary ID
    goal_complexity: int        # Syntactic complexity
    goal_depth: int             # AST depth
    n_metavars: int             # Number of holes

    # Hypothesis characteristics
    n_hypotheses: int           # Total hypotheses
    n_eq_hypotheses: int        # Equality hypotheses (for rewriting)
    n_inductive_hypotheses: int # Hypotheses with inductive types
    n_function_hypotheses: int  # Function-typed hypotheses

    # Type characteristics
    has_inductive_target: bool  # Target is inductive type
    has_equality_target: bool   # Target is equality
    has_exists_target: bool     # Target is existential
    has_forall_target: bool     # Target starts with forall

    # Proof state characteristics
    proof_depth: int            # Current depth in proof tree
    n_open_goals: int           # Sibling open goals
    n_solved_goals: int         # Already solved goals

    def to_tensor(self) -> Tensor:
        """Convert to tensor for router input."""
        return torch.tensor([
            self.goal_head_id,
            self.goal_complexity,
            self.goal_depth,
            self.n_metavars,
            self.n_hypotheses,
            self.n_eq_hypotheses,
            self.n_inductive_hypotheses,
            self.n_function_hypotheses,
            float(self.has_inductive_target),
            float(self.has_equality_target),
            float(self.has_exists_target),
            float(self.has_forall_target),
            self.proof_depth,
            self.n_open_goals,
            self.n_solved_goals,
        ], dtype=torch.float32)


@dataclass
class ProofState:
    """
    Complete proof state for attention mechanism.

    This is the primary input to the Tactic-Mesh model,
    encoding everything needed to select and apply tactics.
    """
    # Current goal
    goal: Goal
    goal_embedding: Optional[Tensor] = None  # [seq_len, d_model]

    # Local context
    hypotheses: List[Hypothesis] = field(default_factory=list)
    hyp_embeddings: Optional[Tensor] = None  # [n_hyps, d_model]

    # Proof structure
    proof_dag: Optional[ProofDAG] = None
    graph_embedding: Optional[Tensor] = None  # [n_nodes, d_model]

    # Routing features
    routing_features: Optional[RoutingFeatures] = None

    # Global context (for cross-goal attention)
    sibling_goals: List[Goal] = field(default_factory=list)
    parent_tactics: List[str] = field(default_factory=list)

    @classmethod
    def from_goal(cls, goal: Goal, proof_dag: Optional[ProofDAG] = None) -> ProofState:
        """Create proof state from a goal."""
        state = cls(
            goal=goal,
            hypotheses=goal.hypotheses,
            proof_dag=proof_dag,
        )
        state.routing_features = state._extract_routing_features()
        return state

    def _extract_routing_features(self) -> RoutingFeatures:
        """Extract compact routing features from the proof state."""
        target = self.goal.target

        # Count hypothesis types
        n_eq = sum(1 for h in self.hypotheses if self._is_equality(h.type))
        n_ind = sum(1 for h in self.hypotheses if h.type.has_inductive_type())
        n_fun = sum(1 for h in self.hypotheses if self._is_function(h.type))

        return RoutingFeatures(
            goal_head_id=hash(target.head_constant or "") % 10000,
            goal_complexity=target.complexity,
            goal_depth=self._ast_depth(target),
            n_metavars=target.n_metavars,
            n_hypotheses=len(self.hypotheses),
            n_eq_hypotheses=n_eq,
            n_inductive_hypotheses=n_ind,
            n_function_hypotheses=n_fun,
            has_inductive_target=target.has_inductive_type(),
            has_equality_target=self._is_equality(target),
            has_exists_target=self._is_exists(target),
            has_forall_target=target.kind == TermKind.PI,
            proof_depth=self.proof_dag.depth if self.proof_dag else 0,
            n_open_goals=self.proof_dag.n_open_goals if self.proof_dag else 1,
            n_solved_goals=len(self.proof_dag.nodes) - self.proof_dag.n_open_goals if self.proof_dag else 0,
        )

    @staticmethod
    def _is_equality(term: TermAST) -> bool:
        """Check if term is an equality type."""
        return term.head_constant in ["Eq", "HEq", "="]

    @staticmethod
    def _is_exists(term: TermAST) -> bool:
        """Check if term is an existential."""
        return term.head_constant in ["Exists", "∃", "Sigma"]

    @staticmethod
    def _is_function(term: TermAST) -> bool:
        """Check if term is a function type."""
        return term.kind == TermKind.PI

    @staticmethod
    def _ast_depth(term: TermAST) -> int:
        """Calculate AST depth."""
        if not term.children:
            return 1
        return 1 + max(ProofState._ast_depth(c) for c in term.children)

    @property
    def hash(self) -> str:
        """Content-addressable hash of the proof state."""
        return self.goal.hash


@dataclass
class TacticOutput:
    """Output from a tactic expert."""
    tactic: str                          # Tactic name
    args: Dict[str, Any] = field(default_factory=dict)  # Arguments
    confidence: float = 0.0              # Expert confidence
    expert_id: str = ""                  # Which expert generated this

    def to_lean(self) -> str:
        """Convert to Lean 4 tactic syntax."""
        if not self.args:
            return self.tactic

        args_str = " ".join(str(v) for v in self.args.values())
        return f"{self.tactic} {args_str}"


@dataclass
class Proof:
    """A complete or partial proof."""
    goal: Goal
    tactic: TacticOutput
    subproofs: List[Proof] = field(default_factory=list)
    verified: bool = False
    lean_certificate: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        """Check if proof is complete (no open goals)."""
        return self.verified or all(sp.is_complete for sp in self.subproofs)

    @property
    def hash(self) -> str:
        """Content-addressable hash for caching."""
        content = f"{self.goal.hash}:{self.tactic.tactic}:"
        content += ":".join(sp.hash for sp in self.subproofs)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_lean(self) -> str:
        """Convert to Lean 4 proof script."""
        lines = [self.tactic.to_lean()]
        for subproof in self.subproofs:
            sublines = subproof.to_lean().split("\n")
            lines.extend(["  " + line for line in sublines])
        return "\n".join(lines)
