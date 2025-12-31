# Tactic-Mesh: Decentralized Mixture-of-Tactics for Formal Theorem Proving

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Lean 4](https://img.shields.io/badge/Lean-4-orange.svg)](https://leanprover.github.io/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)

> **A revolutionary fusion of DeepSeek-V3's Mixture-of-Experts architecture with formal theorem proving, running on a decentralized P2P network.**

## Vision

Tactic-Mesh reimagines theorem proving by treating mathematical tactics as "experts" in a Mixture-of-Experts architecture. Instead of routing to language model experts, we route proof states to specialized **tactic experts**—each mastering a different proof strategy (induction, rewriting, case analysis, etc.).

The system runs on a decentralized network where:
- **Proofs are distributed** across nodes like BitTorrent for mathematics
- **Verification is guaranteed** by Lean 4's type checker
- **Learning is continuous** from every proof attempt
- **Anyone can contribute** compute, conjectures, or proofs

## Key Innovations

### 1. Mixture-of-Tactics Architecture (MoTA)

Unlike traditional LLM-based provers that generate tactic strings, MoTA treats each tactic as a specialized expert:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Proof State Encoder                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │ Goal    │  │ Hyps    │  │ Types   │  │ Context │            │
│  │ Encoder │  │ Encoder │  │ Encoder │  │ Graph   │            │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘            │
│       └──────────┬─┴───────────┴──────────────┘                │
│                  ▼                                              │
│         ┌───────────────┐                                       │
│         │ Proof-State   │                                       │
│         │ Latent        │  ◄── Adapted from DeepSeek's MLA     │
│         │ Attention     │                                       │
│         └───────┬───────┘                                       │
│                 ▼                                                │
│    ┌────────────────────────┐                                   │
│    │   Tactic Router        │  ◄── Auxiliary-loss-free         │
│    │   (Sparse Gating)      │      load balancing              │
│    └────────────┬───────────┘                                   │
│                 ▼                                                │
│  ┌──────┬──────┬──────┬──────┬──────┬──────┐                   │
│  │Intro │Rewrite│Induct│Cases │Simp  │Auto  │  ◄── Tactic      │
│  │Expert│Expert │Expert│Expert│Expert│Expert│      Experts     │
│  └──┬───┴──┬───┴──┬───┴──┬───┴──┬───┴──┬───┘                   │
│     └──────┴──────┴──────┴──────┴──────┘                        │
│                        ▼                                         │
│              ┌─────────────────┐                                 │
│              │ Lean 4 Verifier │  ◄── Formal Guarantee          │
│              └─────────────────┘                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Proof-State Latent Attention (PSLA)

Adapts DeepSeek's Multi-head Latent Attention for proof states:

- **Hierarchical encoding**: AST → Type Graph → Proof State
- **De Bruijn indexing**: Position-invariant term representation
- **Structural canonicalization**: Commutative sorting for reuse
- **Compact routing features**: Goal head, complexity metrics, hypothesis patterns

### 3. Distributed Proof Search Protocol (DPSP)

A P2P protocol for collaborative theorem proving:

```
┌─────────────────────────────────────────────────────────────────┐
│                    DISTRIBUTED PROOF MESH                        │
│                                                                  │
│     ┌─────┐         ┌─────┐         ┌─────┐                     │
│     │Node │◄───────►│Node │◄───────►│Node │                     │
│     │  A  │         │  B  │         │  C  │                     │
│     └──┬──┘         └──┬──┘         └──┬──┘                     │
│        │               │               │                         │
│        ▼               ▼               ▼                         │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐                     │
│   │Induction│    │Rewriting│    │ Simp    │  ◄── Expert Shards  │
│   │ Expert  │    │ Expert  │    │ Expert  │                     │
│   └─────────┘    └─────────┘    └─────────┘                     │
│                                                                  │
│   ┌───────────────────────────────────────────────────────────┐ │
│   │              Content-Addressed Proof Cache                 │ │
│   │   goal_hash → (tactic, subgoals, lean_certificate)        │ │
│   └───────────────────────────────────────────────────────────┘ │
│                                                                  │
│   ┌───────────────────────────────────────────────────────────┐ │
│   │                 Proof Ledger (BFT Consensus)               │ │
│   │   Append-only log of verified proof steps                  │ │
│   └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 4. Self-Amplifying Learning

The system improves from every proof attempt:

- **Hierarchical RL**: Step-level + episode-level rewards
- **Proof tree credit assignment**: Backpropagate through proof DAG
- **Hindsight experience replay**: Learn from partial progress
- **Federated expert updates**: Local training with global distillation

## Architecture

```
tactic-mesh/
├── core/                    # Core abstractions
│   ├── proof_state.py       # Proof state representation
│   ├── tactic.py            # Tactic interface
│   └── router.py            # MoE routing logic
├── experts/                 # Tactic experts
│   ├── intro.py             # Introduction tactics
│   ├── rewrite.py           # Rewriting tactics
│   ├── induction.py         # Induction tactics
│   ├── cases.py             # Case analysis
│   ├── simp.py              # Simplification
│   └── auto.py              # Automation
├── attention/               # PSLA implementation
│   ├── proof_encoder.py     # Proof state encoding
│   ├── latent_attention.py  # Multi-head latent attention
│   └── graph_encoder.py     # GNN for proof graphs
├── network/                 # P2P infrastructure
│   ├── dht.py               # Distributed hash table
│   ├── proof_cache.py       # Content-addressed caching
│   ├── peer_router.py       # Expert-aware routing
│   └── consensus.py         # BFT for proof ledger
├── lean/                    # Lean 4 integration
│   ├── kernel.py            # Lean kernel interface
│   ├── verifier.py          # Formal verification
│   ├── soft_kernel.py       # Differentiable surrogate
│   └── tactics/             # Lean tactic bindings
├── training/                # Learning infrastructure
│   ├── rl.py                # Hierarchical RL
│   ├── credit.py            # Proof tree credit assignment
│   ├── federated.py         # Federated learning
│   └── distillation.py      # Knowledge distillation
└── cli/                     # Command-line interface
    ├── prove.py             # Prove conjectures
    ├── node.py              # Run network node
    └── train.py             # Training scripts
```

## Technical Deep Dive

### Proof State Encoding

```python
@dataclass
class ProofState:
    """Encoded proof state for attention mechanism."""

    # Goal representation (de Bruijn indexed)
    goal: TermAST
    goal_embedding: Tensor  # [seq_len, d_model]

    # Local hypotheses with types
    hypotheses: List[Hypothesis]
    hyp_embeddings: Tensor  # [n_hyps, d_model]

    # Proof graph (goals as nodes, tactics as edges)
    proof_graph: ProofDAG
    graph_embedding: Tensor  # [n_nodes, d_model]

    # Compact routing features
    routing_features: RoutingFeatures  # [d_route]

@dataclass
class RoutingFeatures:
    """Lightweight features for fast expert routing."""
    goal_head: int           # Head constant ID
    syntactic_complexity: float
    n_hypotheses: int
    n_metavars: int
    has_inductive_motive: bool
    type_class_instances: List[int]
```

### Tactic Expert Interface

```python
class TacticExpert(nn.Module):
    """Base class for tactic experts."""

    def __init__(self, expert_id: str, d_model: int):
        super().__init__()
        self.expert_id = expert_id
        self.tactic_head = nn.Linear(d_model, self.output_dim)

    @abstractmethod
    def forward(
        self,
        proof_state: ProofState,
        attention_output: Tensor
    ) -> TacticOutput:
        """Generate tactic application."""
        pass

    @abstractmethod
    def specialize(self, lean_tactic: str) -> None:
        """Specialize to a particular Lean tactic."""
        pass

class InductionExpert(TacticExpert):
    """Expert for induction tactics."""

    def forward(self, proof_state: ProofState, attn: Tensor) -> TacticOutput:
        # Identify inductive types in goal
        inductives = self.find_inductives(proof_state.goal)

        # Select induction target and motive
        target = self.select_target(inductives, attn)
        motive = self.generate_motive(proof_state, target, attn)

        return TacticOutput(
            tactic="induction",
            args={"target": target, "motive": motive},
            confidence=self.confidence_head(attn)
        )
```

### Distributed Proof Protocol

```python
class ProofMeshNode:
    """Node in the distributed proof network."""

    def __init__(self, config: NodeConfig):
        self.dht = KademliaDHT(config.bootstrap_peers)
        self.proof_cache = ContentAddressedCache()
        self.local_experts = self.load_experts(config.expert_shards)
        self.router = RouterReplica()
        self.verifier = LeanVerifier()

    async def prove(self, goal: Goal) -> Optional[Proof]:
        """Attempt to prove a goal using the mesh."""

        # Check cache first
        cached = await self.proof_cache.get(goal.hash())
        if cached and self.verifier.verify(cached):
            return cached

        # Encode proof state
        state = self.encode_proof_state(goal)

        # Route to top-k experts
        expert_scores = self.router(state.routing_features)
        top_experts = expert_scores.topk(k=3)

        # Query experts (local or remote)
        tactic_outputs = await self.query_experts(top_experts, state)

        # Try each tactic, verify with Lean
        for output in tactic_outputs:
            result = await self.verifier.apply_tactic(goal, output)
            if result.success:
                # Recursively prove subgoals
                subproofs = await asyncio.gather(*[
                    self.prove(subgoal) for subgoal in result.subgoals
                ])
                if all(subproofs):
                    proof = Proof(goal, output, subproofs)
                    await self.proof_cache.store(proof)
                    await self.broadcast_proof(proof)
                    return proof

        return None

    async def query_experts(
        self,
        experts: List[ExpertId],
        state: ProofState
    ) -> List[TacticOutput]:
        """Query experts, routing to remote peers if needed."""
        outputs = []
        for expert_id in experts:
            if expert_id in self.local_experts:
                output = self.local_experts[expert_id](state)
            else:
                # Find peer hosting this expert
                peer = await self.dht.find_expert_peer(expert_id)
                output = await peer.query_expert(expert_id, state)
            outputs.append(output)
        return outputs
```

## Inspired By

### DeepSeek-V3
- **Multi-head Latent Attention (MLA)** → Proof-State Latent Attention
- **Auxiliary-loss-free load balancing** → Tactic expert balancing
- **Multi-Token Prediction** → Multi-step proof sketching
- **FP8 training** → Efficient expert training
- **Knowledge distillation from CoT** → Proof distillation

### Equational Theories (Terry Tao)
- **4694 equations, 22M implications** → Training/evaluation corpus
- **Lean 4 formalization** → Verification backend
- **Collaborative methodology** → Distributed proof network
- **Implication graph** → Proof state graph structure
- **ATP integration** → Symbolic expert backends

## Roadmap

### Phase 1: Foundation (Q1 2025)
- [ ] Proof state encoding and PSLA implementation
- [ ] Basic tactic experts (intro, rewrite, simp)
- [ ] Lean 4 kernel integration
- [ ] Single-node proof search

### Phase 2: Scaling (Q2 2025)
- [ ] Full MoE architecture with all tactic experts
- [ ] Hierarchical RL training pipeline
- [ ] Soft kernel for differentiable training
- [ ] Evaluation on equational theories

### Phase 3: Distribution (Q3 2025)
- [ ] P2P network infrastructure
- [ ] Content-addressed proof cache
- [ ] Federated expert training
- [ ] BFT proof ledger

### Phase 4: Production (Q4 2025)
- [ ] Public proof mesh network
- [ ] Integration with Mathlib4
- [ ] Continuous learning pipeline
- [ ] Community contribution tools

## Getting Started

```bash
# Clone the repository
git clone https://github.com/yurekami/tactic-mesh.git
cd tactic-mesh

# Install dependencies
pip install -e ".[dev]"

# Install Lean 4 (via elan)
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh

# Run a simple proof
tactic-mesh prove "∀ n : Nat, n + 0 = n"

# Start a network node
tactic-mesh node --port 31337 --experts induction,rewrite

# Train on equational theories
tactic-mesh train --dataset equational_theories --epochs 100
```

## Why This Matters

1. **Scales theorem proving horizontally**: Like MapReduce for mathematics
2. **Guarantees correctness**: Every output verified by Lean's kernel
3. **Learns continuously**: Improves from every proof attempt across the network
4. **Democratizes mathematics**: Anyone can contribute compute or proofs
5. **Bridges neural and symbolic**: Best of both worlds

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

```bibtex
@software{tactic_mesh_2025,
  title = {Tactic-Mesh: Decentralized Mixture-of-Tactics for Formal Theorem Proving},
  author = {yurekami},
  year = {2025},
  url = {https://github.com/yurekami/tactic-mesh}
}
```

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

---

*"The best way to predict the future of mathematics is to prove it."*
