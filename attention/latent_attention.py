"""
Proof-State Latent Attention (PSLA).

Adapts DeepSeek-V3's Multi-head Latent Attention (MLA) for proof states.

Key innovations:
- Latent compression of proof states for efficiency
- Hierarchical attention over goals, hypotheses, and proof graph
- De Bruijn-aware positional encoding
- Structural canonicalization for better generalization
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from core.proof_state import ProofState, TermAST, TermKind


@dataclass
class PSLAConfig:
    """Configuration for Proof-State Latent Attention."""
    d_model: int = 1024           # Model dimension
    n_heads: int = 16             # Number of attention heads
    d_latent: int = 256           # Latent dimension for compression
    d_rope: int = 64              # RoPE dimension
    max_seq_len: int = 4096       # Maximum sequence length
    dropout: float = 0.1         # Dropout rate

    # Proof-specific settings
    use_graph_attention: bool = True
    use_structural_encoding: bool = True
    canonicalize_terms: bool = True


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) for position-aware attention.

    Adapted for proof states where positions are structural (AST depth)
    rather than sequential.
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute rotation frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos/sin for efficiency
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def forward(self, x: Tensor, positions: Optional[Tensor] = None) -> Tensor:
        """Apply rotary embedding to input tensor."""
        seq_len = x.shape[1]

        if positions is None:
            positions = torch.arange(seq_len, device=x.device)

        cos = self.cos_cached[positions]  # [seq_len, dim/2]
        sin = self.sin_cached[positions]

        # Apply rotation
        x1, x2 = x[..., : self.dim // 2], x[..., self.dim // 2 : self.dim]
        x_rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ], dim=-1)

        # Concat with unrotated dimensions if any
        if x.shape[-1] > self.dim:
            x_rotated = torch.cat([x_rotated, x[..., self.dim:]], dim=-1)

        return x_rotated


class LatentCompression(nn.Module):
    """
    Latent compression layer for efficient attention.

    Compresses proof state representations to a lower-dimensional
    latent space, reducing memory and compute for attention.
    """

    def __init__(self, d_model: int, d_latent: int):
        super().__init__()
        self.d_model = d_model
        self.d_latent = d_latent

        # Down-projection to latent space
        self.down_proj = nn.Linear(d_model, d_latent, bias=False)

        # Up-projection back to model dimension
        self.up_proj = nn.Linear(d_latent, d_model, bias=False)

        # Layer norm for stability
        self.norm = nn.LayerNorm(d_latent)

    def compress(self, x: Tensor) -> Tensor:
        """Compress to latent space."""
        return self.norm(self.down_proj(x))

    def decompress(self, z: Tensor) -> Tensor:
        """Decompress from latent space."""
        return self.up_proj(z)


class ProofStateLatentAttention(nn.Module):
    """
    Proof-State Latent Attention (PSLA).

    This is the core attention mechanism for Tactic-Mesh, adapted from
    DeepSeek's Multi-head Latent Attention for proof states.

    Key features:
    - Latent compression for efficiency
    - Hierarchical attention (goal → hypotheses → graph)
    - Structural positional encoding
    - De Bruijn-aware representation
    """

    def __init__(self, config: PSLAConfig):
        super().__init__()
        self.config = config

        # Dimensions
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.d_latent = config.d_latent
        self.d_rope = config.d_rope

        # Latent compression
        self.kv_compression = LatentCompression(config.d_model, config.d_latent)

        # Query projection (not compressed)
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # Key-value projections from latent space
        self.kv_proj = nn.Linear(config.d_latent, 2 * config.d_model, bias=False)

        # Output projection
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # Rotary embeddings for queries and keys
        self.rope = RotaryPositionalEmbedding(config.d_rope, config.max_seq_len)

        # Structural encoding for proof terms
        if config.use_structural_encoding:
            self.struct_embed = StructuralEncoding(config.d_model)

        # Layer norms
        self.q_norm = nn.LayerNorm(self.d_head)
        self.k_norm = nn.LayerNorm(self.d_head)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.d_head)

    def forward(
        self,
        x: Tensor,
        kv_cache: Optional[Tensor] = None,
        positions: Optional[Tensor] = None,
        structural_positions: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass for PSLA.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            kv_cache: Cached compressed KV for incremental decoding
            positions: Token positions for RoPE
            structural_positions: AST depth positions for structural encoding
            attention_mask: Attention mask [batch, seq_len, seq_len]

        Returns:
            output: Attention output [batch, seq_len, d_model]
            new_kv_cache: Updated KV cache
        """
        batch_size, seq_len, _ = x.shape

        # Apply structural encoding if enabled
        if self.config.use_structural_encoding and structural_positions is not None:
            x = x + self.struct_embed(structural_positions)

        # Compute queries (full dimension)
        q = self.q_proj(x)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head)
        q = q.transpose(1, 2)  # [batch, heads, seq, d_head]

        # Compress KV to latent space
        kv_latent = self.kv_compression.compress(x)

        # Update cache if provided
        if kv_cache is not None:
            kv_latent = torch.cat([kv_cache, kv_latent], dim=1)

        # Project from latent to full KV
        kv = self.kv_proj(kv_latent)
        k, v = kv.chunk(2, dim=-1)

        kv_seq_len = k.shape[1]
        k = k.view(batch_size, kv_seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, kv_seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE to queries and keys
        q_rope = self.rope(q[..., :self.d_rope], positions)
        k_rope = self.rope(k[..., :self.d_rope], positions)

        # Reconstruct Q and K with rotated positions
        q = torch.cat([q_rope, q[..., self.d_rope:]], dim=-1)
        k = torch.cat([k_rope, k[..., self.d_rope:]], dim=-1)

        # Normalize Q and K (from DeepSeek)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(
                attention_mask == 0,
                float("-inf")
            )

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute output
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)
        output = self.o_proj(output)

        # Return new KV cache
        new_cache = kv_latent if kv_cache is not None else None

        return output, new_cache


class StructuralEncoding(nn.Module):
    """
    Structural encoding for proof terms.

    Encodes the structure of terms (AST depth, binder depth, etc.)
    rather than sequential position.
    """

    def __init__(self, d_model: int, max_depth: int = 64):
        super().__init__()
        self.d_model = d_model
        self.max_depth = max_depth

        # Embeddings for different structural features
        self.depth_embed = nn.Embedding(max_depth, d_model)
        self.binder_embed = nn.Embedding(max_depth, d_model)
        self.kind_embed = nn.Embedding(len(TermKind), d_model)

        # Combine embeddings
        self.combine = nn.Linear(3 * d_model, d_model)

    def forward(self, structural_positions: Tensor) -> Tensor:
        """
        Compute structural encoding.

        Args:
            structural_positions: [batch, seq_len, 3]
                - [:, :, 0]: AST depth
                - [:, :, 1]: Binder depth (de Bruijn)
                - [:, :, 2]: Term kind

        Returns:
            Structural encoding [batch, seq_len, d_model]
        """
        depth = self.depth_embed(structural_positions[..., 0].clamp(0, self.max_depth - 1))
        binder = self.binder_embed(structural_positions[..., 1].clamp(0, self.max_depth - 1))
        kind = self.kind_embed(structural_positions[..., 2])

        combined = torch.cat([depth, binder, kind], dim=-1)
        return self.combine(combined)


class HierarchicalProofAttention(nn.Module):
    """
    Hierarchical attention over proof state components.

    Applies attention at three levels:
    1. Goal-level: Attention over the target term
    2. Hypothesis-level: Cross-attention from goal to hypotheses
    3. Graph-level: Attention over the proof DAG structure
    """

    def __init__(self, config: PSLAConfig):
        super().__init__()
        self.config = config

        # Goal-level attention
        self.goal_attention = ProofStateLatentAttention(config)

        # Hypothesis cross-attention
        self.hyp_cross_attention = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # Graph attention (if enabled)
        if config.use_graph_attention:
            self.graph_attention = GraphAttentionLayer(
                config.d_model,
                config.n_heads,
            )

        # Final combination
        self.combine = nn.Sequential(
            nn.Linear(config.d_model * 3 if config.use_graph_attention else config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        goal_embedding: Tensor,
        hyp_embeddings: Tensor,
        graph_embedding: Optional[Tensor] = None,
        structural_positions: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Hierarchical attention over proof state.

        Args:
            goal_embedding: [batch, goal_len, d_model]
            hyp_embeddings: [batch, n_hyps, d_model]
            graph_embedding: [batch, n_nodes, d_model] (optional)
            structural_positions: Structural positions for goal

        Returns:
            Combined proof state representation [batch, d_model]
        """
        # Goal-level attention
        goal_attn, _ = self.goal_attention(
            goal_embedding,
            structural_positions=structural_positions,
        )
        goal_repr = goal_attn.mean(dim=1)  # Pool over sequence

        # Hypothesis cross-attention
        # Query: goal, Key/Value: hypotheses
        goal_query = goal_repr.unsqueeze(1)  # [batch, 1, d_model]
        hyp_attn, _ = self.hyp_cross_attention(
            goal_query,
            hyp_embeddings,
            hyp_embeddings,
        )
        hyp_repr = hyp_attn.squeeze(1)  # [batch, d_model]

        # Graph attention (if available)
        if self.config.use_graph_attention and graph_embedding is not None:
            graph_repr = self.graph_attention(graph_embedding)
            combined = torch.cat([goal_repr, hyp_repr, graph_repr], dim=-1)
        else:
            combined = torch.cat([goal_repr, hyp_repr], dim=-1)

        # Combine all representations
        output = self.combine(combined)
        output = self.norm(output)

        return output


class GraphAttentionLayer(nn.Module):
    """
    Graph attention for proof DAG structure.

    Attends over the proof graph where nodes are goals/subgoals
    and edges are tactic applications.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        # Edge type embeddings
        self.edge_embed = nn.Embedding(100, d_model)  # 100 tactic types

        self.scale = 1.0 / math.sqrt(self.d_head)

    def forward(
        self,
        node_embeddings: Tensor,
        edge_types: Optional[Tensor] = None,
        adjacency: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Graph attention over proof nodes.

        Args:
            node_embeddings: [batch, n_nodes, d_model]
            edge_types: [batch, n_nodes, n_nodes] - tactic type IDs
            adjacency: [batch, n_nodes, n_nodes] - adjacency matrix

        Returns:
            Pooled graph representation [batch, d_model]
        """
        batch_size, n_nodes, _ = node_embeddings.shape

        q = self.q_proj(node_embeddings).view(batch_size, n_nodes, self.n_heads, self.d_head)
        k = self.k_proj(node_embeddings).view(batch_size, n_nodes, self.n_heads, self.d_head)
        v = self.v_proj(node_embeddings).view(batch_size, n_nodes, self.n_heads, self.d_head)

        # Compute attention
        q = q.transpose(1, 2)  # [batch, heads, nodes, d_head]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply adjacency mask if provided
        if adjacency is not None:
            attn = attn.masked_fill(adjacency.unsqueeze(1) == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, n_nodes, self.d_model)
        output = self.o_proj(output)

        # Global pooling
        return output.mean(dim=1)
