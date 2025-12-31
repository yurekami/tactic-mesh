"""
Content-Addressed Proof Cache.

Implements a distributed cache for verified proofs using content hashing.
Similar to Git's object store or IPFS content addressing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from core.proof_state import Goal, Proof, TacticOutput

logger = logging.getLogger(__name__)


@dataclass
class CachedProof:
    """A cached proof with metadata."""
    proof: Proof
    goal_hash: str
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    verified: bool = False
    lean_certificate: Optional[str] = None

    # Provenance
    source_peer: Optional[str] = None
    computation_time_ms: float = 0.0

    @property
    def content_hash(self) -> str:
        """Content-addressable hash of the proof."""
        return self.proof.hash

    def touch(self) -> None:
        """Update access time and count."""
        self.timestamp = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache statistics."""
    total_proofs: int = 0
    verified_proofs: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_bytes: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class ProofCache:
    """
    Content-addressed cache for proofs.

    Features:
    - Content-addressable storage (goal_hash → proof)
    - LRU eviction policy
    - Optional disk persistence
    - Verification status tracking
    """

    def __init__(
        self,
        max_size: int = 10000,
        max_bytes: int = 100 * 1024 * 1024,  # 100MB
        persist_path: Optional[Path] = None,
    ):
        self.max_size = max_size
        self.max_bytes = max_bytes
        self.persist_path = persist_path

        # Main storage: goal_hash → CachedProof
        self._cache: Dict[str, CachedProof] = {}

        # Secondary index: proof_hash → goal_hash (for deduplication)
        self._proof_index: Dict[str, str] = {}

        # LRU tracking
        self._access_order: List[str] = []

        # Statistics
        self.stats = CacheStats()

        # Load from disk if available
        if persist_path and persist_path.exists():
            self._load_from_disk()

    def get(self, goal: Goal) -> Optional[Proof]:
        """
        Get cached proof for a goal.

        Args:
            goal: The goal to look up

        Returns:
            Cached proof if found, None otherwise
        """
        goal_hash = goal.hash

        if goal_hash in self._cache:
            cached = self._cache[goal_hash]
            cached.touch()
            self._update_lru(goal_hash)
            self.stats.cache_hits += 1
            return cached.proof

        self.stats.cache_misses += 1
        return None

    def get_by_hash(self, goal_hash: str) -> Optional[Proof]:
        """Get cached proof by goal hash."""
        if goal_hash in self._cache:
            cached = self._cache[goal_hash]
            cached.touch()
            self._update_lru(goal_hash)
            self.stats.cache_hits += 1
            return cached.proof

        self.stats.cache_misses += 1
        return None

    def put(
        self,
        goal: Goal,
        proof: Proof,
        verified: bool = False,
        lean_certificate: Optional[str] = None,
        source_peer: Optional[str] = None,
        computation_time_ms: float = 0.0,
    ) -> str:
        """
        Cache a proof.

        Args:
            goal: The goal that was proved
            proof: The proof
            verified: Whether the proof was verified by Lean
            lean_certificate: Lean verification certificate
            source_peer: Peer that computed the proof
            computation_time_ms: Time to compute the proof

        Returns:
            Content hash of the cached proof
        """
        goal_hash = goal.hash
        proof_hash = proof.hash

        # Check for duplicate proof (same proof for different goal representation)
        if proof_hash in self._proof_index:
            existing_goal_hash = self._proof_index[proof_hash]
            if existing_goal_hash != goal_hash:
                logger.debug(f"Duplicate proof detected: {proof_hash}")

        # Create cached entry
        cached = CachedProof(
            proof=proof,
            goal_hash=goal_hash,
            verified=verified,
            lean_certificate=lean_certificate,
            source_peer=source_peer,
            computation_time_ms=computation_time_ms,
        )

        # Evict if necessary
        self._ensure_capacity()

        # Store
        self._cache[goal_hash] = cached
        self._proof_index[proof_hash] = goal_hash
        self._access_order.append(goal_hash)

        # Update stats
        self.stats.total_proofs = len(self._cache)
        if verified:
            self.stats.verified_proofs += 1

        logger.debug(f"Cached proof for goal {goal_hash[:8]}")
        return proof_hash

    def mark_verified(
        self,
        goal_hash: str,
        lean_certificate: Optional[str] = None,
    ) -> bool:
        """Mark a cached proof as verified."""
        if goal_hash not in self._cache:
            return False

        cached = self._cache[goal_hash]
        if not cached.verified:
            cached.verified = True
            cached.lean_certificate = lean_certificate
            self.stats.verified_proofs += 1

        return True

    def remove(self, goal_hash: str) -> bool:
        """Remove a proof from cache."""
        if goal_hash not in self._cache:
            return False

        cached = self._cache[goal_hash]
        proof_hash = cached.proof.hash

        del self._cache[goal_hash]
        if proof_hash in self._proof_index:
            del self._proof_index[proof_hash]

        if goal_hash in self._access_order:
            self._access_order.remove(goal_hash)

        self.stats.total_proofs = len(self._cache)
        return True

    def contains(self, goal: Goal) -> bool:
        """Check if a proof is cached for the goal."""
        return goal.hash in self._cache

    def get_verified_proofs(self) -> List[CachedProof]:
        """Get all verified proofs."""
        return [c for c in self._cache.values() if c.verified]

    def get_unverified_proofs(self) -> List[CachedProof]:
        """Get all unverified proofs."""
        return [c for c in self._cache.values() if not c.verified]

    def _update_lru(self, goal_hash: str) -> None:
        """Update LRU order."""
        if goal_hash in self._access_order:
            self._access_order.remove(goal_hash)
        self._access_order.append(goal_hash)

    def _ensure_capacity(self) -> None:
        """Evict entries if cache is at capacity."""
        while len(self._cache) >= self.max_size:
            self._evict_lru()

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._access_order:
            return

        # Prefer to evict unverified proofs
        for goal_hash in self._access_order:
            if goal_hash in self._cache:
                cached = self._cache[goal_hash]
                if not cached.verified:
                    self.remove(goal_hash)
                    self.stats.evictions += 1
                    return

        # Fall back to LRU eviction
        goal_hash = self._access_order.pop(0)
        self.remove(goal_hash)
        self.stats.evictions += 1

    def _load_from_disk(self) -> None:
        """Load cache from disk."""
        if not self.persist_path:
            return

        try:
            cache_file = self.persist_path / "proof_cache.pkl"
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)
                    self._cache = data.get("cache", {})
                    self._proof_index = data.get("index", {})
                    self._access_order = data.get("order", [])
                    self.stats = data.get("stats", CacheStats())

                logger.info(f"Loaded {len(self._cache)} proofs from disk")

        except Exception as e:
            logger.error(f"Failed to load cache from disk: {e}")

    def save_to_disk(self) -> None:
        """Save cache to disk."""
        if not self.persist_path:
            return

        try:
            self.persist_path.mkdir(parents=True, exist_ok=True)
            cache_file = self.persist_path / "proof_cache.pkl"

            with open(cache_file, "wb") as f:
                pickle.dump({
                    "cache": self._cache,
                    "index": self._proof_index,
                    "order": self._access_order,
                    "stats": self.stats,
                }, f)

            logger.info(f"Saved {len(self._cache)} proofs to disk")

        except Exception as e:
            logger.error(f"Failed to save cache to disk: {e}")

    def export_to_json(self, path: Path) -> None:
        """Export verified proofs to JSON."""
        verified = self.get_verified_proofs()

        data = []
        for cached in verified:
            data.append({
                "goal_hash": cached.goal_hash,
                "proof_hash": cached.content_hash,
                "tactic": cached.proof.tactic.to_lean(),
                "verified": cached.verified,
                "lean_certificate": cached.lean_certificate,
                "timestamp": cached.timestamp,
            })

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(data)} verified proofs to {path}")


class DistributedProofCache:
    """
    Distributed proof cache using DHT.

    Extends ProofCache with network distribution capabilities.
    """

    def __init__(
        self,
        local_cache: ProofCache,
        dht: Any,  # ProofMeshDHT
    ):
        self.local = local_cache
        self.dht = dht

        # Pending requests
        self._pending: Dict[str, asyncio.Event] = {}

    async def get(self, goal: Goal) -> Optional[Proof]:
        """Get proof from local cache or network."""
        # Check local first
        proof = self.local.get(goal)
        if proof is not None:
            return proof

        # Try network
        return await self._fetch_from_network(goal)

    async def _fetch_from_network(self, goal: Goal) -> Optional[Proof]:
        """Fetch proof from DHT network."""
        goal_hash = goal.hash
        key = f"proof:{goal_hash}".encode()

        data = await self.dht.get(key)
        if data is None:
            return None

        try:
            cached = pickle.loads(data)
            if isinstance(cached, CachedProof):
                # Cache locally
                self.local.put(
                    goal,
                    cached.proof,
                    verified=cached.verified,
                    lean_certificate=cached.lean_certificate,
                    source_peer="network",
                )
                return cached.proof

        except Exception as e:
            logger.error(f"Failed to deserialize proof: {e}")

        return None

    async def put(
        self,
        goal: Goal,
        proof: Proof,
        verified: bool = False,
        broadcast: bool = True,
    ) -> str:
        """
        Cache proof locally and optionally broadcast to network.

        Args:
            goal: The goal
            proof: The proof
            verified: Whether verified
            broadcast: Whether to broadcast to DHT

        Returns:
            Proof hash
        """
        # Store locally
        proof_hash = self.local.put(goal, proof, verified=verified)

        # Broadcast to DHT
        if broadcast:
            await self._broadcast_proof(goal, proof, verified)

        return proof_hash

    async def _broadcast_proof(
        self,
        goal: Goal,
        proof: Proof,
        verified: bool,
    ) -> None:
        """Broadcast proof to DHT network."""
        goal_hash = goal.hash
        key = f"proof:{goal_hash}".encode()

        cached = CachedProof(
            proof=proof,
            goal_hash=goal_hash,
            verified=verified,
        )

        data = pickle.dumps(cached)
        await self.dht.store(key, data)

        logger.debug(f"Broadcast proof {goal_hash[:8]} to network")

    async def sync_verified(self) -> int:
        """Sync verified proofs to network."""
        verified = self.local.get_verified_proofs()
        synced = 0

        for cached in verified:
            try:
                key = f"proof:{cached.goal_hash}".encode()
                data = pickle.dumps(cached)
                if await self.dht.store(key, data):
                    synced += 1
            except Exception as e:
                logger.error(f"Failed to sync proof: {e}")

        logger.info(f"Synced {synced} verified proofs to network")
        return synced


class ProofLedger:
    """
    Append-only ledger of verified proofs.

    Provides an immutable record of all verified proofs
    for audit and reproducibility.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = path
        self.entries: List[Dict[str, Any]] = []
        self._entry_index: Dict[str, int] = {}

        if path and path.exists():
            self._load()

    def append(
        self,
        goal_hash: str,
        proof_hash: str,
        tactic: str,
        lean_certificate: str,
        timestamp: Optional[float] = None,
    ) -> int:
        """
        Append a verified proof to the ledger.

        Returns:
            Index of the new entry
        """
        if goal_hash in self._entry_index:
            # Already in ledger
            return self._entry_index[goal_hash]

        entry = {
            "index": len(self.entries),
            "goal_hash": goal_hash,
            "proof_hash": proof_hash,
            "tactic": tactic,
            "lean_certificate": lean_certificate,
            "timestamp": timestamp or time.time(),
            "prev_hash": self._prev_hash(),
        }

        # Compute entry hash
        entry["entry_hash"] = self._compute_hash(entry)

        self.entries.append(entry)
        self._entry_index[goal_hash] = entry["index"]

        if self.path:
            self._append_to_file(entry)

        return entry["index"]

    def get(self, goal_hash: str) -> Optional[Dict[str, Any]]:
        """Get ledger entry by goal hash."""
        if goal_hash not in self._entry_index:
            return None
        idx = self._entry_index[goal_hash]
        return self.entries[idx]

    def verify_integrity(self) -> bool:
        """Verify ledger integrity (hash chain)."""
        for i, entry in enumerate(self.entries):
            # Verify index
            if entry["index"] != i:
                logger.error(f"Index mismatch at {i}")
                return False

            # Verify prev hash
            if i > 0:
                expected_prev = self.entries[i - 1]["entry_hash"]
                if entry["prev_hash"] != expected_prev:
                    logger.error(f"Hash chain broken at {i}")
                    return False

            # Verify entry hash
            expected_hash = self._compute_hash(entry)
            if entry["entry_hash"] != expected_hash:
                logger.error(f"Entry hash mismatch at {i}")
                return False

        return True

    def _prev_hash(self) -> str:
        """Get hash of previous entry."""
        if not self.entries:
            return "0" * 64
        return self.entries[-1]["entry_hash"]

    def _compute_hash(self, entry: Dict[str, Any]) -> str:
        """Compute hash of entry (excluding entry_hash field)."""
        data = {k: v for k, v in entry.items() if k != "entry_hash"}
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def _load(self) -> None:
        """Load ledger from file."""
        if not self.path:
            return

        try:
            with open(self.path, "r") as f:
                for line in f:
                    entry = json.loads(line)
                    self.entries.append(entry)
                    self._entry_index[entry["goal_hash"]] = entry["index"]

            logger.info(f"Loaded {len(self.entries)} ledger entries")

        except Exception as e:
            logger.error(f"Failed to load ledger: {e}")

    def _append_to_file(self, entry: Dict[str, Any]) -> None:
        """Append entry to file."""
        if not self.path:
            return

        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to append to ledger: {e}")

    def export(self, path: Path) -> None:
        """Export ledger to file."""
        with open(path, "w") as f:
            for entry in self.entries:
                f.write(json.dumps(entry) + "\n")

        logger.info(f"Exported {len(self.entries)} ledger entries")
