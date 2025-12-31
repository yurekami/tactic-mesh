"""
Distributed Hash Table for Proof Mesh Network.

Implements a Kademlia-style DHT for:
- Peer discovery
- Proof caching and retrieval
- Expert location services
"""

from __future__ import annotations

import asyncio
import hashlib
import heapq
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class NodeID:
    """160-bit node identifier."""
    value: bytes

    def __init__(self, value: Optional[bytes] = None):
        if value is None:
            import os
            value = os.urandom(20)
        self.value = value

    @classmethod
    def from_string(cls, s: str) -> NodeID:
        """Create NodeID from string (hashed)."""
        return cls(hashlib.sha1(s.encode()).digest())

    def distance(self, other: NodeID) -> int:
        """XOR distance metric."""
        return int.from_bytes(
            bytes(a ^ b for a, b in zip(self.value, other.value)),
            byteorder='big'
        )

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NodeID):
            return False
        return self.value == other.value

    def __lt__(self, other: NodeID) -> bool:
        return self.value < other.value

    def __repr__(self) -> str:
        return f"NodeID({self.value.hex()[:8]}...)"


@dataclass
class PeerInfo:
    """Information about a peer in the network."""
    node_id: NodeID
    address: str
    port: int
    experts: List[str] = field(default_factory=list)  # Hosted experts
    last_seen: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    reputation: float = 1.0

    @property
    def endpoint(self) -> str:
        return f"{self.address}:{self.port}"

    def update_seen(self) -> None:
        self.last_seen = time.time()

    def is_stale(self, timeout: float = 300.0) -> bool:
        """Check if peer hasn't been seen recently."""
        return time.time() - self.last_seen > timeout


@dataclass
class KBucket:
    """K-bucket for Kademlia routing table."""
    k: int = 20  # Bucket size
    peers: List[PeerInfo] = field(default_factory=list)

    def add(self, peer: PeerInfo) -> bool:
        """Add peer to bucket, return True if added."""
        # Check if already present
        for i, existing in enumerate(self.peers):
            if existing.node_id == peer.node_id:
                # Move to end (most recently seen)
                self.peers.pop(i)
                self.peers.append(peer)
                return True

        # Add if space available
        if len(self.peers) < self.k:
            self.peers.append(peer)
            return True

        # Bucket full - check if least recent is stale
        if self.peers[0].is_stale():
            self.peers.pop(0)
            self.peers.append(peer)
            return True

        return False

    def remove(self, node_id: NodeID) -> bool:
        """Remove peer from bucket."""
        for i, peer in enumerate(self.peers):
            if peer.node_id == node_id:
                self.peers.pop(i)
                return True
        return False

    def get_peers(self) -> List[PeerInfo]:
        """Get all peers in bucket."""
        return list(self.peers)


class RoutingTable:
    """Kademlia routing table with 160 k-buckets."""

    def __init__(self, local_id: NodeID, k: int = 20):
        self.local_id = local_id
        self.k = k
        self.buckets: List[KBucket] = [KBucket(k=k) for _ in range(160)]

    def _bucket_index(self, node_id: NodeID) -> int:
        """Get bucket index for a node ID."""
        distance = self.local_id.distance(node_id)
        if distance == 0:
            return 0
        return 159 - distance.bit_length() + 1

    def add_peer(self, peer: PeerInfo) -> bool:
        """Add peer to routing table."""
        if peer.node_id == self.local_id:
            return False
        idx = self._bucket_index(peer.node_id)
        return self.buckets[idx].add(peer)

    def remove_peer(self, node_id: NodeID) -> bool:
        """Remove peer from routing table."""
        idx = self._bucket_index(node_id)
        return self.buckets[idx].remove(node_id)

    def find_closest(self, target: NodeID, count: int = 20) -> List[PeerInfo]:
        """Find closest peers to target."""
        all_peers = []
        for bucket in self.buckets:
            all_peers.extend(bucket.get_peers())

        # Sort by distance to target
        all_peers.sort(key=lambda p: target.distance(p.node_id))
        return all_peers[:count]

    def get_all_peers(self) -> List[PeerInfo]:
        """Get all known peers."""
        peers = []
        for bucket in self.buckets:
            peers.extend(bucket.get_peers())
        return peers


@dataclass
class StoredValue:
    """Value stored in the DHT."""
    key: bytes
    value: bytes
    timestamp: float = field(default_factory=time.time)
    publisher: Optional[NodeID] = None
    ttl: float = 3600.0  # 1 hour default

    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl


class ProofMeshDHT:
    """
    Distributed Hash Table for the Proof Mesh network.

    Provides:
    - Peer discovery and routing
    - Proof storage and retrieval
    - Expert location services
    """

    def __init__(
        self,
        node_id: Optional[NodeID] = None,
        address: str = "0.0.0.0",
        port: int = 31337,
        k: int = 20,
        alpha: int = 3,  # Parallelism factor
    ):
        self.node_id = node_id or NodeID()
        self.address = address
        self.port = port
        self.k = k
        self.alpha = alpha

        self.routing_table = RoutingTable(self.node_id, k)
        self.storage: Dict[bytes, StoredValue] = {}

        # Expert registry
        self.local_experts: Set[str] = set()
        self.expert_locations: Dict[str, List[PeerInfo]] = {}

        # Network state
        self._running = False
        self._server: Optional[asyncio.Server] = None

        # Callbacks
        self._on_peer_discovered: Optional[Callable[[PeerInfo], None]] = None
        self._on_value_received: Optional[Callable[[bytes, bytes], None]] = None

    async def start(self, bootstrap_peers: Optional[List[str]] = None) -> None:
        """Start the DHT node."""
        logger.info(f"Starting DHT node {self.node_id} on {self.address}:{self.port}")

        self._running = True

        # Start server
        self._server = await asyncio.start_server(
            self._handle_connection,
            self.address,
            self.port,
        )

        # Bootstrap from known peers
        if bootstrap_peers:
            await self._bootstrap(bootstrap_peers)

        # Start maintenance tasks
        asyncio.create_task(self._maintenance_loop())

        logger.info("DHT node started")

    async def stop(self) -> None:
        """Stop the DHT node."""
        logger.info("Stopping DHT node")
        self._running = False

        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _bootstrap(self, peers: List[str]) -> None:
        """Bootstrap from known peers."""
        for peer_addr in peers:
            try:
                host, port = peer_addr.rsplit(":", 1)
                await self._ping_peer(host, int(port))
            except Exception as e:
                logger.warning(f"Failed to bootstrap from {peer_addr}: {e}")

        # Refresh routing table
        await self._refresh_routing_table()

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle incoming connection."""
        try:
            data = await reader.read(65536)
            if not data:
                return

            response = await self._process_message(data)
            if response:
                writer.write(response)
                await writer.drain()
        except Exception as e:
            logger.error(f"Connection error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def _process_message(self, data: bytes) -> Optional[bytes]:
        """Process incoming DHT message."""
        import json

        try:
            msg = json.loads(data.decode())
            msg_type = msg.get("type")

            if msg_type == "ping":
                return self._handle_ping(msg)
            elif msg_type == "find_node":
                return self._handle_find_node(msg)
            elif msg_type == "find_value":
                return self._handle_find_value(msg)
            elif msg_type == "store":
                return self._handle_store(msg)
            elif msg_type == "find_expert":
                return self._handle_find_expert(msg)
            else:
                logger.warning(f"Unknown message type: {msg_type}")
                return None

        except Exception as e:
            logger.error(f"Message processing error: {e}")
            return None

    def _handle_ping(self, msg: Dict) -> bytes:
        """Handle ping message."""
        import json

        # Update routing table with sender
        sender_id = NodeID(bytes.fromhex(msg["sender_id"]))
        peer = PeerInfo(
            node_id=sender_id,
            address=msg.get("address", ""),
            port=msg.get("port", 0),
            experts=msg.get("experts", []),
        )
        self.routing_table.add_peer(peer)

        response = {
            "type": "pong",
            "sender_id": self.node_id.value.hex(),
            "experts": list(self.local_experts),
        }
        return json.dumps(response).encode()

    def _handle_find_node(self, msg: Dict) -> bytes:
        """Handle find_node message."""
        import json

        target = NodeID(bytes.fromhex(msg["target"]))
        closest = self.routing_table.find_closest(target, self.k)

        response = {
            "type": "nodes",
            "nodes": [
                {
                    "node_id": p.node_id.value.hex(),
                    "address": p.address,
                    "port": p.port,
                    "experts": p.experts,
                }
                for p in closest
            ],
        }
        return json.dumps(response).encode()

    def _handle_find_value(self, msg: Dict) -> bytes:
        """Handle find_value message."""
        import json

        key = bytes.fromhex(msg["key"])

        if key in self.storage:
            stored = self.storage[key]
            if not stored.is_expired():
                return json.dumps({
                    "type": "value",
                    "key": key.hex(),
                    "value": stored.value.hex(),
                }).encode()

        # Return closest nodes instead
        target = NodeID(key)
        return self._handle_find_node({"target": target.value.hex()})

    def _handle_store(self, msg: Dict) -> bytes:
        """Handle store message."""
        import json

        key = bytes.fromhex(msg["key"])
        value = bytes.fromhex(msg["value"])
        ttl = msg.get("ttl", 3600.0)

        self.storage[key] = StoredValue(
            key=key,
            value=value,
            ttl=ttl,
        )

        return json.dumps({"type": "stored", "success": True}).encode()

    def _handle_find_expert(self, msg: Dict) -> bytes:
        """Handle find_expert message."""
        import json

        expert_id = msg["expert_id"]

        if expert_id in self.local_experts:
            # We host this expert
            return json.dumps({
                "type": "expert_found",
                "expert_id": expert_id,
                "peer": {
                    "node_id": self.node_id.value.hex(),
                    "address": self.address,
                    "port": self.port,
                },
            }).encode()

        # Return known locations
        if expert_id in self.expert_locations:
            peers = self.expert_locations[expert_id]
            return json.dumps({
                "type": "expert_locations",
                "expert_id": expert_id,
                "peers": [
                    {
                        "node_id": p.node_id.value.hex(),
                        "address": p.address,
                        "port": p.port,
                    }
                    for p in peers
                ],
            }).encode()

        return json.dumps({"type": "expert_not_found"}).encode()

    async def _ping_peer(self, address: str, port: int) -> Optional[PeerInfo]:
        """Ping a peer and add to routing table."""
        import json

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(address, port),
                timeout=5.0,
            )

            msg = json.dumps({
                "type": "ping",
                "sender_id": self.node_id.value.hex(),
                "address": self.address,
                "port": self.port,
                "experts": list(self.local_experts),
            }).encode()

            writer.write(msg)
            await writer.drain()

            response = await asyncio.wait_for(reader.read(65536), timeout=5.0)
            writer.close()
            await writer.wait_closed()

            resp = json.loads(response.decode())
            if resp.get("type") == "pong":
                peer = PeerInfo(
                    node_id=NodeID(bytes.fromhex(resp["sender_id"])),
                    address=address,
                    port=port,
                    experts=resp.get("experts", []),
                )
                self.routing_table.add_peer(peer)

                if self._on_peer_discovered:
                    self._on_peer_discovered(peer)

                return peer

        except Exception as e:
            logger.debug(f"Ping failed for {address}:{port}: {e}")

        return None

    async def _refresh_routing_table(self) -> None:
        """Refresh routing table by looking up random IDs."""
        for _ in range(3):
            random_id = NodeID()
            await self.find_node(random_id)

    async def _maintenance_loop(self) -> None:
        """Periodic maintenance tasks."""
        while self._running:
            try:
                # Clean expired storage
                expired = [
                    k for k, v in self.storage.items()
                    if v.is_expired()
                ]
                for k in expired:
                    del self.storage[k]

                # Refresh routing table
                await self._refresh_routing_table()

                # Republish local values
                await self._republish_values()

            except Exception as e:
                logger.error(f"Maintenance error: {e}")

            await asyncio.sleep(60)  # Run every minute

    async def _republish_values(self) -> None:
        """Republish values we're responsible for."""
        for key, value in list(self.storage.items()):
            if value.publisher == self.node_id:
                await self.store(key, value.value, value.ttl)

    # Public API

    async def find_node(self, target: NodeID) -> List[PeerInfo]:
        """Find nodes closest to target."""
        import json

        closest = self.routing_table.find_closest(target, self.k)
        queried: Set[NodeID] = set()
        results: List[PeerInfo] = list(closest)

        while True:
            # Get unqueried nodes closest to target
            to_query = [
                p for p in results
                if p.node_id not in queried
            ][:self.alpha]

            if not to_query:
                break

            # Query in parallel
            tasks = []
            for peer in to_query:
                queried.add(peer.node_id)
                tasks.append(self._query_find_node(peer, target))

            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # Process responses
            for resp in responses:
                if isinstance(resp, list):
                    for peer in resp:
                        if peer.node_id not in {p.node_id for p in results}:
                            results.append(peer)
                            self.routing_table.add_peer(peer)

            # Sort by distance
            results.sort(key=lambda p: target.distance(p.node_id))
            results = results[:self.k]

        return results

    async def _query_find_node(
        self,
        peer: PeerInfo,
        target: NodeID,
    ) -> List[PeerInfo]:
        """Query a peer for nodes closest to target."""
        import json

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(peer.address, peer.port),
                timeout=5.0,
            )

            msg = json.dumps({
                "type": "find_node",
                "target": target.value.hex(),
            }).encode()

            writer.write(msg)
            await writer.drain()

            response = await asyncio.wait_for(reader.read(65536), timeout=5.0)
            writer.close()
            await writer.wait_closed()

            resp = json.loads(response.decode())
            if resp.get("type") == "nodes":
                return [
                    PeerInfo(
                        node_id=NodeID(bytes.fromhex(n["node_id"])),
                        address=n["address"],
                        port=n["port"],
                        experts=n.get("experts", []),
                    )
                    for n in resp["nodes"]
                ]

        except Exception as e:
            logger.debug(f"Find node query failed: {e}")

        return []

    async def store(self, key: bytes, value: bytes, ttl: float = 3600.0) -> bool:
        """Store a value in the DHT."""
        import json

        # Find closest nodes
        target = NodeID(key)
        closest = await self.find_node(target)

        if not closest:
            # Store locally
            self.storage[key] = StoredValue(
                key=key,
                value=value,
                publisher=self.node_id,
                ttl=ttl,
            )
            return True

        # Store on closest nodes
        success = False
        for peer in closest[:self.k]:
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(peer.address, peer.port),
                    timeout=5.0,
                )

                msg = json.dumps({
                    "type": "store",
                    "key": key.hex(),
                    "value": value.hex(),
                    "ttl": ttl,
                }).encode()

                writer.write(msg)
                await writer.drain()

                response = await asyncio.wait_for(reader.read(65536), timeout=5.0)
                writer.close()
                await writer.wait_closed()

                resp = json.loads(response.decode())
                if resp.get("success"):
                    success = True

            except Exception as e:
                logger.debug(f"Store failed on {peer.endpoint}: {e}")

        return success

    async def get(self, key: bytes) -> Optional[bytes]:
        """Retrieve a value from the DHT."""
        import json

        # Check local storage first
        if key in self.storage:
            stored = self.storage[key]
            if not stored.is_expired():
                return stored.value

        # Find value in network
        target = NodeID(key)
        closest = await self.find_node(target)

        for peer in closest:
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(peer.address, peer.port),
                    timeout=5.0,
                )

                msg = json.dumps({
                    "type": "find_value",
                    "key": key.hex(),
                }).encode()

                writer.write(msg)
                await writer.drain()

                response = await asyncio.wait_for(reader.read(65536), timeout=5.0)
                writer.close()
                await writer.wait_closed()

                resp = json.loads(response.decode())
                if resp.get("type") == "value":
                    value = bytes.fromhex(resp["value"])
                    # Cache locally
                    self.storage[key] = StoredValue(key=key, value=value)
                    return value

            except Exception as e:
                logger.debug(f"Get failed from {peer.endpoint}: {e}")

        return None

    def register_expert(self, expert_id: str) -> None:
        """Register a locally hosted expert."""
        self.local_experts.add(expert_id)
        logger.info(f"Registered expert: {expert_id}")

    async def find_expert(self, expert_id: str) -> Optional[PeerInfo]:
        """Find a peer hosting the specified expert."""
        import json

        # Check local
        if expert_id in self.local_experts:
            return PeerInfo(
                node_id=self.node_id,
                address=self.address,
                port=self.port,
                experts=list(self.local_experts),
            )

        # Check cached locations
        if expert_id in self.expert_locations:
            peers = self.expert_locations[expert_id]
            for peer in peers:
                if not peer.is_stale():
                    return peer

        # Search network
        for peer in self.routing_table.get_all_peers():
            if expert_id in peer.experts:
                return peer

            # Query peer
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(peer.address, peer.port),
                    timeout=5.0,
                )

                msg = json.dumps({
                    "type": "find_expert",
                    "expert_id": expert_id,
                }).encode()

                writer.write(msg)
                await writer.drain()

                response = await asyncio.wait_for(reader.read(65536), timeout=5.0)
                writer.close()
                await writer.wait_closed()

                resp = json.loads(response.decode())
                if resp.get("type") == "expert_found":
                    found_peer = PeerInfo(
                        node_id=NodeID(bytes.fromhex(resp["peer"]["node_id"])),
                        address=resp["peer"]["address"],
                        port=resp["peer"]["port"],
                        experts=[expert_id],
                    )
                    # Cache location
                    if expert_id not in self.expert_locations:
                        self.expert_locations[expert_id] = []
                    self.expert_locations[expert_id].append(found_peer)
                    return found_peer

            except Exception as e:
                logger.debug(f"Expert query failed: {e}")

        return None

    def on_peer_discovered(self, callback: Callable[[PeerInfo], None]) -> None:
        """Set callback for peer discovery."""
        self._on_peer_discovered = callback

    def on_value_received(self, callback: Callable[[bytes, bytes], None]) -> None:
        """Set callback for value reception."""
        self._on_value_received = callback
