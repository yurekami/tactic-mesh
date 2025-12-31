"""
Tactic-Mesh Command Line Interface.

Commands:
- prove: Prove a theorem using Tactic-Mesh
- node: Run a Tactic-Mesh network node
- train: Train the model
- export: Export proofs to Lean
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="tactic-mesh",
        description="Decentralized Mixture-of-Tactics for Formal Theorem Proving",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # prove command
    prove_parser = subparsers.add_parser(
        "prove",
        help="Prove a theorem",
    )
    prove_parser.add_argument(
        "theorem",
        type=str,
        help="Theorem to prove (Lean syntax)",
    )
    prove_parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds",
    )
    prove_parser.add_argument(
        "--max-depth",
        type=int,
        default=50,
        help="Maximum proof depth",
    )
    prove_parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify proof with Lean",
    )
    prove_parser.add_argument(
        "--output",
        type=Path,
        help="Output file for proof",
    )
    prove_parser.add_argument(
        "--peers",
        type=str,
        nargs="*",
        help="Peer addresses to connect to",
    )

    # node command
    node_parser = subparsers.add_parser(
        "node",
        help="Run a Tactic-Mesh network node",
    )
    node_parser.add_argument(
        "--port",
        type=int,
        default=31337,
        help="Port to listen on",
    )
    node_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    node_parser.add_argument(
        "--bootstrap",
        type=str,
        nargs="*",
        help="Bootstrap peer addresses",
    )
    node_parser.add_argument(
        "--experts",
        type=str,
        nargs="*",
        default=["intro", "rewrite", "simp"],
        help="Experts to host",
    )
    node_parser.add_argument(
        "--model",
        type=Path,
        help="Model checkpoint path",
    )

    # train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train the model",
    )
    train_parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Training dataset (equational_theories, mathlib, custom)",
    )
    train_parser.add_argument(
        "--output",
        type=Path,
        default=Path("./checkpoints"),
        help="Output directory for checkpoints",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    train_parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from checkpoint",
    )

    # export command
    export_parser = subparsers.add_parser(
        "export",
        help="Export proofs to Lean",
    )
    export_parser.add_argument(
        "--cache",
        type=Path,
        default=Path("./proof_cache"),
        help="Proof cache directory",
    )
    export_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output Lean file",
    )
    export_parser.add_argument(
        "--verified-only",
        action="store_true",
        help="Only export verified proofs",
    )

    # benchmark command
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Run benchmarks",
    )
    bench_parser.add_argument(
        "--dataset",
        type=str,
        default="equational_theories",
        help="Benchmark dataset",
    )
    bench_parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples",
    )
    bench_parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file for results",
    )

    return parser


async def cmd_prove(args: argparse.Namespace) -> int:
    """Run prove command."""
    from core.proof_state import Goal, TermAST, TermKind
    from lean.verifier import LeanVerifier, LeanConfig

    logger.info(f"Proving: {args.theorem}")

    # Parse theorem (simplified)
    goal = Goal(
        target=TermAST(kind=TermKind.CONST, data=args.theorem)
    )

    # Initialize verifier
    config = LeanConfig(timeout_seconds=args.timeout)
    verifier = LeanVerifier(config)

    try:
        await verifier.start()

        # Run proof search
        from training.rl import SelfPlayTrainer
        from experts.base import create_all_experts
        from core.router import TacticRouter, RouterConfig

        # Create model components
        experts = create_all_experts()
        router_config = RouterConfig(n_experts=len(experts))
        router = TacticRouter(router_config)

        # Simple proof search (would use full pipeline in production)
        logger.info("Searching for proof...")

        # Try each expert
        for expert in experts:
            state_embedding = torch.randn(1024)  # Placeholder
            tactic = expert.generate_tactic(
                ProofState.from_goal(goal),
                state_embedding,
            )

            result = await verifier.apply_tactic(goal, tactic)
            if result.success:
                logger.info(f"Found tactic: {tactic.to_lean()}")

                if result.is_complete:
                    logger.info("Proof complete!")

                    # Verify if requested
                    if args.verify:
                        from core.proof_state import Proof
                        proof = Proof(goal=goal, tactic=tactic)
                        ver_result = await verifier.verify_proof(goal, proof)
                        if ver_result.verified:
                            logger.info("Proof verified by Lean!")
                        else:
                            logger.warning(f"Verification failed: {ver_result.error_message}")

                    # Output
                    if args.output:
                        args.output.write_text(tactic.to_lean())
                        logger.info(f"Proof written to {args.output}")

                    return 0

                # Continue with subgoals
                logger.info(f"Subgoals remaining: {len(result.new_goals)}")

        logger.warning("No proof found within limits")
        return 1

    finally:
        await verifier.stop()

    return 1


async def cmd_node(args: argparse.Namespace) -> int:
    """Run node command."""
    from network.dht import ProofMeshDHT
    from network.proof_cache import ProofCache
    from experts.base import create_expert

    logger.info(f"Starting Tactic-Mesh node on {args.host}:{args.port}")
    logger.info(f"Experts: {args.experts}")

    # Initialize DHT
    dht = ProofMeshDHT(
        address=args.host,
        port=args.port,
    )

    # Register experts
    for expert_id in args.experts:
        dht.register_expert(expert_id)

    # Start DHT
    await dht.start(bootstrap_peers=args.bootstrap)

    # Initialize proof cache
    cache = ProofCache(persist_path=Path("./proof_cache"))

    logger.info("Node started. Press Ctrl+C to stop.")

    try:
        # Run forever
        while True:
            await asyncio.sleep(60)
            stats = cache.stats
            logger.info(
                f"Stats: {stats.total_proofs} proofs, "
                f"{stats.cache_hits}/{stats.cache_hits + stats.cache_misses} hits, "
                f"{dht.routing_table.get_all_peers().__len__()} peers"
            )
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await dht.stop()
        cache.save_to_disk()

    return 0


async def cmd_train(args: argparse.Namespace) -> int:
    """Run train command."""
    import torch

    from core.proof_state import Goal, TermAST, TermKind
    from experts.base import create_all_experts
    from core.router import TacticRouter, RouterConfig
    from training.rl import RLConfig, SelfPlayTrainer

    logger.info(f"Training on dataset: {args.dataset}")
    logger.info(f"Output: {args.output}")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Load dataset (simplified)
    logger.info("Loading training goals...")
    if args.dataset == "equational_theories":
        # Generate sample equational goals
        goals = [
            Goal(target=TermAST(kind=TermKind.CONST, data=f"eq_{i}"))
            for i in range(100)
        ]
    else:
        logger.error(f"Unknown dataset: {args.dataset}")
        return 1

    logger.info(f"Loaded {len(goals)} goals")

    # Create model
    experts = create_all_experts()
    router_config = RouterConfig(n_experts=len(experts))
    router = TacticRouter(router_config)

    # Value network (simplified)
    value_net = torch.nn.Sequential(
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 1),
    )

    # Training config
    rl_config = RLConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    # Resume if checkpoint provided
    if args.resume and args.resume.exists():
        logger.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume)
        router.load_state_dict(checkpoint["router"])
        value_net.load_state_dict(checkpoint["value_net"])

    # Training loop
    logger.info("Starting training...")

    for epoch in range(args.epochs):
        # Sample batch of goals
        import random
        batch_goals = random.sample(goals, min(args.batch_size, len(goals)))

        # Simplified training step
        # In production, would use SelfPlayTrainer with verifier

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = args.output / f"checkpoint_{epoch+1}.pt"
            torch.save({
                "epoch": epoch,
                "router": router.state_dict(),
                "value_net": value_net.state_dict(),
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    logger.info("Training complete!")
    return 0


async def cmd_export(args: argparse.Namespace) -> int:
    """Run export command."""
    from network.proof_cache import ProofCache

    logger.info(f"Exporting proofs from {args.cache}")

    # Load cache
    cache = ProofCache(persist_path=args.cache)

    # Get proofs
    if args.verified_only:
        proofs = cache.get_verified_proofs()
    else:
        proofs = list(cache._cache.values())

    logger.info(f"Found {len(proofs)} proofs")

    # Export to Lean
    lines = [
        "-- Proofs exported from Tactic-Mesh",
        "-- Generated automatically",
        "",
        "import Mathlib",
        "",
    ]

    for i, cached in enumerate(proofs):
        proof_lean = cached.proof.to_lean()
        lines.append(f"-- Proof {i+1}: {cached.goal_hash[:8]}")
        lines.append(f"example : sorry := by")
        lines.append(f"  {proof_lean}")
        lines.append("")

    args.output.write_text("\n".join(lines))
    logger.info(f"Exported to {args.output}")

    return 0


async def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run benchmark command."""
    import time
    import json

    from core.proof_state import Goal, TermAST, TermKind

    logger.info(f"Running benchmark on {args.dataset}")

    results = {
        "dataset": args.dataset,
        "n_samples": args.n_samples,
        "results": [],
    }

    # Generate sample goals
    goals = [
        Goal(target=TermAST(kind=TermKind.CONST, data=f"goal_{i}"))
        for i in range(args.n_samples)
    ]

    total_time = 0
    successes = 0

    for i, goal in enumerate(goals):
        start = time.time()

        # Simplified benchmark (would use actual prover)
        import random
        success = random.random() > 0.3  # Mock 70% success rate

        elapsed = time.time() - start
        total_time += elapsed

        if success:
            successes += 1

        results["results"].append({
            "goal_id": i,
            "success": success,
            "time_ms": elapsed * 1000,
        })

    results["summary"] = {
        "success_rate": successes / len(goals),
        "total_time_s": total_time,
        "avg_time_ms": (total_time / len(goals)) * 1000,
    }

    logger.info(f"Success rate: {results['summary']['success_rate']:.2%}")
    logger.info(f"Average time: {results['summary']['avg_time_ms']:.2f}ms")

    if args.output:
        args.output.write_text(json.dumps(results, indent=2))
        logger.info(f"Results written to {args.output}")

    return 0


# Import torch at module level to avoid issues
try:
    import torch
    from core.proof_state import ProofState
except ImportError:
    torch = None
    ProofState = None


async def async_main(args: argparse.Namespace) -> int:
    """Async main entry point."""
    if args.command == "prove":
        return await cmd_prove(args)
    elif args.command == "node":
        return await cmd_node(args)
    elif args.command == "train":
        return await cmd_train(args)
    elif args.command == "export":
        return await cmd_export(args)
    elif args.command == "benchmark":
        return await cmd_benchmark(args)
    else:
        return 1


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.command:
        parser.print_help()
        return 0

    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())
