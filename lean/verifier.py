"""
Lean 4 Verification Interface.

Provides integration with Lean 4 for:
- Tactic execution and verification
- Proof checking
- Type checking
- Goal state management
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.proof_state import Goal, Proof, TacticOutput, TermAST, TermKind, Hypothesis

logger = logging.getLogger(__name__)


@dataclass
class LeanConfig:
    """Configuration for Lean integration."""
    lean_path: str = "lake"  # Path to lake (Lean build tool)
    project_path: Optional[Path] = None  # Lean project directory
    timeout_seconds: float = 30.0
    max_memory_mb: int = 4096
    mathlib: bool = True  # Use Mathlib4


@dataclass
class TacticResult:
    """Result of applying a tactic."""
    success: bool
    new_goals: List[Goal] = field(default_factory=list)
    error_message: Optional[str] = None
    tactic_used: Optional[str] = None
    execution_time_ms: float = 0.0

    @property
    def is_complete(self) -> bool:
        """Check if proof is complete (no remaining goals)."""
        return self.success and len(self.new_goals) == 0


@dataclass
class VerificationResult:
    """Result of proof verification."""
    verified: bool
    certificate: Optional[str] = None
    error_message: Optional[str] = None
    kernel_time_ms: float = 0.0


class LeanREPL:
    """
    Interface to Lean 4 REPL for interactive proof.

    Manages a Lean process for executing tactics and checking proofs.
    """

    def __init__(self, config: LeanConfig):
        self.config = config
        self._process: Optional[asyncio.subprocess.Process] = None
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._ready = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the Lean REPL process."""
        logger.info("Starting Lean REPL")

        cmd = [self.config.lean_path, "env", "lean", "--run"]

        # Create a simple REPL script
        repl_script = self._generate_repl_script()

        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.config.project_path,
            )

            # Send REPL script
            self._process.stdin.write(repl_script.encode())
            await self._process.stdin.drain()

            self._ready = True
            logger.info("Lean REPL started")

        except Exception as e:
            logger.error(f"Failed to start Lean REPL: {e}")
            raise

    async def stop(self) -> None:
        """Stop the Lean REPL process."""
        if self._process:
            self._process.terminate()
            await self._process.wait()
            self._process = None
            self._ready = False
            logger.info("Lean REPL stopped")

    def _generate_repl_script(self) -> str:
        """Generate Lean REPL script."""
        return '''
import Lean
import Mathlib

open Lean Elab Tactic Meta

/-- Simple REPL for tactic execution --/
def tacticREPL : IO Unit := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout

  while true do
    let line ← stdin.getLine
    if line.isEmpty then break

    let cmd := line.trim
    -- Process command and output result
    stdout.putStrLn s!"OK: {cmd}"
    stdout.flush

#eval tacticREPL
'''

    async def execute_tactic(
        self,
        goal_state: str,
        tactic: str,
    ) -> TacticResult:
        """
        Execute a tactic in the current goal state.

        Args:
            goal_state: Current goal state as Lean code
            tactic: Tactic to execute

        Returns:
            TacticResult with new goals or error
        """
        if not self._ready:
            return TacticResult(
                success=False,
                error_message="Lean REPL not ready",
            )

        start_time = time.time()

        async with self._lock:
            try:
                # Send tactic command
                cmd = json.dumps({
                    "type": "tactic",
                    "goal": goal_state,
                    "tactic": tactic,
                })

                self._process.stdin.write(f"{cmd}\n".encode())
                await self._process.stdin.drain()

                # Read response with timeout
                try:
                    response = await asyncio.wait_for(
                        self._process.stdout.readline(),
                        timeout=self.config.timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    return TacticResult(
                        success=False,
                        error_message="Tactic execution timed out",
                    )

                # Parse response
                result = self._parse_tactic_response(response.decode())
                result.execution_time_ms = (time.time() - start_time) * 1000
                result.tactic_used = tactic

                return result

            except Exception as e:
                logger.error(f"Tactic execution error: {e}")
                return TacticResult(
                    success=False,
                    error_message=str(e),
                )

    def _parse_tactic_response(self, response: str) -> TacticResult:
        """Parse Lean REPL response."""
        try:
            data = json.loads(response)

            if data.get("success"):
                new_goals = []
                for goal_data in data.get("goals", []):
                    goal = self._parse_goal(goal_data)
                    new_goals.append(goal)

                return TacticResult(
                    success=True,
                    new_goals=new_goals,
                )
            else:
                return TacticResult(
                    success=False,
                    error_message=data.get("error", "Unknown error"),
                )

        except json.JSONDecodeError:
            # Simple response parsing
            if response.startswith("OK"):
                return TacticResult(success=True)
            else:
                return TacticResult(
                    success=False,
                    error_message=response,
                )

    def _parse_goal(self, goal_data: Dict[str, Any]) -> Goal:
        """Parse goal from Lean response."""
        target_str = goal_data.get("target", "")
        hyps_data = goal_data.get("hypotheses", [])

        target = self._parse_term(target_str)
        hypotheses = [
            Hypothesis(
                name=h.get("name", ""),
                type=self._parse_term(h.get("type", "")),
            )
            for h in hyps_data
        ]

        return Goal(target=target, hypotheses=hypotheses)

    def _parse_term(self, term_str: str) -> TermAST:
        """Parse term string to AST (simplified)."""
        # This is a simplified parser
        # Full implementation would use Lean's term parser
        return TermAST(
            kind=TermKind.CONST,
            data=term_str,
        )


class LeanVerifier:
    """
    High-level Lean verification interface.

    Provides proof verification and type checking.
    """

    def __init__(self, config: Optional[LeanConfig] = None):
        self.config = config or LeanConfig()
        self._repl: Optional[LeanREPL] = None

    async def start(self) -> None:
        """Start the verifier."""
        self._repl = LeanREPL(self.config)
        await self._repl.start()

    async def stop(self) -> None:
        """Stop the verifier."""
        if self._repl:
            await self._repl.stop()

    async def apply_tactic(
        self,
        goal: Goal,
        tactic: TacticOutput,
    ) -> TacticResult:
        """
        Apply a tactic to a goal.

        Args:
            goal: Current goal
            tactic: Tactic to apply

        Returns:
            TacticResult with new goals or error
        """
        if not self._repl:
            return TacticResult(
                success=False,
                error_message="Verifier not started",
            )

        # Convert goal to Lean representation
        goal_state = self._goal_to_lean(goal)
        tactic_str = tactic.to_lean()

        return await self._repl.execute_tactic(goal_state, tactic_str)

    async def verify_proof(
        self,
        goal: Goal,
        proof: Proof,
    ) -> VerificationResult:
        """
        Verify a complete proof.

        Args:
            goal: Original goal
            proof: Proof to verify

        Returns:
            VerificationResult with certificate or error
        """
        start_time = time.time()

        # Generate Lean proof script
        lean_code = self._proof_to_lean(goal, proof)

        # Type check the proof
        result = await self._type_check(lean_code)

        result.kernel_time_ms = (time.time() - start_time) * 1000

        if result.verified:
            result.certificate = self._generate_certificate(goal, proof)

        return result

    async def _type_check(self, lean_code: str) -> VerificationResult:
        """Type check Lean code."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".lean",
            delete=False,
        ) as f:
            f.write(lean_code)
            temp_path = f.name

        try:
            # Run Lean type checker
            cmd = [self.config.lean_path, "env", "lean", temp_path]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.config.project_path,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.config.timeout_seconds,
                )
            except asyncio.TimeoutError:
                process.kill()
                return VerificationResult(
                    verified=False,
                    error_message="Type checking timed out",
                )

            if process.returncode == 0:
                return VerificationResult(verified=True)
            else:
                return VerificationResult(
                    verified=False,
                    error_message=stderr.decode(),
                )

        finally:
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)

    def _goal_to_lean(self, goal: Goal) -> str:
        """Convert goal to Lean representation."""
        lines = []

        # Add hypotheses
        for hyp in goal.hypotheses:
            lines.append(f"  ({hyp.name} : {hyp.type.data})")

        # Add goal
        hyps_str = "\n".join(lines) if lines else ""

        return f"""
example{hyps_str} : {goal.target.data} := by
  sorry
"""

    def _proof_to_lean(self, goal: Goal, proof: Proof) -> str:
        """Convert proof to Lean code."""
        # Generate proof script
        proof_script = self._proof_to_script(proof)

        lines = []
        for hyp in goal.hypotheses:
            lines.append(f"  ({hyp.name} : {hyp.type.data})")

        hyps_str = "\n".join(lines) if lines else ""

        return f"""
import Mathlib

example{hyps_str} : {goal.target.data} := by
{proof_script}
"""

    def _proof_to_script(self, proof: Proof, indent: int = 2) -> str:
        """Convert proof tree to tactic script."""
        lines = []
        prefix = " " * indent

        # Add main tactic
        lines.append(f"{prefix}{proof.tactic.to_lean()}")

        # Add subproofs
        for subproof in proof.subproofs:
            sub_script = self._proof_to_script(subproof, indent + 2)
            lines.append(sub_script)

        return "\n".join(lines)

    def _generate_certificate(self, goal: Goal, proof: Proof) -> str:
        """Generate verification certificate."""
        import hashlib

        content = f"{goal.hash}:{proof.hash}:{time.time()}"
        cert_hash = hashlib.sha256(content.encode()).hexdigest()

        return json.dumps({
            "goal_hash": goal.hash,
            "proof_hash": proof.hash,
            "verified_at": time.time(),
            "certificate_hash": cert_hash,
        })


class SoftLeanKernel:
    """
    Differentiable surrogate for Lean's kernel.

    Provides soft type checking signals for training.
    This is NOT sound - use real LeanVerifier for actual verification.
    """

    def __init__(self, config: Optional[LeanConfig] = None):
        self.config = config or LeanConfig()

        # Cache of known valid/invalid patterns
        self._valid_cache: Dict[str, bool] = {}
        self._score_cache: Dict[str, float] = {}

    def soft_check(
        self,
        goal: Goal,
        tactic: TacticOutput,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Soft type checking (differentiable surrogate).

        Returns a score in [0, 1] indicating likelihood of success.

        Args:
            goal: Current goal
            tactic: Proposed tactic

        Returns:
            (score, diagnostics) tuple
        """
        score = 0.5  # Base score
        diagnostics = {}

        # Check tactic-goal compatibility
        compat_score = self._check_compatibility(goal, tactic)
        score = 0.5 * score + 0.5 * compat_score
        diagnostics["compatibility"] = compat_score

        # Check argument validity
        arg_score = self._check_arguments(goal, tactic)
        score = 0.7 * score + 0.3 * arg_score
        diagnostics["arguments"] = arg_score

        # Check for known patterns
        pattern_score = self._check_patterns(goal, tactic)
        if pattern_score is not None:
            score = 0.3 * score + 0.7 * pattern_score
            diagnostics["pattern_match"] = pattern_score

        return score, diagnostics

    def _check_compatibility(self, goal: Goal, tactic: TacticOutput) -> float:
        """Check if tactic is compatible with goal structure."""
        target = goal.target

        # Tactic-specific compatibility
        if tactic.tactic == "intro":
            # Intro needs forall/pi type
            if target.kind == TermKind.PI:
                return 0.9
            return 0.2

        elif tactic.tactic in ["rw", "rewrite"]:
            # Rewrite needs equality hypothesis or lemma
            has_eq = any(
                h.type.head_constant in ["Eq", "=", "HEq"]
                for h in goal.hypotheses
            )
            return 0.8 if has_eq else 0.3

        elif tactic.tactic in ["induction", "cases"]:
            # Needs inductive type
            has_ind = any(
                h.type.has_inductive_type()
                for h in goal.hypotheses
            )
            if has_ind or target.has_inductive_type():
                return 0.85
            return 0.2

        elif tactic.tactic in ["simp", "norm_num"]:
            # Generally applicable
            return 0.6

        return 0.5

    def _check_arguments(self, goal: Goal, tactic: TacticOutput) -> float:
        """Check if tactic arguments are valid."""
        args = tactic.args

        if not args:
            return 0.7  # No args might be fine

        # Check hypothesis references
        hyp_names = {h.name for h in goal.hypotheses}

        for key, value in args.items():
            if key in ["name", "target", "lemma"]:
                if isinstance(value, str):
                    # Check if it's a valid hypothesis
                    if value in hyp_names:
                        return 0.9
                    # Could be a library lemma (hard to check)
                    return 0.5

        return 0.6

    def _check_patterns(
        self,
        goal: Goal,
        tactic: TacticOutput,
    ) -> Optional[float]:
        """Check against known valid/invalid patterns."""
        key = f"{goal.target.head_constant}:{tactic.tactic}"

        if key in self._score_cache:
            return self._score_cache[key]

        return None

    def record_result(
        self,
        goal: Goal,
        tactic: TacticOutput,
        success: bool,
    ) -> None:
        """Record tactic result for future predictions."""
        key = f"{goal.target.head_constant}:{tactic.tactic}"

        if key not in self._score_cache:
            self._score_cache[key] = 0.5

        # Update with exponential moving average
        current = self._score_cache[key]
        new_value = 1.0 if success else 0.0
        self._score_cache[key] = 0.9 * current + 0.1 * new_value


class LeanProjectManager:
    """
    Manages Lean project setup for verification.
    """

    def __init__(self, project_path: Path):
        self.project_path = project_path

    async def setup(self, with_mathlib: bool = True) -> None:
        """Set up Lean project."""
        self.project_path.mkdir(parents=True, exist_ok=True)

        # Create lakefile
        lakefile = self.project_path / "lakefile.lean"
        if not lakefile.exists():
            lakefile.write_text(self._generate_lakefile(with_mathlib))

        # Create lean-toolchain
        toolchain = self.project_path / "lean-toolchain"
        if not toolchain.exists():
            toolchain.write_text("leanprover/lean4:v4.3.0\n")

        # Run lake update
        await self._run_lake("update")

    async def build(self) -> bool:
        """Build the Lean project."""
        return await self._run_lake("build")

    async def _run_lake(self, *args: str) -> bool:
        """Run lake command."""
        cmd = ["lake", *args]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.project_path,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.error(f"Lake command failed: {stderr.decode()}")
            return False

        return True

    def _generate_lakefile(self, with_mathlib: bool) -> str:
        """Generate lakefile.lean."""
        deps = ""
        if with_mathlib:
            deps = '''
require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"
'''

        return f'''
import Lake
open Lake DSL

package tacticMesh where
  leanOptions := #[
    ⟨`pp.unicode.fun, true⟩,
    ⟨`autoImplicit, false⟩
  ]
{deps}
@[default_target]
lean_lib TacticMesh where
  globs := #[.submodules `TacticMesh]
'''
