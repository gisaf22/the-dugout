"""
Research Pipeline Runner — Reproducibility Layer

Sequentially invokes all frozen pipeline stages and generates a consolidated
evidence report.

This is orchestration only — no modeling code is modified.

Usage:
    python -m dugout.research.pipeline.run_all

Stages Executed:
    Stage 2:  targets.py               → targets.csv
    Stage 3:  features_participation.py → features_participation.csv
    Stage 4a: features_performance.py   → features_performance.csv
    Stage 5:  belief_models.py          → beliefs.csv, models/*.pkl
    Stage 6a: stage_6a_captain_policy.py → evaluation_captain.csv
    Stage 7a: stage_7a_transfer_in.py    → evaluation_transfer_in.csv
    Stage 8a: stage_8a_multigw_beliefs.py → beliefs_multigw.csv
    Stage 8b: stage_8b_multigw_hold.py   → evaluation_multigw_hold.csv
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple


class StageResult(NamedTuple):
    """Result of a stage execution."""
    name: str
    success: bool
    duration_sec: float
    message: str


def get_project_root() -> Path:
    """Get project root from this file's location."""
    # pipeline/run_all.py -> pipeline -> research -> dugout -> src -> project_root
    return Path(__file__).parent.parent.parent.parent.parent


def run_stage(stage_name: str, module_path: str, project_root: Path) -> StageResult:
    """
    Run a single pipeline stage as a subprocess.
    
    Args:
        stage_name: Human-readable stage name for logging
        module_path: Python module path (e.g., dugout.pipeline.targets)
        project_root: Project root for working directory
    
    Returns:
        StageResult with success status and timing
    """
    start = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", module_path],
            cwd=project_root,
            env={**subprocess.os.environ, "PYTHONPATH": str(project_root / "src")},
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout per stage
        )
        
        duration = time.time() - start
        
        if result.returncode == 0:
            return StageResult(stage_name, True, duration, "OK")
        else:
            error_msg = result.stderr.strip()[-200:] if result.stderr else "Unknown error"
            return StageResult(stage_name, False, duration, error_msg)
            
    except subprocess.TimeoutExpired:
        return StageResult(stage_name, False, 300.0, "Timeout after 5 minutes")
    except Exception as e:
        return StageResult(stage_name, False, time.time() - start, str(e))


def run_pipeline(verbose: bool = True) -> list[StageResult]:
    """
    Execute all pipeline stages sequentially.
    
    Fails fast if any stage errors.
    
    Args:
        verbose: Print progress to stdout
    
    Returns:
        List of StageResult for all executed stages
    """
    project_root = get_project_root()
    
    # Stage definitions: (display_name, module_path)
    stages = [
        ("Stage 2: Targets", "dugout.research.pipeline.targets"),
        ("Stage 3: Participation Features", "dugout.research.pipeline.features_participation"),
        ("Stage 4a: Performance Features", "dugout.research.pipeline.features_performance"),
        ("Stage 5: Belief Models", "dugout.research.pipeline.belief_models"),
        ("Stage 6a: Captain Evaluation", "dugout.research.pipeline.stage_6a_captain_policy"),
        ("Stage 7a: Transfer-IN Evaluation", "dugout.research.pipeline.stage_7a_transfer_in"),
        ("Stage 8a: Multi-GW Beliefs", "dugout.research.pipeline.stage_8a_multigw_beliefs"),
        ("Stage 8b: Multi-GW Hold Evaluation", "dugout.research.pipeline.stage_8b_multigw_hold"),
    ]
    
    results = []
    
    if verbose:
        print("=" * 60)
        print("RESEARCH PIPELINE RUNNER")
        print("=" * 60)
        print()
    
    for stage_name, module_path in stages:
        if verbose:
            print(f"[{stage_name}] Running...", end=" ", flush=True)
        
        result = run_stage(stage_name, module_path, project_root)
        results.append(result)
        
        if verbose:
            if result.success:
                print(f"✓ ({result.duration_sec:.1f}s)")
            else:
                print(f"✗ FAILED")
                print(f"   Error: {result.message}")
        
        # Fail fast
        if not result.success:
            if verbose:
                print()
                print("Pipeline aborted due to stage failure.")
            return results
    
    if verbose:
        print()
        print("-" * 60)
        total_time = sum(r.duration_sec for r in results)
        print(f"All stages complete in {total_time:.1f}s")
        print()
    
    # Generate report
    if verbose:
        print("[Report] Building research_report.json...", end=" ", flush=True)
    
    try:
        from dugout.research.pipeline.report_builder import build_report
        report_path = build_report(project_root)
        if verbose:
            print(f"✓")
            print(f"   Output: {report_path}")
    except Exception as e:
        if verbose:
            print(f"✗ FAILED")
            print(f"   Error: {e}")
    
    if verbose:
        print()
        print("=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
    
    return results


if __name__ == "__main__":
    run_pipeline(verbose=True)
