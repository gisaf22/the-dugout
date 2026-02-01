"""CLI smoke tests for decision scripts.

These are minimal tests that verify:
1. Script runs without crashing
2. Exit code is 0
3. Output exists (stdout or file)

These tests do NOT verify correctness - that's the job of contract tests.
CLI scripts are thin wrappers around decision functions.
"""

import subprocess
import sys
from pathlib import Path

import pytest


# Project root for PYTHONPATH
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def run_script(script_path: Path, args: list = None) -> subprocess.CompletedProcess:
    """Run a script with PYTHONPATH set to src."""
    env = {
        "PYTHONPATH": str(PROJECT_ROOT / "src"),
    }
    
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={**subprocess.os.environ, **env},
        cwd=str(PROJECT_ROOT),
        timeout=120,  # 2 minute timeout
    )


class TestCaptainCLI:
    """Smoke tests for captain CLI script."""

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent.parent / "storage" / "fpl_2025_26.sqlite").exists(),
        reason="Database not available"
    )
    def test_captain_cli_runs_and_exits_zero(self):
        """Captain CLI should run and exit with code 0."""
        script = SCRIPTS_DIR / "decisions" / "captain_cli.py"
        if not script.exists():
            pytest.skip("captain_cli.py not found")
        
        result = run_script(script, ["--gw", "23"])
        
        assert result.returncode == 0, f"Exit code {result.returncode}, stderr: {result.stderr}"

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent.parent / "storage" / "fpl_2025_26.sqlite").exists(),
        reason="Database not available"
    )
    def test_captain_cli_produces_output(self):
        """Captain CLI should produce stdout output."""
        script = SCRIPTS_DIR / "decisions" / "captain_cli.py"
        if not script.exists():
            pytest.skip("captain_cli.py not found")
        
        result = run_script(script, ["--gw", "23"])
        
        # Should have some output
        assert len(result.stdout) > 0 or len(result.stderr) > 0


class TestTransferCLI:
    """Smoke tests for transfer CLI script."""

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent.parent / "storage" / "fpl_2025_26.sqlite").exists(),
        reason="Database not available"
    )
    def test_transfer_cli_runs_and_exits_zero(self):
        """Transfer CLI should run and exit with code 0."""
        script = SCRIPTS_DIR / "decisions" / "transfer_cli.py"
        if not script.exists():
            pytest.skip("transfer_cli.py not found")
        
        result = run_script(script, ["--gw", "23"])
        
        assert result.returncode == 0, f"Exit code {result.returncode}, stderr: {result.stderr}"

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent.parent / "storage" / "fpl_2025_26.sqlite").exists(),
        reason="Database not available"
    )
    def test_transfer_cli_produces_output(self):
        """Transfer CLI should produce stdout output."""
        script = SCRIPTS_DIR / "decisions" / "transfer_cli.py"
        if not script.exists():
            pytest.skip("transfer_cli.py not found")
        
        result = run_script(script, ["--gw", "23"])
        
        assert len(result.stdout) > 0 or len(result.stderr) > 0


class TestFreeHitCLI:
    """Smoke tests for Free Hit CLI script."""

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent.parent / "storage" / "fpl_2025_26.sqlite").exists(),
        reason="Database not available"
    )
    def test_free_hit_cli_runs_and_exits_zero(self):
        """Free Hit CLI should run and exit with code 0."""
        script = SCRIPTS_DIR / "decisions" / "free_hit_cli.py"
        if not script.exists():
            pytest.skip("free_hit_cli.py not found")
        
        result = run_script(script, ["--gw", "23"])
        
        assert result.returncode == 0, f"Exit code {result.returncode}, stderr: {result.stderr}"

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent.parent / "storage" / "fpl_2025_26.sqlite").exists(),
        reason="Database not available"
    )
    def test_free_hit_cli_produces_output(self):
        """Free Hit CLI should produce stdout output."""
        script = SCRIPTS_DIR / "decisions" / "free_hit_cli.py"
        if not script.exists():
            pytest.skip("free_hit_cli.py not found")
        
        result = run_script(script, ["--gw", "23"])
        
        assert len(result.stdout) > 0 or len(result.stderr) > 0


class TestCLIHelpOutput:
    """Test that CLI scripts have help output."""

    def test_captain_cli_has_help(self):
        """Captain CLI should have --help option."""
        script = SCRIPTS_DIR / "decisions" / "captain_cli.py"
        if not script.exists():
            pytest.skip("captain_cli.py not found")
        
        result = run_script(script, ["--help"])
        
        # Help should exit 0 and produce output
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "gw" in result.stdout.lower()

    def test_transfer_cli_has_help(self):
        """Transfer CLI should have --help option."""
        script = SCRIPTS_DIR / "decisions" / "transfer_cli.py"
        if not script.exists():
            pytest.skip("transfer_cli.py not found")
        
        result = run_script(script, ["--help"])
        
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "gw" in result.stdout.lower()

    def test_free_hit_cli_has_help(self):
        """Free Hit CLI should have --help option."""
        script = SCRIPTS_DIR / "run_free_hit.py"
        if not script.exists():
            pytest.skip("run_free_hit.py not found")
        
        result = run_script(script, ["--help"])
        
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "gw" in result.stdout.lower()
