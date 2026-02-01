#!/usr/bin/env python
"""Free Hit Optimization — CLI wrapper for decision function.

Decision Rule (Frozen): Maximize Σ(expected_points)
Validated by research pipeline.

CONTRACT: This script MUST call optimize_free_hit()
from dugout.production.decisions. No alternate execution paths allowed.

Notes:
    - Availability weighting (p_play × points) intentionally NOT applied
    - Fixture adjustments intentionally NOT applied
    - This reflects research findings: constraints already regularize risk

Usage:
    PYTHONPATH=src python scripts/decisions/free_hit_cli.py
    PYTHONPATH=src python scripts/decisions/free_hit_cli.py --gw 24
    PYTHONPATH=src python scripts/decisions/free_hit_cli.py --budget 95
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from dugout.production.decisions import optimize_free_hit


def main():
    parser = argparse.ArgumentParser(description="Optimize Free Hit squad")
    parser.add_argument("--gw", type=int, default=None, help="Target GW to optimize for (default: next GW)")
    parser.add_argument("--budget", type=float, default=100.0, help="Budget in millions (default: 100)")
    args = parser.parse_args()
    
    # Use shared decision function (contract enforced)
    print("\n" + "=" * 70)
    print("FREE HIT OPTIMIZER")
    print("=" * 70)
    
    try:
        result, _predictions_df, target_gw, model_type = optimize_free_hit(
            gw=args.gw,
            budget=args.budget
        )
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"ERROR: {e}")
        return 1
    
    if result is None:
        print("Optimization failed!")
        return 1
    
    # Output
    print(f"\n🎯 GW{target_gw} FREE HIT SQUAD")
    print(f"Model: {model_type}")
    result.print_squad()
    
    # Save to CSV (includes model_type column)
    output_path = Path(f"storage/production/reports/gw{target_gw}_free_hit.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    squad_data = []
    for i, p in enumerate(result.starting_xi, 1):
        squad_data.append({"rank": i, "role": "XI", "model_type": model_type, **p})
    for i, p in enumerate(result.bench, 1):
        squad_data.append({"rank": 11 + i, "role": "Bench", "model_type": model_type, **p})
    pd.DataFrame(squad_data).to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    
    print("\n" + "-" * 70)
    print("Decision Rule:")
    print("  maximize Σ(predicted_points)")
    print("-" * 70)
    
    print("\n📋 Interpretation Guidance:")
    print("  • Squad is EV-maximized, not risk-adjusted")
    print("  • Bench value may be sacrificed for XI strength")
    print("  • Override only for: confirmed injury, suspension, verified news")
    print("=" * 70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
