#!/usr/bin/env python
"""Captain Recommendation — CLI wrapper for decision function.

Decision Rule (Frozen): Captain = argmax(expected_points)
Validated by research pipeline Stage 6d.

CONTRACT: This script MUST call get_captain_candidates() and pick_captain()
from dugout.production.decisions. No alternate execution paths allowed.

Usage:
    PYTHONPATH=src python scripts/decisions/captain_cli.py --gw 23
    PYTHONPATH=src python scripts/decisions/captain_cli.py  # defaults to next GW
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dugout.production.decisions import get_captain_candidates, pick_captain


def main():
    parser = argparse.ArgumentParser(description="Generate captain recommendation")
    parser.add_argument("--gw", type=int, default=None, help="Target GW to predict (default: next GW)")
    args = parser.parse_args()
    
    # Use shared decision function (contract enforced)
    try:
        candidates, target_gw, model_type = get_captain_candidates(gw=args.gw, top_n=10)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"ERROR: {e}")
        return 1
    
    # Show data context
    history_gw = target_gw - 1
    
    # Apply frozen decision rule
    captain = pick_captain(candidates)
    
    # Output
    print("\n" + "=" * 70)
    print(f"GW{target_gw} CAPTAIN RECOMMENDATION")
    print(f"Model: {model_type} | Using data through GW{history_gw}")
    print("=" * 70)
    
    print("\n" + "-" * 70)
    print(f"🎯 CAPTAIN: {captain['player_name']}")
    print(f"   Expected Points: {captain['predicted_points']:.2f}")
    print("-" * 70)
    
    print("\nTop 5 Candidates:")
    print("-" * 50)
    for i, (_, row) in enumerate(candidates.head(5).iterrows(), 1):
        marker = "👑" if i == 1 else "  "
        name = row.get("player_name", "Unknown")
        team = row.get("team_name", "")
        pts = row["predicted_points"]
        print(f"{marker} {i}. {name:22} ({team:12}) {pts:.2f} pts")
    
    print("\n" + "-" * 70)
    print("Decision Rule:")
    print("  argmax(expected_points)")
    print("  (Regret-validated, single-GW)")
    print("-" * 70)
    
    print("\n📋 Interpretation Guidance:")
    print("  • Small differences (<0.5 pts) = noise")
    print("  • Override only for: confirmed injury, suspension, verified news")
    print("  • This is decision-support, not a guarantee")
    print("=" * 70 + "\n")
    
    # Save to CSV (includes model_type column)
    output_path = Path(f"storage/production/reports/gw{target_gw}_captain.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.head(5).to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
