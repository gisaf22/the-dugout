#!/usr/bin/env python
"""Transfer-IN Recommendation — CLI wrapper for decision function.

Decision Rule (Frozen): Transfer-IN = argmax(expected_points)
Validated by research pipeline Stage 7a.

CONTRACT: This script MUST call get_transfer_recommendations()
from dugout.production.decisions. No alternate execution paths allowed.

Usage:
    PYTHONPATH=src python scripts/transfer.py --gw 24
    PYTHONPATH=src python scripts/transfer.py --gw 24 --owned my_team.csv --top 10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from dugout.production.decisions import get_transfer_recommendations


def load_owned_players(path: str | None) -> set[int]:
    """Load owned player IDs from CSV file."""
    if path is None:
        return set()
    
    df = pd.read_csv(path)
    # Handle various column names
    if "player_id" in df.columns:
        return set(df["player_id"].tolist())
    elif len(df.columns) == 1:
        return set(df.iloc[:, 0].tolist())
    else:
        raise ValueError(f"Cannot determine player_id column in {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate transfer-in recommendations")
    parser.add_argument("--gw", type=int, default=None, help="Target GW to predict (default: next GW)")
    parser.add_argument("--owned", type=str, default=None, help="CSV file with owned player_ids")
    parser.add_argument("--top", type=int, default=5, help="Number of recommendations (default: 5)")
    args = parser.parse_args()
    
    # Load owned players from file (if provided)
    owned_ids = load_owned_players(args.owned)
    
    # Use shared decision function (contract enforced)
    try:
        recommendations, target_gw, model_type = get_transfer_recommendations(
            gw=args.gw,
            top_n=args.top,
            exclude_ids=owned_ids
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}")
        return 1
    
    # Top pick
    best = recommendations.iloc[0]
    
    # Output
    print("\n" + "=" * 70)
    print(f"GW{target_gw} TRANSFER-IN RECOMMENDATIONS")
    print("=" * 70)
    
    print("\n" + "-" * 70)
    print(f"🎯 TOP TRANSFER-IN: {best['player_name']}")
    print(f"   Expected Points: {best['predicted_points']:.2f}")
    print(f"   Price: £{best['now_cost']:.1f}m")
    print("-" * 70)
    
    print(f"\nTop {len(recommendations)} Candidates:")
    print("-" * 60)
    for i, (_, row) in enumerate(recommendations.iterrows(), 1):
        marker = "⬆️" if i == 1 else "  "
        name = row.get("player_name", "Unknown")
        team = row.get("team_name", "")
        pts = row["predicted_points"]
        price = row["now_cost"]  # Already in millions from feature builder
        print(f"{marker} {i}. {name:22} ({team:12}) {pts:.2f} pts  £{price:.1f}m")
    
    print("\n" + "-" * 70)
    print("Decision Rule:")
    print("  argmax(expected_points)")
    print("  (Regret-validated, Stage 7a)")
    print("-" * 70)
    
    print("\n📋 Interpretation Guidance:")
    print("  • Small differences (<0.5 pts) = noise")
    print("  • Consider price/value only for budget constraints")
    print("  • Override only for: confirmed injury, suspension, verified news")
    print("=" * 70 + "\n")
    
    # Save to CSV
    output_path = Path(f"storage/production/reports/gw{target_gw}_transfer_in.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_df = recommendations[["player_id", "player_name", "team_name", "predicted_points"]].copy()
    output_df["rank"] = range(1, len(output_df) + 1)
    output_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
