"""Train/Validation/Test dataset creation for FPL ML training.

Provides time-aware dataset splitting based on gameweek counts to prevent
temporal data leakage. Essential for fair model evaluation.

Key Classes:
    DatasetBuilder - Creates train/val/test splits
    Datasets - Container for the three splits

Split Strategy:
    - Test set: Last N gameweeks (default 4)
    - Validation set: Previous M gameweeks (default 4)
    - Train set: All remaining earlier gameweeks
    
This ensures the model is always evaluated on "future" data it hasn't seen.

Usage:
    from dugout.production.features import DatasetBuilder
    
    builder = DatasetBuilder(test_gws=4, val_gws=4)
    datasets = builder.build(df)
    
    X_train = datasets.train[features]
    y_train = datasets.train['target_points']
    
    # Or save to CSV files
    builder.build_and_save(df, out_dir='storage/datasets')
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import pandas as pd


@dataclass
class DatasetConfig:
    """Configuration for train/val/test dataset creation.
    
    Attributes:
        test_gws: Number of most recent GWs for test set.
        val_gws: Number of GWs before test for validation set.
        min_train_gws: Minimum GWs required for training set.
        gw_column: Column name containing gameweek numbers.
    """
    test_gws: int = 4
    val_gws: int = 4
    min_train_gws: int = 3
    gw_column: str = "round"


@dataclass
class Datasets:
    """Container for train/val/test datasets.
    
    Attributes:
        train: Training DataFrame.
        val: Validation DataFrame.
        test: Test DataFrame.
        train_gws: List of GWs in training set.
        val_gws: List of GWs in validation set.
        test_gws: List of GWs in test set.
    """
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_gws: List[int] = field(default_factory=list)
    val_gws: List[int] = field(default_factory=list)
    test_gws: List[int] = field(default_factory=list)
    
    @property
    def summary(self) -> Dict[str, any]:
        """Get summary statistics for the datasets."""
        return {
            "train_rows": len(self.train),
            "val_rows": len(self.val),
            "test_rows": len(self.test),
            "train_gws": self.train_gws,
            "val_gws": self.val_gws,
            "test_gws": self.test_gws,
        }
    
    def print_summary(self) -> None:
        """Print dataset summary to console."""
        print(f"Datasets Summary:")
        print(f"  Train: {len(self.train):,} rows | GWs {self.train_gws}")
        print(f"  Val:   {len(self.val):,} rows | GWs {self.val_gws}")
        print(f"  Test:  {len(self.test):,} rows | GWs {self.test_gws}")


class DatasetBuilder:
    """Time-based dataset builder for FPL data.
    
    Creates train/val/test datasets by gameweek to prevent temporal leakage:
    - Test set: Most recent N gameweeks
    - Validation set: Preceding N gameweeks
    - Training set: All remaining gameweeks
    
    Example:
        builder = DatasetBuilder(test_gws=4, val_gws=4)
        datasets = builder.build(df)
        
        # Access datasets
        X_train, y_train = datasets.train[features], datasets.train["target"]
        X_val, y_val = datasets.val[features], datasets.val["target"]
    """
    
    def __init__(
        self,
        test_gws: int = 4,
        val_gws: int = 4,
        min_train_gws: int = 3,
        gw_column: str = "round",
    ) -> None:
        """Initialize builder with gameweek counts.
        
        Args:
            test_gws: Number of GWs for test set.
            val_gws: Number of GWs for validation set.
            min_train_gws: Minimum GWs required for training.
            gw_column: Column name containing gameweek numbers (default: "round").
        """
        self.config = DatasetConfig(
            test_gws=test_gws,
            val_gws=val_gws,
            min_train_gws=min_train_gws,
            gw_column=gw_column,
        )
    
    @classmethod
    def create_from_db(
        cls,
        out_dir: str = "storage/datasets",
        test_gws: int = 4,
        val_gws: int = 4,
        db_path: str | None = None,
    ) -> Datasets:
        """Create train/val/test datasets from FPL database.
        
        Extracts gameweek data, builds features, splits by GW, and saves CSVs.
        
        Args:
            out_dir: Output directory for CSV files.
            test_gws: Number of GWs for test set.
            val_gws: Number of GWs for validation set.
            db_path: Optional path to database.
            
        Returns:
            Datasets container with the splits.
        """
        from dugout.production.data import DataReader
        from dugout.production.data import queries as Q
        from dugout.production.features import FeatureBuilder
        
        reader = DataReader(db_path)
        print(f"Database: {reader.db_path}")
        
        # Extract gameweek data
        gw_df = pd.DataFrame(reader.query(Q.TRAINING_DATA_BASE))
        print(f"Extracted {len(gw_df):,} rows, {gw_df['player_id'].nunique()} players")
        
        # Build features per player-GW
        builder = FeatureBuilder(reader=reader)
        rows = []
        
        for player_id, pdf in gw_df.groupby("player_id"):
            pdf = pdf.sort_values("gw").reset_index(drop=True)
            if len(pdf) < 2 or pdf["minutes"].sum() < 1:
                continue
            
            for i in range(1, len(pdf)):
                target = pdf.iloc[i]
                features = builder.build_for_player(pdf.iloc[:i])
                rows.append({
                    "player_id": player_id,
                    "player_name": target["player_name"],
                    "team_name": target["team_name"],
                    "team_id": target["team_id"],
                    "position": target["position"],
                    "gw": int(target["gw"]),
                    "target_points": target["total_points"],
                    **features,
                })
        
        df = pd.DataFrame(rows)
        print(f"Built {len(df):,} training rows")
        
        # Split and save
        splitter = cls(test_gws=test_gws, val_gws=val_gws, gw_column="gw")
        return splitter.build_and_save(df, out_dir=out_dir)
    
    def build(self, df: pd.DataFrame) -> Datasets:
        """Build train/val/test datasets from DataFrame.
        
        Args:
            df: DataFrame with gameweek column.
            
        Returns:
            Datasets container with train, val, test DataFrames.
            
        Raises:
            ValueError: If not enough gameweeks for requested split.
        """
        gw_col = self.config.gw_column
        
        if gw_col not in df.columns:
            raise ValueError(f"Column '{gw_col}' not found in DataFrame")
        
        # Get sorted unique gameweeks
        all_gws = sorted(df[gw_col].unique())
        n_gws = len(all_gws)
        
        required = self.config.test_gws + self.config.val_gws + self.config.min_train_gws
        if n_gws < required:
            raise ValueError(
                f"Need at least {required} GWs for datasets "
                f"(test={self.config.test_gws}, val={self.config.val_gws}, "
                f"min_train={self.config.min_train_gws}), got {n_gws}"
            )
        
        # Assign GWs to each dataset
        test_gws = all_gws[-self.config.test_gws:]
        val_gws = all_gws[-(self.config.test_gws + self.config.val_gws):-self.config.test_gws]
        train_gws = all_gws[:-(self.config.test_gws + self.config.val_gws)]
        
        # Create DataFrames
        train_df = df[df[gw_col].isin(train_gws)].copy()
        val_df = df[df[gw_col].isin(val_gws)].copy()
        test_df = df[df[gw_col].isin(test_gws)].copy()
        
        return Datasets(
            train=train_df,
            val=val_df,
            test=test_df,
            train_gws=list(train_gws),
            val_gws=list(val_gws),
            test_gws=list(test_gws),
        )
    
    def create_splits(
        self,
        df: pd.DataFrame,
    ) -> Datasets:
        """Build train/val/test datasets from DataFrame.
        
        Alias for build() with a more intuitive name.
        
        Args:
            df: DataFrame to split.
            
        Returns:
            Datasets container with the datasets.
        """
        return self.build(df)
    
    def save(
        self,
        datasets: Datasets,
        out_dir: str = "storage/datasets",
        prefix: str = "",
    ) -> None:
        """Save datasets to CSV files.
        
        Args:
            datasets: Datasets container to save.
            out_dir: Output directory for CSV files.
            prefix: Optional prefix for filenames.
        """
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        pfx = f"{prefix}_" if prefix else ""
        
        datasets.train.to_csv(out_path / f"{pfx}train.csv", index=False)
        datasets.val.to_csv(out_path / f"{pfx}val.csv", index=False)
        datasets.test.to_csv(out_path / f"{pfx}test.csv", index=False)
        
        # Save metadata
        import json
        meta = {
            "config": {
                "test_gws": self.config.test_gws,
                "val_gws": self.config.val_gws,
                "gw_column": self.config.gw_column,
            },
            "summary": datasets.summary,
        }
        with open(out_path / f"{pfx}dataset_info.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)
        
        print(f"Saved datasets to {out_path}/")
        datasets.print_summary()
    
    @classmethod
    def from_disk(
        cls,
        dataset_dir: str = "storage/datasets",
        prefix: str = "",
    ) -> Datasets:
        """Load previously saved train/val/test datasets.
        
        Args:
            dataset_dir: Directory containing dataset CSV files.
            prefix: Optional prefix used when saving.
            
        Returns:
            Datasets container with loaded DataFrames.
        """
        path = Path(dataset_dir)
        pfx = f"{prefix}_" if prefix else ""
        
        train = pd.read_csv(path / f"{pfx}train.csv")
        val = pd.read_csv(path / f"{pfx}val.csv")
        test = pd.read_csv(path / f"{pfx}test.csv")
        
        # Load metadata if available
        train_gws, val_gws, test_gws = [], [], []
        
        meta_file = path / f"{pfx}dataset_info.json"
        if meta_file.exists():
            import json
            with open(meta_file) as f:
                meta = json.load(f)
                summary = meta.get("summary", {})
                train_gws = summary.get("train_gws", [])
                val_gws = summary.get("val_gws", [])
                test_gws = summary.get("test_gws", [])
        
        return Datasets(
            train=train,
            val=val,
            test=test,
            train_gws=train_gws,
            val_gws=val_gws,
            test_gws=test_gws,
        )


def load_datasets(
    dataset_dir: str = "storage/datasets",
    prefix: str = "",
) -> Datasets:
    """Module-level wrapper for DatasetBuilder.from_disk() for backwards compatibility."""
    return DatasetBuilder.from_disk(dataset_dir, prefix)
