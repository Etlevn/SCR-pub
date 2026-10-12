"""
Data preparation module for Hybrid Recommender methods.

This module handles data preparation for hybrid models that combine
CF and ML approaches.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from config import hy_config as config
from prep.cf_prep import CFDataset, load_cf_data
from prep.ml_prep import load_and_prepare_splits


class HybridDataset:
    """
    Container for Hybrid training/test data combining CF and ML data.
    """
    
    def __init__(
        self,
        cf_dataset: CFDataset,
        ml_train_df: pd.DataFrame,
        ml_test_df: pd.DataFrame,
        feature_columns: list,
    ):
        self.cf_dataset = cf_dataset
        self.ml_train_df = ml_train_df
        self.ml_test_df = ml_test_df
        self.feature_columns = feature_columns
        
        # Statistics
        self.n_users = cf_dataset.n_users
        self.n_items = cf_dataset.n_items
        self.n_features = len(feature_columns)
    
    def get_user_interaction_count(self, accid: str) -> int:
        """Get number of interactions for a user."""
        user_idx = self.cf_dataset.get_user_idx(accid)
        if user_idx is None:
            return 0
        return self.cf_dataset.interaction_matrix[user_idx].nnz
    
    def is_cold_start_user(self, accid: str) -> bool:
        """Check if user is cold-start (not in CF training set)."""
        return self.cf_dataset.is_cold_start_user(accid)
    
    def print_statistics(self) -> None:
        """Print dataset statistics."""
        print("\n=== Hybrid Dataset Statistics ===")
        print(f"CF users                     : {self.n_users:,}")
        print(f"CF items                     : {self.n_items:,}")
        print(f"CF interactions              : {self.cf_dataset.n_interactions:,}")
        print(f"CF matrix sparsity           : {self.cf_dataset.sparsity:.4%}")
        print(f"ML features                  : {self.n_features}")
        print(f"ML train size                : {len(self.ml_train_df):,}")
        print(f"ML test size                 : {len(self.ml_test_df):,}")
        
        # Analyze cold-start in test set
        test_users = self.ml_test_df['accid'].unique()
        cold_start_users = sum(1 for u in test_users if self.is_cold_start_user(str(u)))
        
        print(f"\n=== Cold-Start Analysis (Test Set) ===")
        print(f"Test users                   : {len(test_users):,}")
        print(f"Cold-start users             : {cold_start_users:,} ({cold_start_users/len(test_users):.2%})")
        
        # Analyze interaction distribution
        test_groups = self.ml_test_df.groupby(config.GROUP_COLUMNS)
        n_test_groups = len(test_groups)
        
        cold_start_groups = 0
        low_interaction_groups = 0
        
        for group_key, group_df in test_groups:
            accid = str(group_key[0])
            if self.is_cold_start_user(accid):
                cold_start_groups += 1
            elif self.get_user_interaction_count(accid) < config.SWITCHING_INTERACTION_THRESHOLD:
                low_interaction_groups += 1
        
        print(f"\n=== Test Set Groups ===")
        print(f"Total test groups            : {n_test_groups:,}")
        print(f"Cold-start groups            : {cold_start_groups:,} ({cold_start_groups/n_test_groups:.2%})")
        print(f"Low-interaction groups       : {low_interaction_groups:,} ({low_interaction_groups/n_test_groups:.2%})")
        print(f"Warm-start groups            : {n_test_groups - cold_start_groups:,} ({(n_test_groups - cold_start_groups)/n_test_groups:.2%})")


def load_hybrid_data() -> HybridDataset:
    """
    Load and prepare data for Hybrid models.
    
    This function:
    1. Loads CF data (interaction matrix)
    2. Loads ML data (features)
    3. Combines them into HybridDataset
    
    Returns:
        HybridDataset object containing both CF and ML data
    """
    print("Loading CF data...")
    cf_dataset = load_cf_data()
    
    print("\nLoading ML data...")
    ml_train_df, ml_test_df = load_and_prepare_splits()
    
    # Get feature columns from config
    from config import ml_config
    feature_columns = ml_config.FEATURE_COLUMNS
    
    print(f"ML train size: {len(ml_train_df):,}, ML test size: {len(ml_test_df):,}")
    
    # Create HybridDataset object
    dataset = HybridDataset(
        cf_dataset=cf_dataset,
        ml_train_df=ml_train_df,
        ml_test_df=ml_test_df,
        feature_columns=feature_columns,
    )
    
    # Print statistics
    dataset.print_statistics()
    
    return dataset


if __name__ == "__main__":
    """
    Test the Hybrid data preparation pipeline.
    """
    print("="*60)
    print("Testing Hybrid Data Preparation")
    print("="*60)
    
    dataset = load_hybrid_data()
    
    print("\n" + "="*60)
    print("Hybrid Data Preparation Test Completed!")
    print("="*60)

