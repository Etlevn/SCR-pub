"""
Data preparation module for Collaborative Filtering methods.

This module handles:
1. Extracting user-item interactions from raw data
2. Building sparse interaction matrices
3. Creating user/item ID mappings
4. Handling train/test splits for CF
5. Detecting and reporting cold-start cases
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from tqdm import tqdm

from config import cf_config as config_cf
from config import ml_config as config
from data.read_master_final_sampled import load_master_sampled


class CFDataset:
    """
    Container for CF training/test data with user-item mappings.
    """
    
    def __init__(
        self,
        interaction_matrix: csr_matrix,
        user_to_idx: Dict[str, int],
        item_to_idx: Dict[int, int],
        idx_to_user: Dict[int, str],
        idx_to_item: Dict[int, int],
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ):
        self.interaction_matrix = interaction_matrix
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.idx_to_user = idx_to_user
        self.idx_to_item = idx_to_item
        self.train_df = train_df
        self.test_df = test_df
        
        # Statistics
        self.n_users = len(user_to_idx)
        self.n_items = len(item_to_idx)
        self.n_interactions = interaction_matrix.nnz
        self.sparsity = 1 - (self.n_interactions / (self.n_users * self.n_items))
    
    def get_user_idx(self, accid: str) -> int | None:
        """Get user index for a given accid. Returns None if not found."""
        return self.user_to_idx.get(accid)
    
    def get_item_idx(self, permno: float) -> int | None:
        """Get item index for a given permno. Returns None if not found."""
        return self.item_to_idx.get(permno)
    
    def is_cold_start_user(self, accid: str) -> bool:
        """Check if user is cold-start (not in training set)."""
        return accid not in self.user_to_idx
    
    def is_cold_start_item(self, permno: float) -> bool:
        """Check if item is cold-start (not in training set)."""
        return permno not in self.item_to_idx
    
    def print_statistics(self) -> None:
        """Print dataset statistics."""
        print("\n=== CF Dataset Statistics ===")
        print(f"Number of users (accid)      : {self.n_users:,}")
        print(f"Number of items (permno)     : {self.n_items:,}")
        print(f"Number of interactions       : {self.n_interactions:,}")
        print(f"Matrix sparsity              : {self.sparsity:.4%}")
        print(f"Avg interactions per user    : {self.n_interactions / self.n_users:.2f}")
        print(f"Avg interactions per item    : {self.n_interactions / self.n_items:.2f}")
        
        # Analyze cold-start in test set
        test_users = self.test_df['accid'].unique()
        test_items = self.test_df['permno'].unique()
        
        cold_start_users = sum(1 for u in test_users if self.is_cold_start_user(u))
        cold_start_items = sum(1 for i in test_items if self.is_cold_start_item(i))
        
        print(f"\n=== Cold-Start Analysis (Test Set) ===")
        print(f"Test users                   : {len(test_users):,}")
        print(f"Cold-start users             : {cold_start_users:,} ({cold_start_users/len(test_users):.2%})")
        print(f"Test items                   : {len(test_items):,}")
        print(f"Cold-start items             : {cold_start_items:,} ({cold_start_items/len(test_items):.2%})")
        
        # Analyze test set groups
        test_groups = self.test_df.groupby(config_cf.GROUP_COLUMNS)
        n_test_groups = len(test_groups)
        
        cold_start_groups = 0
        for group_key, group_df in test_groups:
            accid = group_key[0]
            if self.is_cold_start_user(accid):
                cold_start_groups += 1
        
        print(f"\n=== Test Set Groups ===")
        print(f"Total test groups            : {n_test_groups:,}")
        print(f"Cold-start groups            : {cold_start_groups:,} ({cold_start_groups/n_test_groups:.2%})")
        print(f"Warm-start groups            : {n_test_groups - cold_start_groups:,} ({(n_test_groups - cold_start_groups)/n_test_groups:.2%})")


def _extract_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract positive interactions from the dataframe.
    
    An interaction is defined as (accid, now_permno, date) where the user
    actually traded the stock on that date.
    
    Args:
        df: DataFrame with columns [accid, date, now_permno, permno]
    
    Returns:
        DataFrame with columns [accid, permno, date, count]
    """
    # Filter to only rows where permno == now_permno (positive interactions)
    interactions = df[df['permno'] == df['now_permno']].copy()
    
    # Select relevant columns
    interactions = interactions[['accid', 'now_permno', 'date']].copy()
    interactions.rename(columns={'now_permno': 'permno'}, inplace=True)
    
    # Group by (accid, permno) and count interactions
    if config_cf.INTERACTION_WEIGHT == "binary":
        # Binary: 1 if interacted, 0 otherwise
        interactions['count'] = 1
        interactions = interactions.groupby(['accid', 'permno'], as_index=False).agg({
            'count': 'max',  # Binary: max is 1
            'date': 'max'    # Keep most recent date
        })
    elif config_cf.INTERACTION_WEIGHT == "count":
        # Count: number of interactions
        interactions['count'] = 1
        interactions = interactions.groupby(['accid', 'permno'], as_index=False).agg({
            'count': 'sum',  # Sum of interactions
            'date': 'max'    # Keep most recent date
        })
    elif config_cf.INTERACTION_WEIGHT == "time_decay":
        # Time decay: weight by recency
        # This will be handled later when we know the time range
        interactions['count'] = 1
        interactions = interactions.groupby(['accid', 'permno'], as_index=False).agg({
            'count': 'sum',
            'date': 'max'
        })
    else:
        raise ValueError(f"Unknown INTERACTION_WEIGHT: {config_cf.INTERACTION_WEIGHT}")
    
    return interactions


def _apply_time_decay(interactions: pd.DataFrame, max_date: pd.Timestamp) -> pd.DataFrame:
    """
    Apply exponential time decay to interaction weights.
    
    More recent interactions get higher weights:
    weight = count * exp(-decay_rate * days_since_interaction)
    
    Args:
        interactions: DataFrame with columns [accid, permno, date, count]
        max_date: Reference date (usually the last date in training set)
    
    Returns:
        DataFrame with updated 'count' column
    """
    if config_cf.INTERACTION_WEIGHT != "time_decay":
        return interactions
    
    interactions = interactions.copy()
    
    # Calculate days since interaction
    interactions['days_since'] = (max_date - interactions['date']).dt.days
    
    # Apply exponential decay
    decay_rate = config_cf.TIME_DECAY_RATE
    interactions['count'] = interactions['count'] * np.exp(-decay_rate * interactions['days_since'] / 365.0)
    
    # Drop temporary column
    interactions.drop(columns=['days_since'], inplace=True)
    
    return interactions


def _build_interaction_matrix(
    interactions: pd.DataFrame,
    user_to_idx: Dict[str, int],
    item_to_idx: Dict[int, int],
) -> csr_matrix:
    """
    Build sparse user-item interaction matrix.
    
    Args:
        interactions: DataFrame with columns [accid, permno, count]
        user_to_idx: Mapping from accid to user index
        item_to_idx: Mapping from permno to item index
    
    Returns:
        Sparse matrix of shape (n_users, n_items)
    """
    # Map accid and permno to indices
    user_indices = interactions['accid'].map(user_to_idx).values
    item_indices = interactions['permno'].map(item_to_idx).values
    values = interactions['count'].values
    
    # Filter out any unmapped entries (shouldn't happen, but be safe)
    valid_mask = ~(pd.isna(user_indices) | pd.isna(item_indices))
    user_indices = user_indices[valid_mask].astype(int)
    item_indices = item_indices[valid_mask].astype(int)
    values = values[valid_mask]
    
    n_users = len(user_to_idx)
    n_items = len(item_to_idx)
    
    # Create COO matrix and convert to CSR for efficient operations
    matrix = coo_matrix(
        (values, (user_indices, item_indices)),
        shape=(n_users, n_items),
        dtype=np.float32
    )
    
    return matrix.tocsr()


def _create_mappings(
    train_interactions: pd.DataFrame,
) -> Tuple[Dict[str, int], Dict[int, int], Dict[int, str], Dict[int, int]]:
    """
    Create bidirectional mappings between accid/permno and matrix indices.
    
    Args:
        train_interactions: DataFrame with columns [accid, permno, ...]
    
    Returns:
        Tuple of (user_to_idx, item_to_idx, idx_to_user, idx_to_item)
    """
    # Get unique users and items from training set
    unique_users = sorted(train_interactions['accid'].unique())
    unique_items = sorted(train_interactions['permno'].unique())
    
    # Create mappings
    user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
    
    idx_to_user = {idx: user for user, idx in user_to_idx.items()}
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}
    
    return user_to_idx, item_to_idx, idx_to_user, idx_to_item


def _split_by_date(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into train and test sets by date.
    
    This is the same splitting logic as in prep.py, but we keep it here
    for CF-specific preprocessing.
    
    For each accid, dates are sorted chronologically and divided according
    to TRAIN_RATIO.
    """
    date_col = config.DATE_COLUMN
    accid_col = "accid"
    train_ratio = config.TRAIN_RATIO
    
    if not (0 < train_ratio < 1):
        raise ValueError(f"TRAIN_RATIO must be between 0 and 1, got {train_ratio}")
    
    df = df.copy()
    df["_split"] = None
    
    unique_accids = df[accid_col].unique()
    
    for accid in tqdm(unique_accids, desc="Splitting data by date", unit="accid", leave=False):
        accid_mask = df[accid_col] == accid
        accid_subset = df.loc[accid_mask]
        
        accid_dates = accid_subset[date_col].unique()
        accid_dates_sorted = sorted(accid_dates)
        
        n_dates = len(accid_dates_sorted)
        if n_dates == 0:
            continue
        
        train_end_idx = int(n_dates * train_ratio)
        
        if train_end_idx == 0 and n_dates > 0:
            train_end_idx = 1
        if train_end_idx >= n_dates and n_dates > 0:
            train_end_idx = n_dates - 1
        
        train_dates = set(accid_dates_sorted[:train_end_idx])
        test_dates = set(accid_dates_sorted[train_end_idx:])
        
        date_values = df.loc[accid_mask, date_col]
        train_mask = date_values.isin(train_dates)
        test_mask = date_values.isin(test_dates)
        
        df.loc[accid_mask & train_mask, "_split"] = "train"
        df.loc[accid_mask & test_mask, "_split"] = "test"
        
        remaining_mask = accid_mask & df["_split"].isna()
        df.loc[remaining_mask, "_split"] = "train"
    
    train_df = df[df["_split"] == "train"].drop(columns=["_split"]).copy()
    test_df = df[df["_split"] == "test"].drop(columns=["_split"]).copy()
    
    # Add label column for evaluation
    train_df['label'] = (train_df['permno'] == train_df['now_permno']).astype(int)
    test_df['label'] = (test_df['permno'] == test_df['now_permno']).astype(int)
    
    return train_df, test_df


def load_cf_data() -> CFDataset:
    """
    Load and prepare data for Collaborative Filtering.
    
    This function:
    1. Loads raw data
    2. Splits into train/test by date
    3. Extracts positive interactions from training set
    4. Builds user-item interaction matrix
    5. Creates ID mappings
    6. Returns CFDataset object
    
    Returns:
        CFDataset object containing all necessary data for CF training/evaluation
    """
    print("Loading raw data...")
    raw_df = load_master_sampled()
    
    # Convert date column to datetime
    raw_df[config.DATE_COLUMN] = pd.to_datetime(raw_df[config.DATE_COLUMN])
    
    # Convert accid to string for consistency
    raw_df['accid'] = raw_df['accid'].astype(str)
    
    print("Splitting data by date...")
    train_df, test_df = _split_by_date(raw_df)
    
    print(f"Train size: {len(train_df):,}, Test size: {len(test_df):,}")
    
    print("Extracting interactions from training set...")
    train_interactions = _extract_interactions(train_df)
    
    print(f"Found {len(train_interactions):,} unique (user, item) interactions")
    
    # Apply time decay if configured
    if config_cf.INTERACTION_WEIGHT == "time_decay":
        print("Applying time decay to interactions...")
        max_train_date = train_df[config.DATE_COLUMN].max()
        train_interactions = _apply_time_decay(train_interactions, max_train_date)
    
    print("Creating user/item mappings...")
    user_to_idx, item_to_idx, idx_to_user, idx_to_item = _create_mappings(train_interactions)
    
    print(f"Number of users: {len(user_to_idx):,}")
    print(f"Number of items: {len(item_to_idx):,}")
    
    print("Building interaction matrix...")
    interaction_matrix = _build_interaction_matrix(
        train_interactions,
        user_to_idx,
        item_to_idx,
    )
    
    print(f"Matrix shape: {interaction_matrix.shape}")
    print(f"Matrix sparsity: {1 - interaction_matrix.nnz / (interaction_matrix.shape[0] * interaction_matrix.shape[1]):.4%}")
    
    # Create CFDataset object
    dataset = CFDataset(
        interaction_matrix=interaction_matrix,
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        idx_to_user=idx_to_user,
        idx_to_item=idx_to_item,
        train_df=train_df,
        test_df=test_df,
    )
    
    # Print statistics
    dataset.print_statistics()
    
    return dataset


def get_item_popularity(dataset: CFDataset) -> Dict[int, float]:
    """
    Calculate item popularity scores from training interactions.
    
    Popularity is defined as the number of unique users who interacted with the item.
    This is used for cold-start handling.
    
    Args:
        dataset: CFDataset object
    
    Returns:
        Dictionary mapping item index to popularity score (normalized to [0, 1])
    """
    # Sum over users (axis=0) to get item interaction counts
    item_counts = np.array(dataset.interaction_matrix.sum(axis=0)).flatten()
    
    # Normalize to [0, 1]
    max_count = item_counts.max()
    if max_count > 0:
        item_popularity = item_counts / max_count
    else:
        item_popularity = item_counts
    
    # Convert to dictionary
    popularity_dict = {idx: float(pop) for idx, pop in enumerate(item_popularity)}
    
    return popularity_dict


if __name__ == "__main__":
    """
    Test the CF data preparation pipeline.
    """
    print("="*60)
    print("Testing CF Data Preparation")
    print("="*60)
    
    dataset = load_cf_data()
    
    print("\n" + "="*60)
    print("Computing item popularity...")
    print("="*60)
    
    popularity = get_item_popularity(dataset)
    
    # Show top 10 most popular items
    sorted_items = sorted(popularity.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 most popular items:")
    for idx, (item_idx, pop) in enumerate(sorted_items[:10], 1):
        permno = dataset.idx_to_item[item_idx]
        print(f"{idx}. permno={permno}, popularity={pop:.4f}")
    
    print("\n" + "="*60)
    print("CF Data Preparation Test Completed!")
    print("="*60)

