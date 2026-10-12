from __future__ import annotations

from typing import Tuple

import pandas as pd
from tqdm import tqdm

from config import ml_config as config
from data.read_master_final_sampled import load_master_sampled


def _select_and_cast_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only ID and feature columns and perform basic dtype conversions.
    """
    # Determine feature columns dynamically in case the config is out of sync
    available_features = [c for c in config.FEATURE_COLUMNS if c in df.columns]

    required_cols = config.ID_COLUMNS + available_features
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in data: {missing}")

    df = df[required_cols].copy()

    # Basic dtype normalization
    df["accid"] = df["accid"].astype(str)
    df[config.DATE_COLUMN] = pd.to_datetime(df[config.DATE_COLUMN])

    # It is okay to keep identifiers as floats; we mainly need equality checks
    # between now_permno and permno, which work fine with floats.

    return df


def _add_label_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a binary label column indicating whether permno matches now_permno.
    """
    df = df.copy()
    df[config.LABEL_COLUMN] = (df["permno"] == df["now_permno"]).astype(int)
    return df


def _split_by_date(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into train and test sets.
    
    For each accid, all unique dates are sorted chronologically and then
    divided according to TRAIN_RATIO specified in config.py. This ensures
    temporal ordering is preserved per user while maintaining proper
    train/test splits.
    
    Example: If TRAIN_RATIO=0.7:
    - For each accid, the first 70% of dates (chronologically) go to train
    - The last 30% of dates go to test
    """
    date_col = config.DATE_COLUMN
    accid_col = "accid"
    train_ratio = config.TRAIN_RATIO
    
    # Validate train ratio
    if not (0 < train_ratio < 1):
        raise ValueError(f"TRAIN_RATIO must be between 0 and 1, got {train_ratio}")
    
    # Initialize split assignment column
    df = df.copy()
    df["_split"] = None
    
    # Get unique accids for progress tracking
    unique_accids = df[accid_col].unique()
    
    # Process each accid separately with progress bar
    for accid in tqdm(unique_accids, desc="Splitting data by date", unit="accid", leave=False):
        accid_mask = df[accid_col] == accid
        accid_subset = df.loc[accid_mask]
        
        # Get unique dates for this accid and sort chronologically
        accid_dates = accid_subset[date_col].unique()
        accid_dates_sorted = sorted(accid_dates)
        
        n_dates = len(accid_dates_sorted)
        if n_dates == 0:
            continue
        
        # Calculate split boundary (using integer index)
        train_end_idx = int(n_dates * train_ratio)
        
        # Ensure at least one date in each split if possible
        if train_end_idx == 0 and n_dates > 0:
            train_end_idx = 1
        if train_end_idx >= n_dates and n_dates > 0:
            train_end_idx = n_dates - 1
        
        # Determine which dates belong to which split
        train_dates = set(accid_dates_sorted[:train_end_idx])
        test_dates = set(accid_dates_sorted[train_end_idx:])
        
        # Vectorized assignment: assign all rows at once instead of looping
        date_values = df.loc[accid_mask, date_col]
        train_mask = date_values.isin(train_dates)
        test_mask = date_values.isin(test_dates)
        
        # Assign splits using vectorized operations
        df.loc[accid_mask & train_mask, "_split"] = "train"
        df.loc[accid_mask & test_mask, "_split"] = "test"
        
        # Fallback: assign any remaining rows to train (shouldn't happen)
        remaining_mask = accid_mask & df["_split"].isna()
        df.loc[remaining_mask, "_split"] = "train"
    
    # Create split dataframes
    train_df = df[df["_split"] == "train"].drop(columns=["_split"]).copy()
    test_df = df[df["_split"] == "test"].drop(columns=["_split"]).copy()
    
    return train_df, test_df


def load_and_prepare_splits() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the raw Stata data, keep only necessary columns, add labels, and
    split into train/test sets according to config.
    """
    raw_df = load_master_sampled()
    df = _select_and_cast_columns(raw_df)
    df = _add_label_column(df)
    train_df, test_df = _split_by_date(df)
    return train_df, test_df

