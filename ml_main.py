from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import ml_config as config
from model import ml_model as model_module
from out.results_saver import save_results_to_csv
from prep.ml_prep import load_and_prepare_splits


def _print_config_summary() -> None:
    """Print a short summary of the most important configuration values."""
    print("=== Configuration Summary ===")
    
    # Handle both single model and list of models
    if isinstance(config.MODEL_NAME, list):
        print(f"Model names          : {config.MODEL_NAME}")
        print(f"Number of models     : {len(config.MODEL_NAME)}")
    else:
        print(f"Model name           : {config.MODEL_NAME}")
    
    print(f"Date column          : {config.DATE_COLUMN}")
    print(f"Split ratios         : Train={config.TRAIN_RATIO:.1%}, Test={1-config.TRAIN_RATIO:.1%}")
    
    # Display reduction configuration
    if config.REDUCTION_MODE == "ratio":
        reduction_pct = (1 - config.REDUCTION_TARGET) * 100
        print(f"Reduction mode       : {config.REDUCTION_MODE} (keep top {config.REDUCTION_TARGET:.1%}, reduce by {reduction_pct:.1f}%)")
    else:
        print(f"Reduction mode       : {config.REDUCTION_MODE} (keep top {config.REDUCTION_TARGET} candidates)")
    
    print(f"Random seed          : {config.RANDOM_SEED}")
    print(f"Feature columns      : {config.FEATURE_COLUMNS}")
    print(f"Label column         : {config.LABEL_COLUMN}")
    print("Note: All model hyperparameters use scikit-learn defaults")
    print()


def _evaluate_group_topk(
    df: pd.DataFrame,
    scores: np.ndarray,
    group_cols: List[str],
    label_col: str,
    reduction_mode: str,
    reduction_target: float,
) -> Dict[str, float]:
    """
    Evaluate group-wise Top-K recall with flexible reduction strategy.

    For each group (accid, date, now_permno), candidates are sorted by score
    in descending order. We then check whether at least one positive example
    remains within the retained candidates of that group.

    Args:
        reduction_mode: "ratio" or "count"
            - "ratio": reduction_target is a ratio (0-1), e.g., 0.5 means keep top 50%
            - "count": reduction_target is absolute count, e.g., 50 means keep top 50
        reduction_target: The target value based on reduction_mode
    """
    df_eval = df.copy()
    df_eval["score"] = scores

    # Sort by group columns (ascending) and score (descending)
    sort_cols = group_cols + ["score"]
    ascending = [True] * len(group_cols) + [False]
    df_eval = df_eval.sort_values(sort_cols, ascending=ascending)

    total_groups = 0
    groups_with_positive = 0
    hits_at_k = 0
    retained_candidates = 0
    total_original_candidates = 0

    # Get group iterator for progress tracking
    group_iterator = df_eval.groupby(group_cols, sort=False)
    total_groups_estimate = len(df_eval.groupby(group_cols).size())
    
    for _, group in tqdm(group_iterator, desc="Evaluating groups", total=total_groups_estimate, unit="group", leave=False):
        total_groups += 1
        group_size = len(group)
        total_original_candidates += group_size
        
        # Calculate how many candidates to retain based on mode
        if reduction_mode == "ratio":
            # Keep top N% of candidates
            k = max(1, int(group_size * reduction_target))
        else:  # reduction_mode == "count"
            # Keep top N candidates
            k = int(reduction_target)
        
        # Ensure k doesn't exceed group size
        k = min(k, group_size)
        
        # Always count retained candidates for all groups
        retained_candidates += k
        
        has_positive = (group[label_col] == 1).any()
        if not has_positive:
            continue

        groups_with_positive += 1

        top = group.head(k)
        if (top[label_col] == 1).any():
            hits_at_k += 1

    if groups_with_positive == 0:
        recall_at_k = 0.0
    else:
        recall_at_k = hits_at_k / groups_with_positive

    # Calculate total success rate (after reduction)
    # This is the proportion of all groups where Top-K contains at least one positive
    total_success_rate = hits_at_k / total_groups if total_groups > 0 else 0.0

    # Calculate baseline success rate (before reduction)
    # This is the proportion of all groups that contain at least one positive example
    baseline_success_rate = groups_with_positive / total_groups if total_groups > 0 else 0.0

    avg_candidates_retained = (
        retained_candidates / total_groups if total_groups > 0 else 0.0
    )
    
    avg_reduction_rate = (
        (1 - retained_candidates / total_original_candidates) * 100
        if total_original_candidates > 0 else 0.0
    )

    return {
        "num_groups": float(total_groups),
        "num_groups_with_positive": float(groups_with_positive),
        "hits_at_k": float(hits_at_k),
        "recall_at_k": float(recall_at_k),
        "total_success_rate": float(total_success_rate),
        "baseline_success_rate": float(baseline_success_rate),
        "avg_candidates_retained": float(avg_candidates_retained),
        "avg_reduction_rate": float(avg_reduction_rate),
        "reduction_mode": reduction_mode,
        "reduction_target": float(reduction_target),
    }


def _prepare_xy(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract X (features) and y (labels) from a preprocessed dataframe."""
    X = df[config.FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[config.LABEL_COLUMN].to_numpy(dtype=int)
    return X, y


def _print_dataset_info(
    df: pd.DataFrame,
    name: str,
    group_cols: List[str],
    reduction_mode: str,
    reduction_target: float,
) -> None:
    """
    Print objective information about a dataset (without model evaluation).
    
    Args:
        df: DataFrame to analyze
        name: Name of the dataset (e.g., "train", "test")
        group_cols: Columns that define groups
        reduction_mode: "ratio" or "count"
        reduction_target: Target value for reduction
    """
    label_width = 30
    y = df[config.LABEL_COLUMN].to_numpy(dtype=int)
    
    # Calculate groups info
    total_groups = len(df.groupby(group_cols).size())
    groups_with_positive = 0
    total_original_candidates = 0
    retained_candidates = 0
    
    group_iterator = df.groupby(group_cols, sort=False)
    total_groups_estimate = len(df.groupby(group_cols).size())
    
    for _, group in tqdm(group_iterator, desc=f"Analyzing {name} set", total=total_groups_estimate, unit="group", leave=False):
        group_size = len(group)
        total_original_candidates += group_size
        
        # Calculate how many candidates to retain based on mode
        if reduction_mode == "ratio":
            k = max(1, int(group_size * reduction_target))
        else:  # reduction_mode == "count"
            k = int(reduction_target)
        
        k = min(k, group_size)
        retained_candidates += k
        
        if (group[config.LABEL_COLUMN] == 1).any():
            groups_with_positive += 1
    
    avg_candidates_retained = (
        retained_candidates / total_groups if total_groups > 0 else 0.0
    )
    
    avg_reduction_rate = (
        (1 - retained_candidates / total_original_candidates) * 100
        if total_original_candidates > 0 else 0.0
    )
    
    print(f"\n=== {name.upper()} Set Information ===")
    print(f"{'Number of rows':<{label_width}} : {len(df):,}")
    print(f"{'Positive rate':<{label_width}} : {y.mean():.6f}")
    print(f"{'Groups (total)':<{label_width}} : {total_groups:,}")
    print(f"{'Avg candidates retained':<{label_width}} : {avg_candidates_retained:.2f}")
    print(f"{'Avg reduction rate':<{label_width}} : {avg_reduction_rate:.2f}%")


def main() -> None:
    """Entry point: train the selected model(s) and evaluate Top-K recall."""
    _print_config_summary()

    print("Loading and preparing data splits...")
    train_df, test_df = load_and_prepare_splits()
    print(
        f"Train size: {len(train_df)}, "
        f"Test size: {len(test_df)}"
    )

    X_train, y_train = _prepare_xy(train_df)
    X_test, y_test = _prepare_xy(test_df)

    # Handle both single model and list of models
    if isinstance(config.MODEL_NAME, list):
        model_names = config.MODEL_NAME
    else:
        model_names = [config.MODEL_NAME]

    # Initialize results collection list
    all_results = []

    group_cols = ["accid", "date", "now_permno"]
    
    # Print objective information for both sets (before model training)
    print("\n" + "="*60)
    print("Dataset Information")
    print("="*60)
    _print_dataset_info(
        train_df, "train", group_cols,
        config.REDUCTION_MODE, config.REDUCTION_TARGET
    )
    _print_dataset_info(
        test_df, "test", group_cols,
        config.REDUCTION_MODE, config.REDUCTION_TARGET
    )

    def _run_eval(
        model_name: str,
        name: str,
        split_df: pd.DataFrame,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, float]:
        scores = model_module.get_scores(model, X)
        metrics = _evaluate_group_topk(
            split_df,
            scores,
            group_cols=group_cols,
            label_col=config.LABEL_COLUMN,
            reduction_mode=config.REDUCTION_MODE,
            reduction_target=config.REDUCTION_TARGET,
        )
        
        # Display reduction info based on mode
        if metrics['reduction_mode'] == "ratio":
            target_display = f"{metrics['reduction_target']:.1%}"
            hits_label = f"Hits at Top-{target_display}"
            recall_label = f"Recall@{target_display}"
        else:
            target_display = f"{int(metrics['reduction_target'])}"
            hits_label = f"Hits at Top-{target_display}"
            recall_label = f"Recall@{target_display}"
        
        # Only print three metrics: Hits, Recall, Success Rate
        label_width = 30
        groups_with_positives = int(metrics['num_groups_with_positive'])
        hits_at_k = int(metrics['hits_at_k'])
        print(f"{hits_label:<{label_width}} : {hits_at_k} / {groups_with_positives}")
        print(f"{recall_label:<{label_width}} : {metrics['recall_at_k']:.4f}")
        print(f"{'Total success rate':<{label_width}} : {metrics['total_success_rate']:.4f} ({metrics['baseline_success_rate']:.4f})")
        
        return metrics

    # Train and evaluate each model
    for idx, model_name in enumerate(model_names, 1):
        print(f"\n{'='*60}")
        print(f"Model {idx}/{len(model_names)}: {model_name}")
        print(f"{'='*60}")
        
        print("\nCreating model...")
        model = model_module.create_model(model_name)
        print(model)

        print("\nTraining model...")
        model.fit(X_train, y_train)
        print("Training completed.")

        print("\n=== Evaluation on train set ===")
        train_metrics = _run_eval(model_name, "train", train_df, X_train, y_train)
        
        # Collect train results
        all_results.append({
            'model': model_name,
            'type': 'train',
            'model_type': 'ml',
            'recall': train_metrics['recall_at_k'],
            'recall_w': 0.0,  # ML doesn't track warm/cold-start separately
            'recall_c': 0.0,
            'success': train_metrics['total_success_rate'],
        })
        
        print("\n=== Evaluation on test set ===")
        test_metrics = _run_eval(model_name, "test", test_df, X_test, y_test)
        
        # Collect test results
        all_results.append({
            'model': model_name,
            'type': 'test',
            'model_type': 'ml',
            'recall': test_metrics['recall_at_k'],
            'recall_w': 0.0,  # ML doesn't track warm/cold-start separately
            'recall_c': 0.0,
            'success': test_metrics['total_success_rate'],
        })
        
        if idx < len(model_names):
            print()  # Add blank line between models
    
    # Save all results to CSV
    if config.SAVE_RESULTS_CSV:
        save_results_to_csv(all_results, config.RESULTS_CSV_PATH, append=False)
        print(f"\n{'='*60}")
        print(f"   Results saved to: {config.RESULTS_CSV_PATH}")
        print(f"   Total models: {len(model_names)}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()



