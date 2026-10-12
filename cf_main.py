"""
Main entry point for Collaborative Filtering evaluation.

This script:
1. Loads CF data
2. Trains CF model(s)
3. Evaluates on test set using same metrics as ML baseline
4. Reports results with cold-start analysis
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import cf_config as config_cf
from model.cf_model import create_cf_model
from out.results_saver import save_results_to_csv
from prep.cf_prep import load_cf_data


def _print_config_summary() -> None:
    """Print CF configuration summary."""
    print("=== CF Configuration Summary ===")
    
    if isinstance(config_cf.CF_MODEL_NAME, list):
        print(f"CF model names       : {config_cf.CF_MODEL_NAME}")
        print(f"Number of models     : {len(config_cf.CF_MODEL_NAME)}")
    else:
        print(f"CF model name        : {config_cf.CF_MODEL_NAME}")
    
    print(f"Date column          : {config_cf.DATE_COLUMN}")
    print(f"Split ratios         : Train={config_cf.TRAIN_RATIO:.1%}, Test={1-config_cf.TRAIN_RATIO:.1%}")
    
    if config_cf.REDUCTION_MODE == "ratio":
        reduction_pct = (1 - config_cf.REDUCTION_TARGET) * 100
        print(f"Reduction mode       : {config_cf.REDUCTION_MODE} (keep top {config_cf.REDUCTION_TARGET:.1%}, reduce by {reduction_pct:.1f}%)")
    else:
        print(f"Reduction mode       : {config_cf.REDUCTION_MODE} (keep top {config_cf.REDUCTION_TARGET} candidates)")
    
    print(f"Random seed          : {config_cf.RANDOM_SEED}")
    print(f"Interaction weight   : {config_cf.INTERACTION_WEIGHT}")
    print(f"Cold-start strategy  : {config_cf.COLD_START_STRATEGY}")
    print()


def _evaluate_group_topk(
    df: pd.DataFrame,
    scores: np.ndarray,
    group_cols: List[str],
    label_col: str,
    reduction_mode: str,
    reduction_target: float,
    track_cold_start: bool = False,
    dataset = None,
) -> Dict[str, float]:
    """
    Evaluate group-wise Top-K recall (same as ML baseline).
    
    Optionally tracks cold-start vs warm-start performance separately.
    """
    df_eval = df.copy()
    df_eval["score"] = scores

    sort_cols = group_cols + ["score"]
    ascending = [True] * len(group_cols) + [False]
    df_eval = df_eval.sort_values(sort_cols, ascending=ascending)

    total_groups = 0
    groups_with_positive = 0
    hits_at_k = 0
    retained_candidates = 0
    total_original_candidates = 0
    
    # Cold-start tracking
    cold_start_groups = 0
    cold_start_hits = 0
    cold_start_groups_with_positive = 0
    warm_start_groups = 0
    warm_start_hits = 0
    warm_start_groups_with_positive = 0

    group_iterator = df_eval.groupby(group_cols, sort=False)
    total_groups_estimate = len(df_eval.groupby(group_cols).size())
    
    for group_key, group in tqdm(group_iterator, desc="Evaluating groups", total=total_groups_estimate, unit="group", leave=False):
        total_groups += 1
        group_size = len(group)
        total_original_candidates += group_size
        
        # Calculate k
        if reduction_mode == "ratio":
            k = max(1, int(group_size * reduction_target))
        else:
            k = int(reduction_target)
        
        k = min(k, group_size)
        retained_candidates += k
        
        # Check if group has positive
        has_positive = (group[label_col] == 1).any()
        
        # Check if cold-start (if tracking)
        is_cold_start = False
        if track_cold_start and dataset is not None:
            accid = group_key[0]  # First element of group_key is accid
            is_cold_start = dataset.is_cold_start_user(accid)
        
        # Track cold-start vs warm-start
        if track_cold_start:
            if is_cold_start:
                cold_start_groups += 1
                if has_positive:
                    cold_start_groups_with_positive += 1
            else:
                warm_start_groups += 1
                if has_positive:
                    warm_start_groups_with_positive += 1
        
        if not has_positive:
            continue

        groups_with_positive += 1

        top = group.head(k)
        if (top[label_col] == 1).any():
            hits_at_k += 1
            if track_cold_start:
                if is_cold_start:
                    cold_start_hits += 1
                else:
                    warm_start_hits += 1

    if groups_with_positive == 0:
        recall_at_k = 0.0
    else:
        recall_at_k = hits_at_k / groups_with_positive

    total_success_rate = hits_at_k / total_groups if total_groups > 0 else 0.0
    baseline_success_rate = groups_with_positive / total_groups if total_groups > 0 else 0.0

    avg_candidates_retained = (
        retained_candidates / total_groups if total_groups > 0 else 0.0
    )
    
    avg_reduction_rate = (
        (1 - retained_candidates / total_original_candidates) * 100
        if total_original_candidates > 0 else 0.0
    )
    
    # Calculate cold-start metrics
    cold_start_recall = (
        cold_start_hits / cold_start_groups_with_positive
        if cold_start_groups_with_positive > 0 else 0.0
    )
    warm_start_recall = (
        warm_start_hits / warm_start_groups_with_positive
        if warm_start_groups_with_positive > 0 else 0.0
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
        # Cold-start metrics
        "cold_start_groups": float(cold_start_groups),
        "cold_start_groups_with_positive": float(cold_start_groups_with_positive),
        "cold_start_hits": float(cold_start_hits),
        "cold_start_recall": float(cold_start_recall),
        "warm_start_groups": float(warm_start_groups),
        "warm_start_groups_with_positive": float(warm_start_groups_with_positive),
        "warm_start_hits": float(warm_start_hits),
        "warm_start_recall": float(warm_start_recall),
    }


def _score_test_set(cf_model, test_df: pd.DataFrame) -> np.ndarray:
    """
    Score all test candidates using the CF model.
    
    Args:
        cf_model: Trained CF model
        test_df: Test dataframe with columns [accid, permno, ...]
    
    Returns:
        Array of scores (same length as test_df)
    """
    scores = []
    
    # Group by (accid, date, now_permno) and score each group
    group_cols = config_cf.GROUP_COLUMNS
    groups = test_df.groupby(group_cols, sort=False)
    
    for group_key, group in tqdm(groups, desc="Scoring test groups", unit="group", leave=False):
        accid = group_key[0]
        candidate_permnos = group['permno'].tolist()
        
        # Score candidates
        group_scores = cf_model.score_candidates(accid, candidate_permnos)
        scores.extend(group_scores)
    
    return np.array(scores)


def _print_evaluation_results(
    model_name: str,
    split_name: str,
    metrics: Dict[str, float],
) -> None:
    """Print evaluation results in a formatted way."""
    label_width = 35
    
    # Display reduction info
    if metrics['reduction_mode'] == "ratio":
        target_display = f"{metrics['reduction_target']:.1%}"
        hits_label = f"Hits at Top-{target_display}"
        recall_label = f"Recall@{target_display}"
    else:
        target_display = f"{int(metrics['reduction_target'])}"
        hits_label = f"Hits at Top-{target_display}"
        recall_label = f"Recall@{target_display}"
    
    print(f"\n=== Evaluation on {split_name} set ===")
    
    # Overall metrics
    hits_at_k = int(metrics['hits_at_k'])
    print(f"{recall_label:<{label_width}} : {metrics['recall_at_k']:.4f}")
    print(f"{'Total success rate':<{label_width}} : {metrics['total_success_rate']:.4f} ({metrics['baseline_success_rate']:.4f})")
    
    # Cold-start breakdown
    if config_cf.REPORT_COLD_START_SEPARATELY:
        print(f"\n{'--- Cold-Start Breakdown ---':<{label_width}}")
        
        # Warm-start
        warm_groups = int(metrics['warm_start_groups'])
        warm_hits = int(metrics['warm_start_hits'])
        print(f"{'Warm-start groups':<{label_width}} : {warm_groups}")
        print(f"{'Warm-start recall':<{label_width}} : {metrics['warm_start_recall']:.4f}")
        
        # Cold-start
        cold_groups = int(metrics['cold_start_groups'])
        cold_hits = int(metrics['cold_start_hits'])
        print(f"{'Cold-start groups':<{label_width}} : {cold_groups}")
        print(f"{'Cold-start recall':<{label_width}} : {metrics['cold_start_recall']:.4f}")


def main() -> None:
    """Main entry point for CF evaluation."""
    _print_config_summary()

    print("Loading CF data...")
    dataset = load_cf_data()
    
    # Handle both single model and list of models
    if isinstance(config_cf.CF_MODEL_NAME, list):
        model_names = config_cf.CF_MODEL_NAME
    else:
        model_names = [config_cf.CF_MODEL_NAME]

    # Initialize results collection list
    all_results = []

    # Train and evaluate each model
    for idx, model_name in enumerate(model_names, 1):
        print(f"\n{'='*60}")
        print(f"CF Model {idx}/{len(model_names)}: {model_name}")
        print(f"{'='*60}")
        
        print("\nCreating CF model...")
        cf_model = create_cf_model(model_name)
        
        print("\nTraining CF model...")
        cf_model.fit(dataset)
        
        # Evaluate on train set
        print("\nScoring train set...")
        train_scores = _score_test_set(cf_model, dataset.train_df)
        
        print("Evaluating train set...")
        train_metrics = _evaluate_group_topk(
            dataset.train_df,
            train_scores,
            group_cols=config_cf.GROUP_COLUMNS,
            label_col="label",
            reduction_mode=config_cf.REDUCTION_MODE,
            reduction_target=config_cf.REDUCTION_TARGET,
            track_cold_start=config_cf.REPORT_COLD_START_SEPARATELY,
            dataset=dataset,
        )
        
        _print_evaluation_results(model_name, "train", train_metrics)
        
        # Collect train results
        all_results.append({
            'model': model_name,
            'type': 'train',
            'model_type': 'cf',
            'recall': train_metrics['recall_at_k'],
            'recall_w': train_metrics['warm_start_recall'],
            'recall_c': train_metrics['cold_start_recall'],
            'success': train_metrics['total_success_rate'],
        })
        
        # Evaluate on test set
        print("\nScoring test set...")
        test_scores = _score_test_set(cf_model, dataset.test_df)
        
        print("Evaluating test set...")
        test_metrics = _evaluate_group_topk(
            dataset.test_df,
            test_scores,
            group_cols=config_cf.GROUP_COLUMNS,
            label_col="label",
            reduction_mode=config_cf.REDUCTION_MODE,
            reduction_target=config_cf.REDUCTION_TARGET,
            track_cold_start=config_cf.REPORT_COLD_START_SEPARATELY,
            dataset=dataset,
        )
        
        _print_evaluation_results(model_name, "test", test_metrics)
        
        # Collect test results
        all_results.append({
            'model': model_name,
            'type': 'test',
            'model_type': 'cf',
            'recall': test_metrics['recall_at_k'],
            'recall_w': test_metrics['warm_start_recall'],
            'recall_c': test_metrics['cold_start_recall'],
            'success': test_metrics['total_success_rate'],
        })
        
        if idx < len(model_names):
            print()  # Add blank line between models
    
    # Save all results to CSV
    if config_cf.SAVE_RESULTS_CSV:
        save_results_to_csv(all_results, config_cf.RESULTS_CSV_PATH, append=False)
        print(f"\n{'='*60}")
        print(f"   Results saved to: {config_cf.RESULTS_CSV_PATH}")
        print(f"   Total models: {len(model_names)}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()

