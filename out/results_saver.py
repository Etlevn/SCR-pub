"""
Utility functions for saving evaluation results to CSV.

This module provides functions to save model evaluation results
in a standardized CSV format across ML, CF, and Hybrid models.
"""

import csv
from pathlib import Path
from typing import Dict, List


def save_results_to_csv(
    results: List[Dict],
    csv_path: Path,
    append: bool = True
) -> None:
    """
    Save evaluation results to CSV file with train/test in same row.
    
    Args:
        results: List of result dictionaries with keys:
            - model: model name
            - type: "train" or "test"
            - model_type: "ml", "cf", or "hy"
            - recall: Recall@K (overall)
            - recall_w: Warm-start recall
            - recall_c: Cold-start recall
            - success: Total success rate
        csv_path: Path to CSV file
        append: If True, append to existing file; if False, overwrite
    """
    # Group results by model (train and test together)
    model_results = {}
    for result in results:
        model_name = result.get('model', 'unknown')
        result_type = result.get('type', 'unknown')
        
        if model_name not in model_results:
            model_results[model_name] = {'model_type': result.get('model_type', 'unknown')}
        
        model_results[model_name][result_type] = result
    
    # Define CSV columns with type as first column
    fieldnames = [
        'type',
        'model',
        'train_recall', 'train_recall_w', 'train_recall_c', 'train_success',
        'test_recall', 'test_recall_w', 'test_recall_c', 'test_success'
    ]
    
    # Check if file exists
    file_exists = csv_path.exists()
    
    # Determine write mode
    mode = 'a' if (append and file_exists) else 'w'
    
    # Write to CSV
    with open(csv_path, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Write header if file is new or being overwritten
        if mode == 'w' or not file_exists:
            writer.writeheader()
        
        # Write results (one row per model with train and test columns)
        for model_name, data in model_results.items():
            train_data = data.get('train', {})
            test_data = data.get('test', {})
            model_type = data.get('model_type', 'unknown')
            
            row = {
                'type': model_type,
                'model': model_name,
                'train_recall': f"{train_data.get('recall', 0.0):.4f}",
                'train_recall_w': f"{train_data.get('recall_w', 0.0):.4f}",
                'train_recall_c': f"{train_data.get('recall_c', 0.0):.4f}",
                'train_success': f"{train_data.get('success', 0.0):.4f}",
                'test_recall': f"{test_data.get('recall', 0.0):.4f}",
                'test_recall_w': f"{test_data.get('recall_w', 0.0):.4f}",
                'test_recall_c': f"{test_data.get('recall_c', 0.0):.4f}",
                'test_success': f"{test_data.get('success', 0.0):.4f}",
            }
            writer.writerow(row)


def format_result_dict(
    model_name: str,
    split_type: str,
    metrics: Dict
) -> Dict:
    """
    Format evaluation metrics into standardized result dictionary.
    
    Args:
        model_name: Name of the model
        split_type: "train" or "test"
        metrics: Dictionary containing evaluation metrics
    
    Returns:
        Formatted result dictionary
    """
    return {
        'model': model_name,
        'type': split_type,
        'recall': metrics.get('recall_at_k', 0.0),
        'recall_w': metrics.get('warm_start_recall', 0.0),
        'recall_c': metrics.get('cold_start_recall', 0.0),
        'success': metrics.get('total_success_rate', 0.0),
    }


def print_csv_summary(csv_path: Path) -> None:
    """
    Print summary of saved CSV results.
    
    Args:
        csv_path: Path to CSV file
    """
    if not csv_path.exists():
        print(f"⚠️  CSV file not found: {csv_path}")
        return
    
    # Read CSV
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        print(f"⚠️  CSV file is empty: {csv_path}")
        return
    
    print(f"\n{'='*80}")
    print(f"CSV Results Summary: {csv_path.name}")
    print(f"{'='*80}")
    print(f"Total models: {len(rows)}")
    
    # Find best test result
    if rows:
        best_test = max(rows, key=lambda x: float(x.get('test_recall', 0.0)))
        print(f"\n🏆 Best Test Model:")
        print(f"   Model: {best_test['model']}")
        print(f"   Test Recall: {best_test['test_recall']}")
        print(f"   Test Warm-start: {best_test['test_recall_w']}")
        print(f"   Test Cold-start: {best_test['test_recall_c']}")
        print(f"   Test Success: {best_test['test_success']}")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    """
    Test the CSV saving functionality.
    """
    from pathlib import Path
    
    # Test data - train and test for same model
    test_results = [
        {
            'model': 'logistic_l2',
            'type': 'train',
            'model_type': 'ml',
            'recall': 0.8894,
            'recall_w': 0.9000,
            'recall_c': 0.8500,
            'success': 0.4260,
        },
        {
            'model': 'logistic_l2',
            'type': 'test',
            'model_type': 'ml',
            'recall': 0.9267,
            'recall_w': 0.9375,
            'recall_c': 0.8636,
            'success': 0.5516,
        },
        {
            'model': 'linear_svm',
            'type': 'train',
            'model_type': 'ml',
            'recall': 0.8800,
            'recall_w': 0.8900,
            'recall_c': 0.8400,
            'success': 0.4200,
        },
        {
            'model': 'linear_svm',
            'type': 'test',
            'model_type': 'ml',
            'recall': 0.9200,
            'recall_w': 0.9300,
            'recall_c': 0.8500,
            'success': 0.5400,
        },
    ]
    
    # Test saving
    test_csv = Path("test_results.csv")
    print("Testing CSV saving with train/test in same row...")
    save_results_to_csv(test_results, test_csv, append=False)
    print(f"✅ Results saved to: {test_csv}")
    
    # Test summary
    print_csv_summary(test_csv)
    
    # Clean up
    test_csv.unlink()
    print("✅ Test completed!")

