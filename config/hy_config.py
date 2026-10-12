"""
Configuration file for Hybrid Recommender methods.

This file contains Hybrid-specific configurations that combine
CF and ML approaches.
"""

from pathlib import Path

# Import shared configurations
from config.ml_config import PROJECT_DIR, DATE_COLUMN, RANDOM_SEED, TRAIN_RATIO

# ---------------------------------------------------------------------------
# Hybrid Model Configuration
# ---------------------------------------------------------------------------

# Top 3 ML models from ML evaluation (all achieved 0.9267 Recall@10%)
TOP_ML_MODELS = ["linear_svm", "logistic_l2", "ridge_classifier"]

# Top 3 CF models from CF evaluation (Test Recall@10%)
TOP_CF_MODELS = ["nmf", "als", "svd"]  

# Hybrid types to evaluate
HYBRID_TYPES = ["weighted", "stacking", "feature_cf", "switching"]  # All types

# Generate all combinations based on HYBRID_TYPES × TOP_CF_MODELS × TOP_ML_MODELS
HY_MODEL_NAME = []

# Generate all combinations
for hybrid_type in HYBRID_TYPES:
    for cf_model in TOP_CF_MODELS:
        for ml_model in TOP_ML_MODELS:
            model_name = f"{hybrid_type}_{cf_model}_{ml_model}"
            HY_MODEL_NAME.append(model_name)

# ---------------------------------------------------------------------------
# Weighted Hybrid Hyperparameters
# ---------------------------------------------------------------------------

# Weight for CF score (ML weight = 1 - CF_WEIGHT)
WEIGHTED_CF_WEIGHT = 0.3

# Note: CF and ML models are specified in model name (e.g., "weighted_nmf_logistic_l2")
# No default models needed - all combinations are generated from TOP_CF_MODELS × TOP_ML_MODELS

# ---------------------------------------------------------------------------
# Stacking Hybrid Hyperparameters
# ---------------------------------------------------------------------------

# Base CF models for stacking (use all top CF models)
STACKING_CF_MODELS = TOP_CF_MODELS

# Base ML models for stacking (use all top ML models)
STACKING_ML_MODELS = TOP_ML_MODELS

# Use cross-validation for stacking
STACKING_USE_CV = False

# ---------------------------------------------------------------------------
# Feature-augmented CF Hyperparameters
# ---------------------------------------------------------------------------

# Number of CF embedding dimensions to use as features
FEATURE_CF_EMBEDDING_DIM = 50

# Weight for CF embeddings vs original features
FEATURE_CF_EMBEDDING_WEIGHT = 0.5

# ---------------------------------------------------------------------------
# Switching Hybrid Hyperparameters
# ---------------------------------------------------------------------------

# Threshold for switching between CF and ML
# If user has >= threshold interactions, use CF; otherwise use ML
SWITCHING_INTERACTION_THRESHOLD = 3

# ---------------------------------------------------------------------------
# Cold-Start Handling
# ---------------------------------------------------------------------------

# Strategy for handling cold-start users/items
# Options: "ml", "popularity", "random", "zero"
COLD_START_STRATEGY = "ml"

# Whether to report cold-start performance separately
REPORT_COLD_START_SEPARATELY = True

# ---------------------------------------------------------------------------
# Candidate Reduction Configuration
# ---------------------------------------------------------------------------

# Reduction mode: "ratio" or "count"
REDUCTION_MODE = "ratio"

# Reduction target value
REDUCTION_TARGET = 0.1  # Keep top 10%

# ---------------------------------------------------------------------------
# Evaluation Configuration
# ---------------------------------------------------------------------------

# Columns that define each group for evaluation
GROUP_COLUMNS = ["accid", "date", "now_permno"]

# Whether to use progress bars during evaluation
SHOW_PROGRESS = False

# ---------------------------------------------------------------------------
# Output Configuration
# ---------------------------------------------------------------------------

# Directory to save Hybrid model artifacts
HY_OUTPUT_DIR = PROJECT_DIR / "out" / "hy_outputs"

# Whether to save trained models to disk
SAVE_MODELS = True

# Whether to save evaluation results
SAVE_RESULTS = True

# ---------------------------------------------------------------------------
# Results CSV Configuration
# ---------------------------------------------------------------------------

# Whether to save evaluation results to CSV file
SAVE_RESULTS_CSV = True

# CSV file path for saving results
RESULTS_CSV_PATH = PROJECT_DIR / "out" / "hy_results.csv"

# CSV columns: model, type, recall, recall_w, recall_c, success
# - model: model name
# - type: "train" or "test"
# - recall: Recall@K (overall)
# - recall_w: Warm-start recall
# - recall_c: Cold-start recall
# - success: Total success rate

