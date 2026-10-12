"""
Configuration file for Collaborative Filtering methods.

This file contains CF-specific configurations that are separate from
the ML configuration in config.py.
"""

from pathlib import Path

# Import shared configurations
from config.ml_config import PROJECT_DIR, DATE_COLUMN, RANDOM_SEED, TRAIN_RATIO

# ---------------------------------------------------------------------------
# CF Model Configuration
# ---------------------------------------------------------------------------

# CF model name(s) to use; can be a single string or a list of strings
# Supported values:
#   - "als" (Alternating Least Squares - implicit feedback)
#   - "bpr" (Bayesian Personalized Ranking)
#   - "item_cf" (Item-based Collaborative Filtering)
#   - "user_cf" (User-based Collaborative Filtering)
#   - "svd" (Singular Value Decomposition)
#   - "nmf" (Non-negative Matrix Factorization)
#   - "knn" (K-Nearest Neighbors)
# Examples:
#   CF_MODEL_NAME = "als"  # Single model
#   CF_MODEL_NAME = ["als", "bpr", "item_cf"]  # Multiple models
CF_MODEL_NAME = ["als", "bpr", "item_cf", "user_cf", "svd", "nmf", "knn"]

# ---------------------------------------------------------------------------
# ALS Hyperparameters
# ---------------------------------------------------------------------------

# Number of latent factors (embedding dimension)
ALS_FACTORS = 100

# Regularization parameter (L2 penalty)
ALS_REGULARIZATION = 0.01

# Number of ALS iterations
ALS_ITERATIONS = 30

# Confidence scaling parameter for implicit feedback
# Higher values give more weight to positive interactions
ALS_ALPHA = 40.0

# Use GPU acceleration if available (requires cupy)
ALS_USE_GPU = False

# ---------------------------------------------------------------------------
# BPR Hyperparameters
# ---------------------------------------------------------------------------

# Number of latent factors
BPR_FACTORS = 100

# Learning rate for SGD
BPR_LEARNING_RATE = 0.05

# Regularization parameter
BPR_REGULARIZATION = 0.01

# Number of training iterations
BPR_ITERATIONS = 100

# Number of negative samples per positive sample
BPR_NUM_NEGATIVES = 1

# ---------------------------------------------------------------------------
# Item-based CF Hyperparameters
# ---------------------------------------------------------------------------

# Number of similar items to consider
ITEM_CF_K = 50

# Similarity metric: "cosine", "pearson", or "jaccard"
ITEM_CF_SIMILARITY = "cosine"

# Minimum number of common users for similarity calculation
ITEM_CF_MIN_COMMON = 2

# ---------------------------------------------------------------------------
# User-based CF Hyperparameters
# ---------------------------------------------------------------------------

# Number of similar users to consider
USER_CF_K = 50

# Similarity metric: "cosine", "pearson", or "jaccard"
USER_CF_SIMILARITY = "cosine"

# Minimum number of common items for similarity calculation
USER_CF_MIN_COMMON = 2

# ---------------------------------------------------------------------------
# SVD Hyperparameters
# ---------------------------------------------------------------------------

# Number of latent factors for SVD
SVD_FACTORS = 100

# ---------------------------------------------------------------------------
# NMF Hyperparameters
# ---------------------------------------------------------------------------

# Number of latent factors for NMF
NMF_FACTORS = 100

# Regularization parameter for NMF
NMF_REGULARIZATION = 0.01

# Number of iterations for NMF
NMF_ITERATIONS = 200

# ---------------------------------------------------------------------------
# KNN Hyperparameters
# ---------------------------------------------------------------------------

# Number of nearest neighbors to consider
KNN_K = 50

# Similarity metric: "cosine", "pearson", or "jaccard"
KNN_SIMILARITY = "cosine"

# ---------------------------------------------------------------------------
# Interaction Matrix Configuration
# ---------------------------------------------------------------------------

# How to weight interactions in the user-item matrix
# Options:
#   - "binary": 1 if user interacted with item, 0 otherwise
#   - "count": number of times user interacted with item
#   - "time_decay": apply exponential decay based on recency
INTERACTION_WEIGHT = "binary"

# Time decay parameter (only used if INTERACTION_WEIGHT == "time_decay")
# Higher values = faster decay (recent interactions weighted more)
TIME_DECAY_RATE = 0.1

# ---------------------------------------------------------------------------
# Cold-Start Handling
# ---------------------------------------------------------------------------

# Strategy for handling cold-start users/items in test set
# Options:
#   - "popularity": score based on global item popularity
#   - "random": assign random scores
#   - "zero": assign zero scores (will rank last)
#   - "mean": assign mean score from training set
COLD_START_STRATEGY = "popularity"

# Whether to report cold-start performance separately
REPORT_COLD_START_SEPARATELY = True

# ---------------------------------------------------------------------------
# Candidate Reduction Configuration
# ---------------------------------------------------------------------------

# Reduction mode: "ratio" or "count"
# - "ratio": Use REDUCTION_TARGET as a ratio (e.g., 0.1 means keep top 10%)
# - "count": Use REDUCTION_TARGET as absolute count (e.g., 50 means keep top 50)
REDUCTION_MODE = "ratio"

# Reduction target value
# - If REDUCTION_MODE == "ratio": value between 0 and 1 (e.g., 0.1 = keep top 10%)
# - If REDUCTION_MODE == "count": positive integer (e.g., 50 = keep top 50 candidates)
REDUCTION_TARGET = 0.1  # Keep top 10% (same as ML baseline)

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

# Directory to save CF model artifacts
CF_OUTPUT_DIR = PROJECT_DIR / "out" / "cf_outputs"

# Whether to save trained models to disk
SAVE_MODELS = True

# Whether to save interaction matrices
SAVE_MATRICES = True

# Whether to save evaluation results
SAVE_RESULTS = True

# ---------------------------------------------------------------------------
# Results CSV Configuration
# ---------------------------------------------------------------------------

# Whether to save evaluation results to CSV file
SAVE_RESULTS_CSV = True

# CSV file path for saving results
RESULTS_CSV_PATH = PROJECT_DIR / "out" / "cf_results.csv"

# CSV columns: model, type, recall, recall_w, recall_c, success
# - model: model name
# - type: "train" or "test"
# - recall: Recall@K (overall)
# - recall_w: Warm-start recall
# - recall_c: Cold-start recall
# - success: Total success rate

