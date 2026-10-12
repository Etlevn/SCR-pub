from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Data configuration
# ---------------------------------------------------------------------------

# Column that will be used for time-based splitting
DATE_COLUMN = "date"

# Columns that identify each candidate row
ID_COLUMNS = ["accid", "date", "now_permno", "permno"]

# Feature columns (you can modify this list manually if needed)
FEATURE_COLUMNS = [f"f{i}" for i in range(14)]  # f0, f1, ..., f13

# Name of the binary label column created during preprocessing
LABEL_COLUMN = "label"

# Path to the original Stata data file (used by the loader)
DATA_FILE = PROJECT_DIR / "data" / "master_final_sampled.dta"

# ---------------------------------------------------------------------------
# Split configuration
# ---------------------------------------------------------------------------

# Split ratio for training set
# Data is split per accid: for each accid, dates are sorted chronologically
# and then divided according to this ratio. The remaining dates go to test set.
TRAIN_RATIO = 0.7  # First 70% of dates per accid go to training, last 30% go to test

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

# Model name(s) to use; can be a single string or a list of strings
# Supported values:
#   - "logistic_l2", "logistic_l1", "logistic_enet" (LogisticRegression variants)
#   - "lasso", "ridge", "elasticnet" (Regression models)
#   - "linear_svm" (LinearSVC)
#   - "sgd_hinge", "sgd_log", "sgd_modified_huber" (SGDClassifier variants)
#   - "perceptron" (Perceptron)
#   - "passive_aggressive" (SGDClassifier configured as PassiveAggressive)
#   - "ridge_classifier" (RidgeClassifier)
# Examples:
#   MODEL_NAME = "logistic_l2"  # Single model
#   MODEL_NAME = ["logistic_l2", "logistic_l1", "ridge"]  # Multiple models
MODEL_NAME = ["logistic_l2", "logistic_l1", "logistic_enet", 
    "ridge", "lasso", "elasticnet", "linear_svm", 
    "sgd_hinge", "sgd_log", "sgd_modified_huber", 
    "perceptron", "passive_aggressive", 
    "ridge_classifier"]

# ---------------------------------------------------------------------------
# Candidate reduction configuration
# ---------------------------------------------------------------------------

# Reduction mode: "ratio" or "count"
# - "ratio": Use REDUCTION_TARGET as a ratio (e.g., 0.5 means keep top 50%, reduce by 50%)
# - "count": Use REDUCTION_TARGET as absolute count (e.g., 50 means keep top 50 candidates)
REDUCTION_MODE = "ratio"  # Options: "ratio" or "count"

# Reduction target value
# - If REDUCTION_MODE == "ratio": value between 0 and 1 (e.g., 0.5 = keep top 50%)
# - If REDUCTION_MODE == "count": positive integer (e.g., 50 = keep top 50 candidates)
REDUCTION_TARGET = 0.1  # Keep top 50% of candidates (reduce by 50%)

# Global random seed for reproducibility (used only for random_state in models)
# Set to None to use non-deterministic behavior
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Results saving configuration
# ---------------------------------------------------------------------------

# Whether to save evaluation results to CSV file
SAVE_RESULTS_CSV = True

# CSV file path for saving results
RESULTS_CSV_PATH = PROJECT_DIR / "out" / "ml_results.csv"

# CSV columns: model, type, recall, recall_w, recall_c, success
# - model: model name
# - type: "train" or "test"
# - recall: Recall@K (overall)
# - recall_w: Warm-start recall
# - recall_c: Cold-start recall
# - success: Total success rate

