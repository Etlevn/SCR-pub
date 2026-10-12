from __future__ import annotations

from typing import Any

from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LogisticRegression,
    Perceptron,
    Ridge,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.svm import LinearSVC

from config import ml_config as config


def create_model(model_name: str | None = None) -> Any:
    """
    Create and return a linear model instance.
    
    All models use default hyperparameters from scikit-learn. Only the model
    type (l1_ratio, solver) and random_state (for reproducibility) are specified.

    Args:
        model_name: Name of the model to create. If None, uses config.MODEL_NAME.
                    Supported values:
        - logistic_l2: LogisticRegression with L2 penalty (l1_ratio=0)
        - logistic_l1: LogisticRegression with L1 penalty (l1_ratio=1)
        - logistic_enet: LogisticRegression with elastic-net penalty (l1_ratio=0.5)
        - lasso: Lasso regression
        - ridge: Ridge regression
        - elasticnet: ElasticNet regression
        - linear_svm: LinearSVC
        - sgd_hinge: SGDClassifier with hinge loss (SVM-like)
        - sgd_log: SGDClassifier with log loss (logistic regression-like)
        - sgd_modified_huber: SGDClassifier with modified Huber loss
        - perceptron: Perceptron (simple linear classifier)
        - passive_aggressive: SGDClassifier configured as PassiveAggressive (deprecated class replacement)
        - ridge_classifier: RidgeClassifier (classification version of Ridge)
    """
    if model_name is None:
        # Handle both single string and list from config
        if isinstance(config.MODEL_NAME, list):
            if len(config.MODEL_NAME) == 0:
                raise ValueError("MODEL_NAME list is empty")
            model_name = config.MODEL_NAME[0]  # Use first model as default
        else:
            model_name = config.MODEL_NAME
    
    name = model_name.lower()
    
    # Common kwargs: only random_state if specified
    random_state_kwargs = {}
    if config.RANDOM_SEED is not None:
        random_state_kwargs["random_state"] = config.RANDOM_SEED

    if name == "logistic_l2":
        # Use l1_ratio=0 instead of deprecated penalty="l2"
        return LogisticRegression(
            l1_ratio=0,
            solver="lbfgs",  # Required for L2 penalty
            **random_state_kwargs,
        )

    if name == "logistic_l1":
        # Use l1_ratio=1 instead of deprecated penalty="l1"
        # liblinear supports L1 for binary classification
        return LogisticRegression(
            l1_ratio=1,
            solver="liblinear",  # Required for L1 penalty
            **random_state_kwargs,
        )

    if name == "logistic_enet":
        # Use l1_ratio instead of deprecated penalty="elasticnet"
        # saga supports elastic-net penalty
        # l1_ratio=0.5 gives equal weight to L1 and L2
        return LogisticRegression(
            l1_ratio=0.5,
            solver="saga",  # Required for elastic-net penalty
            **random_state_kwargs,
        )

    if name == "lasso":
        return Lasso(**random_state_kwargs)

    if name == "ridge":
        return Ridge(**random_state_kwargs)

    if name == "elasticnet":
        return ElasticNet(**random_state_kwargs)

    if name == "linear_svm":
        return LinearSVC(**random_state_kwargs)

    if name == "sgd_hinge":
        # SGD with hinge loss (similar to SVM)
        return SGDClassifier(
            loss="hinge",
            **random_state_kwargs,
        )

    if name == "sgd_log":
        # SGD with log loss (similar to logistic regression)
        return SGDClassifier(
            loss="log_loss",
            **random_state_kwargs,
        )

    if name == "sgd_modified_huber":
        # SGD with modified Huber loss (smooth and robust)
        return SGDClassifier(
            loss="modified_huber",
            **random_state_kwargs,
        )

    if name == "perceptron":
        return Perceptron(**random_state_kwargs)

    if name == "passive_aggressive":
        # Use SGDClassifier as replacement for deprecated PassiveAggressiveClassifier
        # This matches the behavior of PassiveAggressiveClassifier
        return SGDClassifier(
            loss="hinge",
            penalty=None,
            learning_rate="pa1",
            eta0=1.0,
            **random_state_kwargs,
        )

    if name == "ridge_classifier":
        return RidgeClassifier(**random_state_kwargs)

    raise ValueError(f"Unsupported model name: {model_name}")


def get_scores(model: Any, x) -> Any:
    """
    Compute a 1D score array for each row in x, regardless of model type.

    The scores are used only for ranking candidates within each group, so
    they do not need to be calibrated probabilities.
    """
    # Preferred: probability of the positive class if available
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        # Assuming binary classification, take probability of class 1
        return proba[:, 1]

    # For SVM and some linear models, use decision_function
    if hasattr(model, "decision_function"):
        return model.decision_function(x)

    # Fallback: use raw predictions as scores (e.g., for regression models)
    return model.predict(x)

