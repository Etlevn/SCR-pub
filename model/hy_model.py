"""
Hybrid Recommender models combining CF and ML approaches.

This module implements various hybrid algorithms:
1. Weighted - Simple weighted combination of CF and ML scores
2. Stacking - Ensemble using CF and ML as base models
3. Feature-CF - CF embeddings as features for ML
4. Switching - Context-based switching between CF and ML
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from config import hy_config as config
from model.cf_model import create_cf_model
from model.ml_model import create_model as create_ml_model, get_scores as get_ml_scores
from prep.hy_prep import HybridDataset


class HybridModel:
    """
    Base class for Hybrid models.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.dataset = None
    
    def fit(self, dataset: HybridDataset) -> None:
        """Train the hybrid model."""
        raise NotImplementedError
    
    def score_candidates(
        self,
        group_df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Score candidates for a group.
        
        Args:
            group_df: DataFrame for one group with features
        
        Returns:
            Array of scores
        """
        raise NotImplementedError


class WeightedHybridModel(HybridModel):
    """
    Weighted combination of CF and ML scores.
    
    final_score = α × cf_score + (1-α) × ml_score
    """
    
    def __init__(self, cf_model_name: str = None, ml_model_name: str = None):
        super().__init__("weighted")
        self.cf_model = None
        self.ml_model = None
        self.cf_weight = config.WEIGHTED_CF_WEIGHT
        self.cf_model_name = cf_model_name  # Must be provided
        self.ml_model_name = ml_model_name  # Must be provided
    
    def fit(self, dataset: HybridDataset) -> None:
        """Train both CF and ML models."""
        self.dataset = dataset
        
        print(f"Training Weighted Hybrid (CF weight={self.cf_weight:.2f})...")
        
        # Train CF model
        print(f"  Training CF model: {self.cf_model_name}")
        self.cf_model = create_cf_model(self.cf_model_name)
        self.cf_model.fit(dataset.cf_dataset)
        
        # Train ML model
        print(f"  Training ML model: {self.ml_model_name}")
        self.ml_model = create_ml_model(self.ml_model_name)
        X_train = dataset.ml_train_df[dataset.feature_columns]
        y_train = dataset.ml_train_df['label']
        self.ml_model.fit(X_train, y_train)
        
        print("Weighted Hybrid training completed.")
    
    def score_candidates(self, group_df: pd.DataFrame) -> np.ndarray:
        """Score using weighted combination."""
        accid = str(group_df['accid'].iloc[0])
        candidate_permnos = group_df['permno'].tolist()
        
        # Get CF scores
        cf_scores = self.cf_model.score_candidates(accid, candidate_permnos)
        
        # Get ML scores
        X = group_df[self.dataset.feature_columns]
        ml_scores = get_ml_scores(self.ml_model, X)
        
        # Normalize scores to [0, 1]
        cf_scores_norm = self._normalize(cf_scores)
        ml_scores_norm = self._normalize(ml_scores)
        
        # Weighted combination
        final_scores = self.cf_weight * cf_scores_norm + (1 - self.cf_weight) * ml_scores_norm
        
        return final_scores
    
    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1]."""
        if len(scores) == 0:
            return scores
        min_score = scores.min()
        max_score = scores.max()
        if max_score == min_score:
            return np.ones_like(scores) * 0.5
        return (scores - min_score) / (max_score - min_score)


class StackingHybridModel(HybridModel):
    """
    Stacking ensemble using CF and ML as base models.
    
    Base models: Multiple CF and ML models
    Meta-learner: Logistic Regression
    """
    
    def __init__(self, cf_model_name: str = None, ml_model_name: str = None):
        super().__init__("stacking")
        self.cf_models = []
        self.ml_models = []
        self.meta_model = None
        self.cf_model_name = cf_model_name  # Must be provided
        self.ml_model_name = ml_model_name  # Must be provided
        # Use logistic regression as default meta-learner
        self.meta_model_name = "logistic_l2"
    
    def fit(self, dataset: HybridDataset) -> None:
        """Train base models and meta-learner."""
        self.dataset = dataset
        
        print(f"Training Stacking Hybrid...")
        
        # Train CF base model
        print(f"  Training CF base model: {self.cf_model_name}")
        cf_model = create_cf_model(self.cf_model_name)
        cf_model.fit(dataset.cf_dataset)
        self.cf_models.append(cf_model)
        
        # Train ML base model
        print(f"  Training ML base model: {self.ml_model_name}")
        X_train = dataset.ml_train_df[dataset.feature_columns]
        y_train = dataset.ml_train_df['label']
        
        ml_model = create_ml_model(self.ml_model_name)
        ml_model.fit(X_train, y_train)
        self.ml_models.append(ml_model)
        
        # Generate meta-features from base models
        print("  Generating meta-features...")
        meta_features = self._generate_meta_features(dataset.ml_train_df)
        
        # Train meta-learner
        print(f"  Training meta-learner: {self.meta_model_name}")
        self.meta_model = create_ml_model(self.meta_model_name)
        self.meta_model.fit(meta_features, y_train)
        
        print("Stacking Hybrid training completed.")
    
    def _generate_meta_features(self, df: pd.DataFrame) -> np.ndarray:
        """Generate meta-features from base model predictions."""
        meta_features_list = []
        
        # CF model predictions
        for cf_model in self.cf_models:
            cf_scores = []
            for _, row in df.iterrows():
                accid = str(row['accid'])
                permno = row['permno']
                score = cf_model.score_candidates(accid, [permno])[0]
                cf_scores.append(score)
            meta_features_list.append(cf_scores)
        
        # ML model predictions
        X = df[self.dataset.feature_columns]
        for ml_model in self.ml_models:
            ml_scores = get_ml_scores(ml_model, X)
            meta_features_list.append(ml_scores)
        
        # Stack features
        meta_features = np.column_stack(meta_features_list)
        return meta_features
    
    def score_candidates(self, group_df: pd.DataFrame) -> np.ndarray:
        """Score using stacking ensemble."""
        # Generate meta-features for this group
        meta_features = self._generate_meta_features(group_df)
        
        # Predict with meta-learner
        final_scores = get_ml_scores(self.meta_model, meta_features)
        
        return final_scores


class FeatureCFModel(HybridModel):
    """
    CF embeddings as features for ML model.
    
    Augments original features with CF user/item embeddings.
    """
    
    def __init__(self, cf_model_name: str = None, ml_model_name: str = None):
        super().__init__("feature_cf")
        self.cf_model = None
        self.ml_model = None
        self.cf_model_name = cf_model_name  # Must be provided
        self.ml_model_name = ml_model_name  # Must be provided
    
    def fit(self, dataset: HybridDataset) -> None:
        """Train CF model and ML model with augmented features."""
        self.dataset = dataset
        
        print(f"Training Feature-CF Hybrid...")
        
        # Train CF model
        print(f"  Training CF model: {self.cf_model_name}")
        self.cf_model = create_cf_model(self.cf_model_name)
        self.cf_model.fit(dataset.cf_dataset)
        
        # Generate augmented features
        print("  Generating augmented features...")
        X_train_aug = self._augment_features(dataset.ml_train_df)
        y_train = dataset.ml_train_df['label']
        
        # Train ML model on augmented features
        print(f"  Training ML model: {self.ml_model_name}")
        self.ml_model = create_ml_model(self.ml_model_name)
        self.ml_model.fit(X_train_aug, y_train)
        
        print("Feature-CF Hybrid training completed.")
    
    def _augment_features(self, df: pd.DataFrame) -> np.ndarray:
        """Augment features with CF embeddings."""
        # Original features
        X_orig = df[self.dataset.feature_columns].values
        
        # Determine actual embedding dimension from CF model
        if hasattr(self.cf_model, 'user_factors') and self.cf_model.user_factors is not None:
            actual_dim = self.cf_model.user_factors.shape[1]
        else:
            actual_dim = config.FEATURE_CF_EMBEDDING_DIM
        
        # Use the minimum of configured and actual dimensions
        embedding_dim = min(config.FEATURE_CF_EMBEDDING_DIM, actual_dim)
        
        # CF embeddings
        cf_embeddings = []
        for _, row in df.iterrows():
            accid = str(row['accid'])
            permno = row['permno']
            
            # Get user embedding
            user_idx = self.dataset.cf_dataset.get_user_idx(accid)
            if user_idx is not None and hasattr(self.cf_model, 'user_factors') and self.cf_model.user_factors is not None:
                user_emb = self.cf_model.user_factors[user_idx][:embedding_dim]
                # Pad if necessary
                if len(user_emb) < embedding_dim:
                    user_emb = np.pad(user_emb, (0, embedding_dim - len(user_emb)), mode='constant')
            else:
                user_emb = np.zeros(embedding_dim)
            
            # Get item embedding
            item_idx = self.dataset.cf_dataset.get_item_idx(permno)
            if item_idx is not None and hasattr(self.cf_model, 'item_factors') and self.cf_model.item_factors is not None:
                item_emb = self.cf_model.item_factors[item_idx][:embedding_dim]
                # Pad if necessary
                if len(item_emb) < embedding_dim:
                    item_emb = np.pad(item_emb, (0, embedding_dim - len(item_emb)), mode='constant')
            else:
                item_emb = np.zeros(embedding_dim)
            
            # Concatenate embeddings (ensure consistent shape)
            emb = np.concatenate([user_emb, item_emb])
            cf_embeddings.append(emb)
        
        # Convert to array (now all embeddings have the same shape)
        cf_embeddings = np.array(cf_embeddings)
        
        # Concatenate original features and CF embeddings
        X_aug = np.concatenate([X_orig, cf_embeddings], axis=1)
        
        return X_aug
    
    def score_candidates(self, group_df: pd.DataFrame) -> np.ndarray:
        """Score using ML model with augmented features."""
        X_aug = self._augment_features(group_df)
        final_scores = get_ml_scores(self.ml_model, X_aug)
        return final_scores


class SwitchingHybridModel(HybridModel):
    """
    Context-based switching between CF and ML.
    
    Uses CF for warm-start users (>= threshold interactions)
    Uses ML for cold-start users (< threshold interactions)
    """
    
    def __init__(self, cf_model_name: str = None, ml_model_name: str = None):
        super().__init__("switching")
        self.cf_model = None
        self.ml_model = None
        self.threshold = config.SWITCHING_INTERACTION_THRESHOLD
        self.cf_model_name = cf_model_name  # Must be provided
        self.ml_model_name = ml_model_name  # Must be provided
    
    def fit(self, dataset: HybridDataset) -> None:
        """Train both CF and ML models."""
        self.dataset = dataset
        
        print(f"Training Switching Hybrid (threshold={self.threshold})...")
        
        # Train CF model
        print(f"  Training CF model: {self.cf_model_name}")
        self.cf_model = create_cf_model(self.cf_model_name)
        self.cf_model.fit(dataset.cf_dataset)
        
        # Train ML model
        print(f"  Training ML model: {self.ml_model_name}")
        self.ml_model = create_ml_model(self.ml_model_name)
        X_train = dataset.ml_train_df[dataset.feature_columns]
        y_train = dataset.ml_train_df['label']
        self.ml_model.fit(X_train, y_train)
        
        print("Switching Hybrid training completed.")
    
    def score_candidates(self, group_df: pd.DataFrame) -> np.ndarray:
        """Score using CF or ML based on user interaction count."""
        accid = str(group_df['accid'].iloc[0])
        
        # Check user interaction count
        interaction_count = self.dataset.get_user_interaction_count(accid)
        
        if interaction_count >= self.threshold:
            # Use CF for warm-start users
            candidate_permnos = group_df['permno'].tolist()
            scores = self.cf_model.score_candidates(accid, candidate_permnos)
        else:
            # Use ML for cold-start users
            X = group_df[self.dataset.feature_columns]
            scores = get_ml_scores(self.ml_model, X)
        
        return scores


def create_hybrid_model(model_name: str) -> HybridModel:
    """
    Factory function to create Hybrid models.
    
    Args:
        model_name: Name of the model. Can be:
            - Base name: "weighted", "stacking", "feature_cf", "switching"
            - With ML suffix: "weighted_logistic_l2", "stacking_linear_svm", etc.
            - With CF and ML suffix: "weighted_nmf_logistic_l2", "stacking_als_linear_svm", etc.
    
    Returns:
        HybridModel instance
    """
    name = model_name.lower()
    
    # Parse model name to extract base type, CF model, and ML model
    # Format: {hybrid_type}_{cf_model}_{ml_model}
    # Examples:
    #   - "weighted_nmf_logistic_l2" -> base="weighted", cf="nmf", ml="logistic_l2"
    #   - "feature_cf_als_linear_svm" -> base="feature_cf", cf="als", ml="linear_svm"
    #   - "weighted_logistic_l2" -> base="weighted", cf=None, ml="logistic_l2"
    #   - "weighted" -> base="weighted", cf=None, ml=None
    
    parts = name.split('_')
    
    # Determine base type
    if parts[0] == "feature" and len(parts) > 1 and parts[1] == "cf":
        base_type = "feature_cf"
        remaining_parts = parts[2:]  # Everything after "feature_cf"
    else:
        base_type = parts[0]
        remaining_parts = parts[1:]  # Everything after base type
    
    # Known CF models
    cf_models = ["als", "bpr", "nmf", "svd", "knn", "item", "user"]
    
    # Parse remaining parts to extract CF and ML models
    cf_model_override = None
    ml_model_override = None
    
    if remaining_parts:
        # Check if first remaining part is a CF model
        if remaining_parts[0] in cf_models:
            cf_model_override = remaining_parts[0]
            # Handle special cases like "item_cf" or "user_cf"
            if len(remaining_parts) > 1 and remaining_parts[1] == "cf":
                cf_model_override = f"{remaining_parts[0]}_cf"
                ml_model_override = "_".join(remaining_parts[2:]) if len(remaining_parts) > 2 else None
            else:
                ml_model_override = "_".join(remaining_parts[1:]) if len(remaining_parts) > 1 else None
        else:
            # No CF model specified, treat all as ML model
            ml_model_override = "_".join(remaining_parts)
    
    # Validate that CF and ML models are provided
    if not cf_model_override or not ml_model_override:
        raise ValueError(
            f"Model name must include both CF and ML models. "
            f"Format: {{hybrid_type}}_{{cf_model}}_{{ml_model}}. "
            f"Got: {model_name}"
        )
    
    # Create model instance with CF and ML models
    if base_type == "weighted":
        return WeightedHybridModel(cf_model_override, ml_model_override)
    elif base_type == "stacking":
        return StackingHybridModel(cf_model_override, ml_model_override)
    elif base_type == "feature_cf":
        return FeatureCFModel(cf_model_override, ml_model_override)
    elif base_type == "switching":
        return SwitchingHybridModel(cf_model_override, ml_model_override)
    else:
        raise ValueError(f"Unknown Hybrid model: {model_name}")


if __name__ == "__main__":
    """
    Test Hybrid model creation and training.
    """
    print("="*60)
    print("Testing Hybrid Model")
    print("="*60)
    
    from prep.hy_prep import load_hybrid_data
    
    # Load data
    print("\nLoading Hybrid data...")
    dataset = load_hybrid_data()
    
    # Test Weighted model
    print("\n" + "="*60)
    print("Testing Weighted Hybrid Model")
    print("="*60)
    
    weighted_model = create_hybrid_model("weighted")
    weighted_model.fit(dataset)
    
    print("\n" + "="*60)
    print("Hybrid Model Test Completed!")
    print("="*60)

