"""
Collaborative Filtering models for candidate ranking.

This module implements various CF algorithms:
1. ALS (Alternating Least Squares) - implicit feedback
2. BPR (Bayesian Personalized Ranking)
3. Item-based Collaborative Filtering
4. User-based Collaborative Filtering
5. SVD (Singular Value Decomposition)
6. NMF (Non-negative Matrix Factorization)
7. KNN (K-Nearest Neighbors)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF as SklearnNMF
from sklearn.metrics.pairwise import cosine_similarity

from config import cf_config as config_cf
from prep.cf_prep import CFDataset


class CFModel:
    """
    Base class for Collaborative Filtering models.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.dataset = None
    
    def fit(self, dataset: CFDataset) -> None:
        """Train the CF model on the interaction matrix."""
        raise NotImplementedError
    
    def score_candidates(
        self,
        accid: str,
        candidate_permnos: List[float],
    ) -> np.ndarray:
        """
        Score a list of candidate items for a given user.
        
        Args:
            accid: User account ID
            candidate_permnos: List of candidate item IDs (permno)
        
        Returns:
            Array of scores (same length as candidate_permnos)
        """
        raise NotImplementedError


class ALSModel(CFModel):
    """
    Alternating Least Squares model for implicit feedback.
    
    Uses the implicit library implementation.
    """
    
    def __init__(self):
        super().__init__("als")
        self.item_popularity = None
    
    def fit(self, dataset: CFDataset) -> None:
        """Train ALS model on the interaction matrix."""
        self.dataset = dataset
        
        print(f"Training ALS with {config_cf.ALS_FACTORS} factors...")
        
        # Create ALS model
        self.model = AlternatingLeastSquares(
            factors=config_cf.ALS_FACTORS,
            regularization=config_cf.ALS_REGULARIZATION,
            iterations=config_cf.ALS_ITERATIONS,
            alpha=config_cf.ALS_ALPHA,
            use_gpu=config_cf.ALS_USE_GPU,
            random_state=config_cf.RANDOM_SEED,
        )
        
        # ALS expects item-user matrix (items × users)
        # We have user-item matrix (users × items), so transpose
        item_user_matrix = dataset.interaction_matrix.T.tocsr()
        
        # Fit the model
        self.model.fit(item_user_matrix, show_progress=config_cf.SHOW_PROGRESS)
        
        # Compute item popularity for cold-start handling
        from prep.cf_prep import get_item_popularity
        self.item_popularity = get_item_popularity(dataset)
        
        print("ALS training completed.")
    
    def score_candidates(
        self,
        accid: str,
        candidate_permnos: List[float],
    ) -> np.ndarray:
        """
        Score candidates using ALS model.
        
        For warm-start users: use ALS predictions
        For cold-start users: use popularity-based fallback
        """
        # Check if user is cold-start
        user_idx = self.dataset.get_user_idx(accid)
        
        if user_idx is None:
            # Cold-start user: use fallback strategy
            return self._cold_start_score(candidate_permnos)
        
        # Warm-start user: use ALS predictions
        scores = []
        # Note: ALS stores factors in item_factors (users) and user_factors (items)
        # because we transposed the matrix during training
        user_embedding = self.model.item_factors[user_idx]
        
        for permno in candidate_permnos:
            item_idx = self.dataset.get_item_idx(permno)
            
            if item_idx is None:
                # Cold-start item: use fallback
                score = self._cold_start_item_score(permno)
            else:
                # Warm-start item: compute dot product
                item_embedding = self.model.user_factors[item_idx]
                score = np.dot(user_embedding, item_embedding)
            
            scores.append(score)
        
        return np.array(scores)
    
    def _cold_start_score(self, candidate_permnos: List[float]) -> np.ndarray:
        """Handle cold-start users based on configured strategy."""
        strategy = config_cf.COLD_START_STRATEGY
        
        if strategy == "popularity":
            scores = []
            for permno in candidate_permnos:
                item_idx = self.dataset.get_item_idx(permno)
                if item_idx is not None:
                    score = self.item_popularity.get(item_idx, 0.0)
                else:
                    score = 0.0
                scores.append(score)
            return np.array(scores)
        
        elif strategy == "random":
            return np.random.rand(len(candidate_permnos))
        
        elif strategy == "zero":
            return np.zeros(len(candidate_permnos))
        
        elif strategy == "mean":
            # Use mean of all item scores
            mean_score = np.mean(list(self.item_popularity.values()))
            return np.full(len(candidate_permnos), mean_score)
        
        else:
            raise ValueError(f"Unknown cold-start strategy: {strategy}")
    
    def _cold_start_item_score(self, permno: float) -> float:
        """Handle cold-start items (not seen in training)."""
        # For cold-start items, return a low score
        return 0.0


class BPRModel(CFModel):
    """
    Bayesian Personalized Ranking model.
    
    Optimizes pairwise ranking loss for implicit feedback.
    """
    
    def __init__(self):
        super().__init__("bpr")
        self.item_popularity = None
    
    def fit(self, dataset: CFDataset) -> None:
        """Train BPR model on the interaction matrix."""
        self.dataset = dataset
        
        print(f"Training BPR with {config_cf.BPR_FACTORS} factors...")
        
        # Create BPR model
        self.model = BayesianPersonalizedRanking(
            factors=config_cf.BPR_FACTORS,
            learning_rate=config_cf.BPR_LEARNING_RATE,
            regularization=config_cf.BPR_REGULARIZATION,
            iterations=config_cf.BPR_ITERATIONS,
            random_state=config_cf.RANDOM_SEED,
        )
        
        # BPR expects item-user matrix (items × users)
        item_user_matrix = dataset.interaction_matrix.T.tocsr()
        
        # Fit the model
        self.model.fit(item_user_matrix, show_progress=config_cf.SHOW_PROGRESS)
        
        # Compute item popularity for cold-start handling
        from prep.cf_prep import get_item_popularity
        self.item_popularity = get_item_popularity(dataset)
        
        print("BPR training completed.")
    
    def score_candidates(
        self,
        accid: str,
        candidate_permnos: List[float],
    ) -> np.ndarray:
        """Score candidates using BPR model."""
        # Same logic as ALS
        user_idx = self.dataset.get_user_idx(accid)
        
        if user_idx is None:
            return self._cold_start_score(candidate_permnos)
        
        scores = []
        # Note: BPR stores factors in item_factors (users) and user_factors (items)
        # because we transposed the matrix during training
        user_embedding = self.model.item_factors[user_idx]
        
        for permno in candidate_permnos:
            item_idx = self.dataset.get_item_idx(permno)
            
            if item_idx is None:
                score = self._cold_start_item_score(permno)
            else:
                item_embedding = self.model.user_factors[item_idx]
                score = np.dot(user_embedding, item_embedding)
            
            scores.append(score)
        
        return np.array(scores)
    
    def _cold_start_score(self, candidate_permnos: List[float]) -> np.ndarray:
        """Handle cold-start users."""
        strategy = config_cf.COLD_START_STRATEGY
        
        if strategy == "popularity":
            scores = []
            for permno in candidate_permnos:
                item_idx = self.dataset.get_item_idx(permno)
                if item_idx is not None:
                    score = self.item_popularity.get(item_idx, 0.0)
                else:
                    score = 0.0
                scores.append(score)
            return np.array(scores)
        
        elif strategy == "random":
            return np.random.rand(len(candidate_permnos))
        
        elif strategy == "zero":
            return np.zeros(len(candidate_permnos))
        
        elif strategy == "mean":
            mean_score = np.mean(list(self.item_popularity.values()))
            return np.full(len(candidate_permnos), mean_score)
        
        else:
            raise ValueError(f"Unknown cold-start strategy: {strategy}")
    
    def _cold_start_item_score(self, permno: float) -> float:
        """Handle cold-start items."""
        return 0.0


class ItemCFModel(CFModel):
    """
    Item-based Collaborative Filtering.
    
    Computes item-item similarity and scores based on user's history.
    """
    
    def __init__(self):
        super().__init__("item_cf")
        self.item_similarity = None
        self.item_popularity = None
    
    def fit(self, dataset: CFDataset) -> None:
        """Compute item-item similarity matrix."""
        self.dataset = dataset
        
        print(f"Computing item-item similarity (method: {config_cf.ITEM_CF_SIMILARITY})...")
        
        # Compute item-item cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Get item-user matrix (items × users)
        item_user_matrix = dataset.interaction_matrix.T.tocsr()
        
        # Compute cosine similarity between items
        self.item_similarity = cosine_similarity(item_user_matrix, dense_output=False)
        
        # Compute item popularity for cold-start handling
        from prep.cf_prep import get_item_popularity
        self.item_popularity = get_item_popularity(dataset)
        
        print("Item-CF preparation completed.")
    
    def score_candidates(
        self,
        accid: str,
        candidate_permnos: List[float],
    ) -> np.ndarray:
        """
        Score candidates based on similarity to user's historical items.
        """
        user_idx = self.dataset.get_user_idx(accid)
        
        if user_idx is None:
            # Cold-start user: use popularity
            return self._cold_start_score(candidate_permnos)
        
        # Get user's historical items
        user_items = self.dataset.interaction_matrix[user_idx].nonzero()[1]
        
        if len(user_items) == 0:
            # User has no history: use popularity
            return self._cold_start_score(candidate_permnos)
        
        # Score each candidate based on similarity to user's items
        scores = []
        for permno in candidate_permnos:
            item_idx = self.dataset.get_item_idx(permno)
            
            if item_idx is None:
                # Cold-start item
                score = 0.0
            else:
                # Compute weighted sum of similarities
                similarities = self.item_similarity[item_idx, user_items].toarray().flatten()
                
                # Use top-K most similar items
                k = min(config_cf.ITEM_CF_K, len(similarities))
                top_k_sim = np.partition(similarities, -k)[-k:]
                score = np.mean(top_k_sim) if k > 0 else 0.0
            
            scores.append(score)
        
        return np.array(scores)
    
    def _cold_start_score(self, candidate_permnos: List[float]) -> np.ndarray:
        """Handle cold-start users."""
        strategy = config_cf.COLD_START_STRATEGY
        
        if strategy == "popularity":
            scores = []
            for permno in candidate_permnos:
                item_idx = self.dataset.get_item_idx(permno)
                if item_idx is not None:
                    score = self.item_popularity.get(item_idx, 0.0)
                else:
                    score = 0.0
                scores.append(score)
            return np.array(scores)
        
        elif strategy == "random":
            return np.random.rand(len(candidate_permnos))
        
        elif strategy == "zero":
            return np.zeros(len(candidate_permnos))
        
        elif strategy == "mean":
            mean_score = np.mean(list(self.item_popularity.values()))
            return np.full(len(candidate_permnos), mean_score)
        
        else:
            raise ValueError(f"Unknown cold-start strategy: {strategy}")


class UserCFModel(CFModel):
    """
    User-based Collaborative Filtering.
    
    Computes user-user similarity and scores based on similar users' preferences.
    """
    
    def __init__(self):
        super().__init__("user_cf")
        self.user_similarity = None
        self.item_popularity = None
    
    def fit(self, dataset: CFDataset) -> None:
        """Compute user-user similarity matrix."""
        self.dataset = dataset
        
        print(f"Computing user-user similarity (method: {config_cf.USER_CF_SIMILARITY})...")
        
        # Compute user-user cosine similarity
        user_item_matrix = dataset.interaction_matrix.tocsr()
        
        # Compute cosine similarity between users
        self.user_similarity = cosine_similarity(user_item_matrix, dense_output=False)
        
        # Compute item popularity for cold-start handling
        from prep.cf_prep import get_item_popularity
        self.item_popularity = get_item_popularity(dataset)
        
        print("User-CF preparation completed.")
    
    def score_candidates(
        self,
        accid: str,
        candidate_permnos: List[float],
    ) -> np.ndarray:
        """
        Score candidates based on similar users' preferences.
        """
        user_idx = self.dataset.get_user_idx(accid)
        
        if user_idx is None:
            # Cold-start user: use popularity
            return self._cold_start_score(candidate_permnos)
        
        # Get top-K similar users (excluding self)
        user_similarities = self.user_similarity[user_idx].toarray().flatten()
        user_similarities[user_idx] = -1  # Exclude self
        
        k = min(config_cf.USER_CF_K, len(user_similarities) - 1)
        if k <= 0:
            return self._cold_start_score(candidate_permnos)
        
        top_k_users = np.argpartition(user_similarities, -k)[-k:]
        top_k_sims = user_similarities[top_k_users]
        
        # Normalize similarities
        if top_k_sims.sum() > 0:
            top_k_sims = top_k_sims / top_k_sims.sum()
        
        # Score each candidate based on similar users' interactions
        scores = []
        for permno in candidate_permnos:
            item_idx = self.dataset.get_item_idx(permno)
            
            if item_idx is None:
                # Cold-start item
                score = 0.0
            else:
                # Weighted sum of similar users' interactions
                item_ratings = self.dataset.interaction_matrix[top_k_users, item_idx].toarray().flatten()
                score = np.dot(top_k_sims, item_ratings)
            
            scores.append(score)
        
        return np.array(scores)
    
    def _cold_start_score(self, candidate_permnos: List[float]) -> np.ndarray:
        """Handle cold-start users."""
        strategy = config_cf.COLD_START_STRATEGY
        
        if strategy == "popularity":
            scores = []
            for permno in candidate_permnos:
                item_idx = self.dataset.get_item_idx(permno)
                if item_idx is not None:
                    score = self.item_popularity.get(item_idx, 0.0)
                else:
                    score = 0.0
                scores.append(score)
            return np.array(scores)
        
        elif strategy == "random":
            return np.random.rand(len(candidate_permnos))
        
        elif strategy == "zero":
            return np.zeros(len(candidate_permnos))
        
        elif strategy == "mean":
            mean_score = np.mean(list(self.item_popularity.values()))
            return np.full(len(candidate_permnos), mean_score)
        
        else:
            raise ValueError(f"Unknown cold-start strategy: {strategy}")


class SVDModel(CFModel):
    """
    Singular Value Decomposition (SVD) for Collaborative Filtering.
    
    Uses truncated SVD to decompose the interaction matrix.
    """
    
    def __init__(self):
        super().__init__("svd")
        self.user_factors = None
        self.item_factors = None
        self.singular_values = None
        self.item_popularity = None
    
    def fit(self, dataset: CFDataset) -> None:
        """Train SVD model on the interaction matrix."""
        self.dataset = dataset
        
        print(f"Training SVD with {config_cf.SVD_FACTORS} factors...")
        
        # Perform truncated SVD
        k = min(config_cf.SVD_FACTORS, 
                min(dataset.interaction_matrix.shape) - 1)
        
        U, sigma, Vt = svds(dataset.interaction_matrix.astype(np.float64), k=k)
        
        # Store factors
        self.user_factors = U  # (n_users, k)
        self.singular_values = sigma  # (k,)
        self.item_factors = Vt.T  # (n_items, k)
        
        # Compute item popularity for cold-start handling
        from prep.cf_prep import get_item_popularity
        self.item_popularity = get_item_popularity(dataset)
        
        print("SVD training completed.")
    
    def score_candidates(
        self,
        accid: str,
        candidate_permnos: List[float],
    ) -> np.ndarray:
        """Score candidates using SVD model."""
        user_idx = self.dataset.get_user_idx(accid)
        
        if user_idx is None:
            return self._cold_start_score(candidate_permnos)
        
        # Get user embedding
        user_embedding = self.user_factors[user_idx] * self.singular_values
        
        scores = []
        for permno in candidate_permnos:
            item_idx = self.dataset.get_item_idx(permno)
            
            if item_idx is None:
                score = 0.0
            else:
                item_embedding = self.item_factors[item_idx]
                score = np.dot(user_embedding, item_embedding)
            
            scores.append(score)
        
        return np.array(scores)
    
    def _cold_start_score(self, candidate_permnos: List[float]) -> np.ndarray:
        """Handle cold-start users."""
        strategy = config_cf.COLD_START_STRATEGY
        
        if strategy == "popularity":
            scores = []
            for permno in candidate_permnos:
                item_idx = self.dataset.get_item_idx(permno)
                if item_idx is not None:
                    score = self.item_popularity.get(item_idx, 0.0)
                else:
                    score = 0.0
                scores.append(score)
            return np.array(scores)
        
        elif strategy == "random":
            return np.random.rand(len(candidate_permnos))
        
        elif strategy == "zero":
            return np.zeros(len(candidate_permnos))
        
        elif strategy == "mean":
            mean_score = np.mean(list(self.item_popularity.values()))
            return np.full(len(candidate_permnos), mean_score)
        
        else:
            raise ValueError(f"Unknown cold-start strategy: {strategy}")


class NMFModel(CFModel):
    """
    Non-negative Matrix Factorization (NMF) for Collaborative Filtering.
    
    Decomposes the interaction matrix into non-negative factors.
    """
    
    def __init__(self):
        super().__init__("nmf")
        self.user_factors = None
        self.item_factors = None
        self.item_popularity = None
    
    def fit(self, dataset: CFDataset) -> None:
        """Train NMF model on the interaction matrix."""
        self.dataset = dataset
        
        print(f"Training NMF with {config_cf.NMF_FACTORS} factors...")
        
        # Convert to dense for NMF (sklearn NMF doesn't support sparse well)
        # For large matrices, this might be memory-intensive
        interaction_dense = dataset.interaction_matrix.toarray()
        
        # Create and fit NMF model
        nmf = SklearnNMF(
            n_components=config_cf.NMF_FACTORS,
            init='random',
            random_state=config_cf.RANDOM_SEED,
            max_iter=config_cf.NMF_ITERATIONS,
            alpha_W=config_cf.NMF_REGULARIZATION,
            alpha_H=config_cf.NMF_REGULARIZATION,
        )
        
        self.user_factors = nmf.fit_transform(interaction_dense)  # (n_users, k)
        self.item_factors = nmf.components_.T  # (n_items, k)
        
        # Compute item popularity for cold-start handling
        from prep.cf_prep import get_item_popularity
        self.item_popularity = get_item_popularity(dataset)
        
        print("NMF training completed.")
    
    def score_candidates(
        self,
        accid: str,
        candidate_permnos: List[float],
    ) -> np.ndarray:
        """Score candidates using NMF model."""
        user_idx = self.dataset.get_user_idx(accid)
        
        if user_idx is None:
            return self._cold_start_score(candidate_permnos)
        
        # Get user embedding
        user_embedding = self.user_factors[user_idx]
        
        scores = []
        for permno in candidate_permnos:
            item_idx = self.dataset.get_item_idx(permno)
            
            if item_idx is None:
                score = 0.0
            else:
                item_embedding = self.item_factors[item_idx]
                score = np.dot(user_embedding, item_embedding)
            
            scores.append(score)
        
        return np.array(scores)
    
    def _cold_start_score(self, candidate_permnos: List[float]) -> np.ndarray:
        """Handle cold-start users."""
        strategy = config_cf.COLD_START_STRATEGY
        
        if strategy == "popularity":
            scores = []
            for permno in candidate_permnos:
                item_idx = self.dataset.get_item_idx(permno)
                if item_idx is not None:
                    score = self.item_popularity.get(item_idx, 0.0)
                else:
                    score = 0.0
                scores.append(score)
            return np.array(scores)
        
        elif strategy == "random":
            return np.random.rand(len(candidate_permnos))
        
        elif strategy == "zero":
            return np.zeros(len(candidate_permnos))
        
        elif strategy == "mean":
            mean_score = np.mean(list(self.item_popularity.values()))
            return np.full(len(candidate_permnos), mean_score)
        
        else:
            raise ValueError(f"Unknown cold-start strategy: {strategy}")


class KNNModel(CFModel):
    """
    K-Nearest Neighbors based Collaborative Filtering.
    
    Uses KNN to find similar items and score based on user's history.
    Similar to Item-CF but with explicit KNN implementation.
    """
    
    def __init__(self):
        super().__init__("knn")
        self.item_similarity = None
        self.item_popularity = None
    
    def fit(self, dataset: CFDataset) -> None:
        """Compute item-item similarity using KNN approach."""
        self.dataset = dataset
        
        print(f"Computing KNN-based similarity (K={config_cf.KNN_K})...")
        
        # Get item-user matrix (items × users)
        item_user_matrix = dataset.interaction_matrix.T.tocsr()
        
        # Compute cosine similarity between items
        self.item_similarity = cosine_similarity(item_user_matrix, dense_output=False)
        
        # Compute item popularity for cold-start handling
        from prep.cf_prep import get_item_popularity
        self.item_popularity = get_item_popularity(dataset)
        
        print("KNN preparation completed.")
    
    def score_candidates(
        self,
        accid: str,
        candidate_permnos: List[float],
    ) -> np.ndarray:
        """
        Score candidates using KNN approach.
        """
        user_idx = self.dataset.get_user_idx(accid)
        
        if user_idx is None:
            return self._cold_start_score(candidate_permnos)
        
        # Get user's historical items with their ratings
        user_items = self.dataset.interaction_matrix[user_idx].nonzero()[1]
        user_ratings = self.dataset.interaction_matrix[user_idx, user_items].toarray().flatten()
        
        if len(user_items) == 0:
            return self._cold_start_score(candidate_permnos)
        
        # Score each candidate
        scores = []
        for permno in candidate_permnos:
            item_idx = self.dataset.get_item_idx(permno)
            
            if item_idx is None:
                score = 0.0
            else:
                # Get similarities to user's items
                similarities = self.item_similarity[item_idx, user_items].toarray().flatten()
                
                # Find K nearest neighbors
                k = min(config_cf.KNN_K, len(similarities))
                if k > 0:
                    # Get top-K similar items
                    top_k_indices = np.argpartition(similarities, -k)[-k:]
                    top_k_sims = similarities[top_k_indices]
                    top_k_ratings = user_ratings[top_k_indices]
                    
                    # Weighted average
                    if top_k_sims.sum() > 0:
                        score = np.dot(top_k_sims, top_k_ratings) / top_k_sims.sum()
                    else:
                        score = 0.0
                else:
                    score = 0.0
            
            scores.append(score)
        
        return np.array(scores)
    
    def _cold_start_score(self, candidate_permnos: List[float]) -> np.ndarray:
        """Handle cold-start users."""
        strategy = config_cf.COLD_START_STRATEGY
        
        if strategy == "popularity":
            scores = []
            for permno in candidate_permnos:
                item_idx = self.dataset.get_item_idx(permno)
                if item_idx is not None:
                    score = self.item_popularity.get(item_idx, 0.0)
                else:
                    score = 0.0
                scores.append(score)
            return np.array(scores)
        
        elif strategy == "random":
            return np.random.rand(len(candidate_permnos))
        
        elif strategy == "zero":
            return np.zeros(len(candidate_permnos))
        
        elif strategy == "mean":
            mean_score = np.mean(list(self.item_popularity.values()))
            return np.full(len(candidate_permnos), mean_score)
        
        else:
            raise ValueError(f"Unknown cold-start strategy: {strategy}")


def create_cf_model(model_name: str) -> CFModel:
    """
    Factory function to create CF models.
    
    Args:
        model_name: Name of the model 
            ("als", "bpr", "item_cf", "user_cf", "svd", "nmf", "knn")
    
    Returns:
        CFModel instance
    """
    name = model_name.lower()
    
    if name == "als":
        return ALSModel()
    elif name == "bpr":
        return BPRModel()
    elif name == "item_cf":
        return ItemCFModel()
    elif name == "user_cf":
        return UserCFModel()
    elif name == "svd":
        return SVDModel()
    elif name == "nmf":
        return NMFModel()
    elif name == "knn":
        return KNNModel()
    else:
        raise ValueError(f"Unknown CF model: {model_name}")


if __name__ == "__main__":
    """
    Test CF model creation and training.
    """
    print("="*60)
    print("Testing CF Model")
    print("="*60)
    
    from prep.cf_prep import load_cf_data
    
    # Load data
    print("\nLoading CF data...")
    dataset = load_cf_data()
    
    # Test ALS model
    print("\n" + "="*60)
    print("Testing ALS Model")
    print("="*60)
    
    als_model = create_cf_model("als")
    als_model.fit(dataset)
    
    # Test scoring on a sample user
    test_accid = list(dataset.user_to_idx.keys())[0]
    test_permnos = list(dataset.item_to_idx.keys())[:10]
    
    print(f"\nTesting scoring for user: {test_accid}")
    print(f"Scoring {len(test_permnos)} candidate items...")
    
    scores = als_model.score_candidates(test_accid, test_permnos)
    print(f"Scores: {scores}")
    
    print("\n" + "="*60)
    print("CF Model Test Completed!")
    print("="*60)

