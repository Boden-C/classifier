import pickle
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
import sys
import os
from scipy.sparse import csr_matrix
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from src.features.manual_string_extraction import manual_string_extraction
from src.utils import find_latest_checkpoint


def analyze_feature_mutual_info(
    matrix: csr_matrix,
    target: np.ndarray,
    features: List[str],
    mi_params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Analyze feature importance using mutual information.
    Returns DataFrame with columns: 'feature', 'mutual_info'.
    """
    params = mi_params or {}
    scores = mutual_info_classif(matrix, target, **params)
    result = pd.DataFrame({"feature": features, "mutual_info": scores})
    result = result.sort_values("mutual_info", ascending=False)
    return result


def analyze_feature_variance(
    matrix: csr_matrix,
    features: List[str],
    var_params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Analyze feature importance using variance threshold.
    Returns DataFrame with columns: 'feature', 'variance'.
    """
    params = var_params or {}
    selector = VarianceThreshold(**params)
    selector.fit(matrix)
    scores = selector.variances_
    result = pd.DataFrame({"feature": features, "variance": scores})
    result = result.sort_values("variance", ascending=False)
    return result


def trim_features_by_mutual_info(
    matrix: csr_matrix,
    target: np.ndarray,
    features: List[str],
    threshold: float,
    mi_params: Optional[Dict[str, Any]] = None,
) -> Tuple[csr_matrix, list[str]]:
    """
    Trim features with mutual information <= threshold.
    Returns (trimmed_matrix, trimmed_features).
    """
    params = mi_params or {}
    scores = mutual_info_classif(matrix, target, **params)
    mask = scores > threshold
    trimmed_matrix = matrix[:, mask]
    trimmed_features = [f for f, keep in zip(features, mask) if keep]
    return trimmed_matrix, trimmed_features


def trim_features_by_variance(
    matrix: csr_matrix,
    features: List[str],
    threshold: float,
    var_params: Optional[Dict[str, Any]] = None,
) -> Tuple[csr_matrix, list[str]]:
    """
    Trim features with variance <= threshold.
    Returns (trimmed_matrix, trimmed_features).
    """
    params = var_params or {}
    selector = VarianceThreshold(threshold=threshold, **params)
    trimmed_matrix = selector.fit_transform(matrix)
    mask = selector.get_support()
    trimmed_features = [f for f, keep in zip(features, mask) if keep]
    return trimmed_matrix, trimmed_features
