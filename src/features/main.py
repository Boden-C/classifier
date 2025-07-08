import pickle
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import sys
import os
from scipy.sparse import csr_matrix
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from src.features.manual_string_extraction import manual_string_extraction
from src.utils import find_latest_checkpoint


def summarize_matrix(matrix: csr_matrix, features: List[str] = None) -> None:
    """
    Print a summary of a sparse feature matrix.
    Shows shape, number of nonzero elements, and first 10 feature names if available.
    """
    print(f"Matrix shape: {matrix.shape}")
    print(f"Nonzero elements: {matrix.nnz}")
    if features:
        print(f"First 10 feature names: {features[:10]}")
