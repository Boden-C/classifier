from typing import Tuple, List, Any
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix


def extract_string_tfidf(
    df: pd.DataFrame,
    text_col: str = "x_str",
    analyzer: str = "char",
    ngram_range: Tuple[int, int] = (3, 5),
    max_features: int = 100000,
) -> Tuple[TfidfVectorizer, csr_matrix]:
    """
    Extract TF-IDF features from a DataFrame column containing C function code as str.

    Args:
        df: DataFrame with a column of C function code.
        text_col: Name of the column containing code as text.
        analyzer: 'char' or 'word'.
        ngram_range: Tuple for n-gram range.
        max_features: Maximum number of features.

    Returns:
        (vectorizer, feature_matrix):
            vectorizer: Fitted TfidfVectorizer.
            feature_matrix: Sparse matrix of TF-IDF features.
    """
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame.")
    vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, max_features=max_features)
    X = vectorizer.fit_transform(df[text_col].astype(str))
    return vectorizer, X


def extract_ast_tfidf(
    df: pd.DataFrame,
    text_col: str = "x_ast",
    ngram_range: Tuple[int, int] = (3, 5),
    max_features: int = 100000,
) -> Tuple[TfidfVectorizer, csr_matrix]:
    """
    Extract TF-IDF features from a DataFrame column containing C function code as str.

    Args:
        df: DataFrame with a column of C function code.
        text_col: Name of the column containing code as text.
        analyzer: 'char' or 'word'.
        ngram_range: Tuple for n-gram range.
        max_features: Maximum number of features.

    Returns:
        (vectorizer, feature_matrix):
            vectorizer: Fitted TfidfVectorizer.
            feature_matrix: Sparse matrix of TF-IDF features.
    """
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame.")
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, max_features=max_features)
    X = vectorizer.fit_transform(df[text_col].show().astype(str))
    return vectorizer, X

