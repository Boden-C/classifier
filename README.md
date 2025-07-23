Research into C AI Vulnerability Detection

## Overview

This project is an advanced machine learning pipeline for automated vulnerability detection in C functions. Leveraging state-of-the-art feature engineering, program analysis, and robust model selection, it aims to push the boundaries of AI-driven software security research. The system is designed for extensibility, reproducibility, and high performance, making it suitable for both academic research and enterprise-scale deployment.

## Motivation

Software vulnerabilities in C code remain a critical threat to system security. Manual code review is labor-intensive and error-prone. This project addresses the challenge by applying modern AI techniques to automatically classify C functions as vulnerable or non-vulnerable, enabling scalable, consistent, and rapid security analysis.

## Key Features

-   **End-to-End Pipeline**: From raw C code ingestion to feature extraction, model training, and evaluation, the workflow is fully automated and checkpointed for reliability.
-   **Hybrid Feature Engineering**: Combines manual string-based metrics, AST (Abstract Syntax Tree) analysis, and TF-IDF vectorization to capture both syntactic and semantic code properties.
-   **AST Parsing with pycparser**: Utilizes pycparser and a custom fake libc to robustly parse C code, even in the presence of complex or incomplete constructs.
-   **Advanced Model Selection**: Employs Bayesian hyperparameter optimization (BayesSearchCV) across multiple classifiers (SVM, Logistic Regression, Decision Trees, Random Forests, XGBoost) for optimal performance.
-   **Class Imbalance Handling**: Integrates class weighting and stratified sampling to address real-world data skew, improving minority class recall.
-   **Rich Visualization**: Provides in-depth analysis of feature importance, model performance, and hyperparameter landscapes using seaborn, matplotlib, and parallel coordinates plots.
-   **Reproducibility**: All intermediate data and models are checkpointed with versioned pickles, ensuring experiments are fully reproducible.
-   **Scalability**: Designed to handle large datasets and high-dimensional feature spaces efficiently using sparse matrices and parallel computation.

## Technical Approach

1. **Data Preparation**

    - Cleans and deduplicates raw C function datasets.
    - Preprocesses code with GCC and fake libc includes for accurate parsing.
    - Parses each function to an AST and stores both the raw string and AST representation.

2. **Feature Extraction**

    - Manual extraction of code metrics (e.g., cyclomatic complexity, unsafe API usage, Halstead metrics).
    - TF-IDF vectorization of both code strings and AST string representations.
    - Feature selection via mutual information and variance analysis.

3. **Model Training & Optimization**

    - Multiple classifiers are trained and tuned using Bayesian optimization.
    - Custom feature combiners allow weighted fusion of manual and automated features.
    - Cross-validation and ROC/AUC analysis ensure robust evaluation.

4. **Evaluation & Visualization**
    - Generates detailed classification reports, confusion matrices, and ROC curves.
    - Visualizes hyperparameter effects and feature importances for interpretability.

## Usage

1. **Environment Setup**

    - Python 3.8+
    - Install dependencies: `pip install -r requirements.txt`
    - Ensure GCC and pycparser_fake_libc are available for AST extraction.

2. **Data Processing**

    - Place raw C function data in `src/data/raw.csv`.
    - Run `src/data/cleanup.py` to deduplicate and clean the dataset.
    - Execute `src/extract/parse_to_dataframe.py` to parse and checkpoint the processed DataFrame.

3. **Feature Engineering**

    - Use `features.ipynb` to extract, analyze, and save feature matrices.

4. **Model Training**

    - Run `train.ipynb` to train, optimize, and evaluate models.
    - Results and models are saved as pickle files for further analysis.

5. **Visualization**
    - Use `display.ipynb` for advanced result visualization and model interpretation.

## Research Impact

This project demonstrates the feasibility and effectiveness of AI-driven static analysis for vulnerability detection. Its modular design and comprehensive feature set make it a strong foundation for further research in program analysis, explainable AI, and secure software engineering. The pipeline is adaptable to new feature types, model architectures, and programming languages, supporting ongoing innovation in the field.

## Citation

If you use this project in your research, please cite it as:

> Boden, [Your Name]. "AI-Powered C Vulnerability Detection: An End-to-End Machine Learning Pipeline." 2024. https://github.com/[your-repo/classifier]

## License

Distributed under the Creative Commons CC0 1.0 Universal license. See LICENSE for details.
