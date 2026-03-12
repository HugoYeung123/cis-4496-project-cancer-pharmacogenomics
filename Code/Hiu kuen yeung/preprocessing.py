"""
preprocessing.py
Preprocessing pipeline and visualization for GDSC dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Preprocessing function

def preprocess_gdsc_data(filepath, target_col='LN_IC50', test_size=0.2, random_state=42):

    # Load data
    df = pd.read_csv(filepath)

    # Drop identifier columns
    cols_to_drop = [
        'NLME_RESULT_ID', 'NLME_CURVE_ID', 'COSMIC_ID', 'SANGER_MODEL_ID',
        'COMPANY_ID', 'WEBRELEASE', 'DATASET', 'DRUG_ID'
    ]
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Identify numeric and categorical features
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Split data
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Fit preprocessor on training data and transform
    X_train_processed = preprocessor.fit_transform(X_train_raw)
    X_test_processed = preprocessor.transform(X_test_raw)

    # Get feature names after preprocessing
    try:
        feature_names = (numeric_features +
                         list(preprocessor.named_transformers_['cat']
                              .named_steps['onehot']
                              .get_feature_names_out(categorical_features)))
    except Exception:
        feature_names = [f'feature_{i}' for i in range(X_train_processed.shape[1])]

    return (X_train_processed, X_test_processed, y_train, y_test,
            preprocessor, feature_names, X_train_raw, X_test_raw)


# Visualization functions

def plot_missing_values(X_train_raw):
    """
    Plot missing values per feature before imputation.
    """
    missing_before = X_train_raw.isnull().sum()
    missing_before = missing_before[missing_before > 0].sort_values(ascending=False)

    if not missing_before.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=missing_before.values, y=missing_before.index)
        plt.title('Missing Values per Feature (Before Imputation)')
        plt.xlabel('Number of Missing Values')
        plt.show()
    else:
        print("No missing values in the original data.")


def plot_numeric_histograms(X_train_raw, X_train_processed, numeric_features):
    """
    Plot histograms of numeric features before and after scaling.
    """
    n_features = len(numeric_features)
    if n_features == 0:
        print("No numeric features found.")
        return

    fig, axes = plt.subplots(n_features, 2, figsize=(12, n_features * 3))
    if n_features == 1:
        axes = axes.reshape(1, -1)

    for i, feat in enumerate(numeric_features):
        # Original data
        orig_vals = X_train_raw[feat].dropna()
        # Scaled data (index in processed array)
        feat_idx = numeric_features.index(feat)
        scaled_vals = X_train_processed[:, feat_idx]

        axes[i, 0].hist(orig_vals, bins=30, edgecolor='black', alpha=0.7)
        axes[i, 0].set_title(f'Original: {feat}')

        axes[i, 1].hist(scaled_vals, bins=30, edgecolor='black', alpha=0.7, color='orange')
        axes[i, 1].set_title(f'Scaled: {feat}')

    plt.tight_layout()
    plt.show()


def plot_numeric_boxplots(X_train_raw, X_train_processed, numeric_features):
    """
    Plot boxplots of numeric features before and after scaling.
    """
    n_features = len(numeric_features)
    if n_features == 0:
        print("No numeric features found.")
        return

    fig, axes = plt.subplots(n_features, 2, figsize=(12, n_features * 2.5))
    if n_features == 1:
        axes = axes.reshape(1, -1)

    for i, feat in enumerate(numeric_features):
        orig_vals = X_train_raw[feat].dropna()
        feat_idx = numeric_features.index(feat)
        scaled_vals = X_train_processed[:, feat_idx]

        axes[i, 0].boxplot(orig_vals)
        axes[i, 0].set_title(f'Original: {feat}')
        axes[i, 0].set_ylabel(feat)

        axes[i, 1].boxplot(scaled_vals)
        axes[i, 1].set_title(f'Scaled: {feat}')
        axes[i, 1].set_ylabel(f'{feat} (scaled)')

    plt.tight_layout()
    plt.show()


def plot_categorical_frequencies(X_train_raw, categorical_features, top_n=15):
    """
    Plot top_n category frequencies for each categorical feature.
    """
    n_features = len(categorical_features)
    if n_features == 0:
        print("No categorical features found.")
        return

    fig, axes = plt.subplots(n_features, 1, figsize=(10, n_features * 4))
    if n_features == 1:
        axes = [axes]

    for i, feat in enumerate(categorical_features):
        top_cats = X_train_raw[feat].value_counts().head(top_n)
        axes[i].barh(top_cats.index, top_cats.values, color='steelblue')
        axes[i].set_title(f'{feat} (total unique: {X_train_raw[feat].nunique()})')
        axes[i].set_xlabel('Frequency')

    plt.tight_layout()
    plt.show()

    # Print cardinalities
    print("\nCategorical feature cardinalities:")
    for feat in categorical_features:
        print(f"{feat}: {X_train_raw[feat].nunique()} unique values")