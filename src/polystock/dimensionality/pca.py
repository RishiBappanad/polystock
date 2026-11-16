from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

# print("Module loaded")


def build_preprocess_pipeline(scaler: str = "standard") -> Pipeline:
    # print(f"scaler={scaler}")
    if scaler == "robust":
        scaler_step = RobustScaler()
    elif scaler == "standard":
        scaler_step = StandardScaler()
    else:
        raise ValueError("scaler must be one of: standard, robust")

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler_step),
        ]
    )


def prepare_numeric_features(
    df: pd.DataFrame,
    drop_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    # print(f"input_shape={df.shape}, requested_drop_cols={drop_cols}")
    columns_to_drop = [col for col in (drop_cols or []) if col in df.columns]
    working_df = df.drop(columns=columns_to_drop)
    # print(f"applied_drop_cols={columns_to_drop}")

    numeric_df = working_df.select_dtypes(include=[np.number]).copy()
    if numeric_df.empty:
        raise ValueError("No numeric columns available for PCA after preprocessing")

    # print(f"numeric_shape={numeric_df.shape}")

    return numeric_df, columns_to_drop


def components_for_variance(cumulative_variance: np.ndarray, threshold: float) -> int:
    # print(f"threshold={threshold}")
    return int(np.searchsorted(cumulative_variance, threshold) + 1)


def compute_feature_scores(
    feature_names: list[str],
    components: np.ndarray,
    explained_variance_ratio: np.ndarray,
    components_used: int,
) -> list[dict[str, float | str]]:
    # print(f"compute_feature_scores components_used={components_used}")
    weighted_abs_loadings = np.abs(components[:components_used]).T @ explained_variance_ratio[
        :components_used
    ]

    ranking = [
        {
            "feature": feature_name,
            "score": float(score),
        }
        for feature_name, score in zip(feature_names, weighted_abs_loadings)
    ]
    ranking.sort(key=lambda item: item["score"], reverse=True)
    return ranking


def run_domain_pca(
    csv_path: str | Path,
    domain_name: str,
    variance_threshold: float = 0.95,
    scaler: str = "standard",
    max_components: int | None = None,
    drop_cols: list[str] | None = None,
) -> dict[str, Any]:
    # print(f"domain={domain_name}, csv_path={csv_path}")
    dataset_path = Path(csv_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"CSV not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    # print(f"loaded_shape={df.shape}")
    numeric_df, dropped_cols = prepare_numeric_features(df, drop_cols=drop_cols)

    preprocess = build_preprocess_pipeline(scaler=scaler)
    transformed = preprocess.fit_transform(numeric_df)
    # print(f"transformed_shape={transformed.shape}")

    n_features = transformed.shape[1]
    if max_components is None:
        n_components = min(transformed.shape[0], n_features)
    else:
        n_components = min(max_components, transformed.shape[0], n_features)

    # print(f"n_features={n_features}, n_components={n_components}")

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(transformed)
    # print("PCA fit complete")

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    components_needed = components_for_variance(cumulative, variance_threshold)
    feature_scores = compute_feature_scores(
        feature_names=numeric_df.columns.tolist(),
        components=pca.components_,
        explained_variance_ratio=explained,
        components_used=components_needed,
    )
    # print(f"components_needed={components_needed}, variance_threshold={variance_threshold}")

    return {
        "domain": domain_name,
        "csv_path": str(dataset_path),
        "n_rows": int(df.shape[0]),
        "n_original_columns": int(df.shape[1]),
        "dropped_columns": dropped_cols,
        "n_numeric_features_used": int(n_features),
        "scaler": scaler,
        "variance_threshold": variance_threshold,
        "components_to_threshold": int(components_needed),
        "components_used_for_feature_scores": int(components_needed),
        "top_explained_variance_ratio": [float(x) for x in explained[:10]],
        "top_cumulative_explained_variance": [float(x) for x in cumulative[:10]],
        "feature_scores": feature_scores,
    }


def compare_domains_pca(
    stock_csv_path: str | Path,
    polymarket_csv_path: str | Path,
    variance_threshold: float = 0.95,
    scaler: str = "standard",
    max_components: int | None = None,
    stock_drop_cols: list[str] | None = None,
    polymarket_drop_cols: list[str] | None = None,
) -> dict[str, Any]:
    # print("Starting stock vs polymarket PCA comparison")
    stock = run_domain_pca(
        csv_path=stock_csv_path,
        domain_name="stock",
        variance_threshold=variance_threshold,
        scaler=scaler,
        max_components=max_components,
        drop_cols=stock_drop_cols,
    )

    polymarket = run_domain_pca(
        csv_path=polymarket_csv_path,
        domain_name="polymarket",
        variance_threshold=variance_threshold,
        scaler=scaler,
        max_components=max_components,
        drop_cols=polymarket_drop_cols,
    )
    # print("Both domain PCA runs complete")

    stock_components = stock["components_to_threshold"]
    poly_components = polymarket["components_to_threshold"]
    # print(f"stock_components={stock_components}, poly_components={poly_components}")

    if stock_components < poly_components:
        lower_domain = "stock"
    elif poly_components < stock_components:
        lower_domain = "polymarket"
    else:
        lower_domain = "tie"

    return {
        "comparison": {
            "variance_threshold": variance_threshold,
            "scaler": scaler,
            "lower_intrinsic_dimensionality_domain": lower_domain,
            "component_difference": abs(stock_components - poly_components),
        },
        "stock": stock,
        "polymarket": polymarket,
    }
