#!/usr/bin/env python3
"""
train_baseline.py

Train baseline ML models to predict QAOA performance from graph features and parameters.
Tests if the dataset contains useful signal for parameter prediction.

Usage:
    python tests/param_optimization/train_baseline.py dataset.parquet --target approximation_ratio
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

# Optional: XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def load_dataset(path: str) -> pd.DataFrame:
    """Load dataset."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    else:
        return pd.read_csv(path)


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Prepare features for ML, handling JSON columns and categoricals."""
    df = df.copy()
    
    # Parse JSON angle columns if present
    if 'init_beta' in df.columns and df['init_beta'].dtype == object:
        try:
            df['beta_mean'] = df['init_beta'].apply(
                lambda x: np.mean(json.loads(x)) if x and isinstance(x, str) else 0.0
            )
            df['beta_std'] = df['init_beta'].apply(
                lambda x: np.std(json.loads(x)) if x and isinstance(x, str) else 0.0
            )
        except:
            pass
    
    if 'init_gamma' in df.columns and df['init_gamma'].dtype == object:
        try:
            df['gamma_mean'] = df['init_gamma'].apply(
                lambda x: np.mean(json.loads(x)) if x and isinstance(x, str) else 0.0
            )
            df['gamma_std'] = df['init_gamma'].apply(
                lambda x: np.std(json.loads(x)) if x and isinstance(x, str) else 0.0
            )
        except:
            pass
    
    # Encode categorical variables
    categorical_cols = ['optimizer', 'graph_type']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
    
    # Select feature columns (exclude targets, IDs, raw JSON)
    exclude_cols = ['graph_id', 'seed', 'init_beta', 'init_gamma', 
                    'final_energy', 'cut_value', 'approximation_ratio', 
                    'success_prob', 'iterations', 'runtime', 'converged',
                    'optimal_cut', 'optimizer', 'graph_type']
    
    feature_cols = [col for col in df.columns 
                    if col not in exclude_cols and df[col].dtype in [np.number, np.int64, np.float64]]
    
    return df, feature_cols


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics."""
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'spearman': spearmanr(y_true, y_pred)[0],
    }


def train_baseline(df: pd.DataFrame, target: str = 'approximation_ratio', 
                   test_size: float = 0.2, random_state: int = 42):
    """Train and evaluate baseline models."""
    
    print("=" * 80)
    print(f"BASELINE MODEL TRAINING: {target}")
    print("=" * 80)
    
    # Prepare features
    df_processed, feature_cols = prepare_features(df)
    
    print(f"\n📋 Features ({len(feature_cols)}):")
    print(f"  {', '.join(feature_cols)}")
    
    # Check target exists
    if target not in df.columns:
        print(f"❌ Target '{target}' not found in dataset!")
        print(f"Available: {df.columns.tolist()}")
        return
    
    # Remove rows with missing target
    df_clean = df_processed.dropna(subset=[target] + feature_cols)
    print(f"\n📊 Dataset: {len(df_clean)} samples (dropped {len(df_processed) - len(df_clean)} with NaN)")
    
    if len(df_clean) < 50:
        print("❌ Too few samples for training!")
        return
    
    X = df_clean[feature_cols].values
    y = df_clean[target].values
    
    # Split: try graph-level split if possible
    if 'graph_id' in df_clean.columns:
        print("\n🔀 Using graph-level split (prevents data leakage)")
        unique_graphs = df_clean['graph_id'].unique()
        np.random.seed(random_state)
        test_graphs = np.random.choice(unique_graphs, 
                                       size=int(len(unique_graphs) * test_size), 
                                       replace=False)
        test_mask = df_clean['graph_id'].isin(test_graphs)
        X_train, X_test = X[~test_mask], X[test_mask]
        y_train, y_test = y[~test_mask], y[test_mask]
    else:
        print("\n🔀 Using random split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test:  {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Baseline: always predict mean
    print(f"\n📊 BASELINE (predict mean)")
    y_pred_mean = np.full_like(y_test, y_train.mean())
    metrics_baseline = evaluate_model(y_test, y_pred_mean)
    for metric, val in metrics_baseline.items():
        print(f"  {metric}: {val:.4f}")
    
    # Random Forest
    print(f"\n🌲 RANDOM FOREST")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, 
                               random_state=random_state, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    metrics_rf = evaluate_model(y_test, y_pred_rf)
    for metric, val in metrics_rf.items():
        print(f"  {metric}: {val:.4f}")
    
    # Feature importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    print(f"\n  Top 10 features:")
    for i, idx in enumerate(indices):
        print(f"    {i+1}. {feature_cols[idx]}: {importances[idx]:.4f}")
    
    # XGBoost if available
    if XGBOOST_AVAILABLE:
        print(f"\n🚀 XGBOOST")
        xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, 
                                     learning_rate=0.1, random_state=random_state)
        xgb_model.fit(X_train_scaled, y_train)
        y_pred_xgb = xgb_model.predict(X_test_scaled)
        metrics_xgb = evaluate_model(y_test, y_pred_xgb)
        for metric, val in metrics_xgb.items():
            print(f"  {metric}: {val:.4f}")
    
    # Cross-validation on train set
    print(f"\n✅ 5-FOLD CV (on train set)")
    cv_scores = cross_val_score(rf, X_train_scaled, y_train, 
                                cv=5, scoring='r2', n_jobs=-1)
    print(f"  R² scores: {cv_scores}")
    print(f"  Mean R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Interpretation
    print(f"\n💡 INTERPRETATION")
    if metrics_rf['r2'] > 0.5:
        print(f"  ✓ Good signal! R²={metrics_rf['r2']:.3f} indicates parameters affect outcomes")
    elif metrics_rf['r2'] > 0.2:
        print(f"  ⚠️  Moderate signal (R²={metrics_rf['r2']:.3f}). Consider:")
        print(f"     - More diverse graphs")
        print(f"     - Real QAOA runs (--use_qiskit)")
        print(f"     - More samples per graph")
    else:
        print(f"  ❌ Weak signal (R²={metrics_rf['r2']:.3f}). Dataset may not be useful!")
        print(f"     - Check if parameters actually vary outcomes")
        print(f"     - Ensure using real QAOA, not just brute-force labels")
    
    if abs(metrics_rf['spearman']) > 0.7:
        print(f"  ✓ Strong rank correlation (ρ={metrics_rf['spearman']:.3f})")
        print(f"    Model can identify better parameter settings")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline ML models on QAOA dataset")
    parser.add_argument("dataset", type=str, help="Path to dataset")
    parser.add_argument("--target", type=str, default="approximation_ratio",
                        help="Target variable to predict")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test set fraction")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    try:
        df = load_dataset(args.dataset)
        train_baseline(df, target=args.target, test_size=args.test_size, 
                      random_state=args.seed)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
