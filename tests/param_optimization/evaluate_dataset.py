#!/usr/bin/env python3
"""
evaluate_dataset.py

Analyse statistique et visualisation d'un dataset QAOA.
Génère des statistiques descriptives, distributions, corrélations.

Usage:
    python tests/param_optimization/evaluate_dataset.py dataset.parquet
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np


def load_dataset(path: str) -> pd.DataFrame:
    """Load dataset from parquet or CSV."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    else:
        return pd.read_csv(path)


def analyze_dataset(df: pd.DataFrame):
    """Print comprehensive dataset analysis."""
    
    print("=" * 80)
    print("DATASET EVALUATION REPORT")
    print("=" * 80)
    
    print(f"\n📊 BASIC STATS")
    print(f"  Total experiments: {len(df)}")
    print(f"  Unique graphs: {df['graph_id'].nunique()}")
    print(f"  Columns: {len(df.columns)}")
    
    print(f"\n📐 GRAPH CHARACTERISTICS")
    if 'num_nodes' in df.columns:
        print(f"  Nodes: {df['num_nodes'].min()}-{df['num_nodes'].max()} "
              f"(mean: {df['num_nodes'].mean():.1f})")
    if 'num_edges' in df.columns:
        print(f"  Edges: {df['num_edges'].min()}-{df['num_edges'].max()} "
              f"(mean: {df['num_edges'].mean():.1f})")
    if 'density' in df.columns:
        print(f"  Density: {df['density'].min():.3f}-{df['density'].max():.3f} "
              f"(mean: {df['density'].mean():.3f})")
    if 'graph_type' in df.columns:
        print(f"  Graph types: {df['graph_type'].value_counts().to_dict()}")
    
    print(f"\n⚙️  QAOA PARAMETERS")
    if 'p' in df.columns:
        print(f"  p values: {sorted(df['p'].unique())}")
        print(f"  p distribution: {df['p'].value_counts().sort_index().to_dict()}")
    if 'optimizer' in df.columns:
        print(f"  Optimizers: {df['optimizer'].value_counts().to_dict()}")
    
    print(f"\n📈 PERFORMANCE METRICS")
    metrics = ['final_energy', 'approximation_ratio', 'success_prob', 
               'iterations', 'runtime']
    for metric in metrics:
        if metric in df.columns:
            vals = df[metric].dropna()
            print(f"  {metric}:")
            print(f"    mean={vals.mean():.4f}, std={vals.std():.4f}")
            print(f"    min={vals.min():.4f}, max={vals.max():.4f}")
    
    print(f"\n🔍 DATA QUALITY")
    print(f"  Missing values per column:")
    missing = df.isnull().sum()
    for col, count in missing[missing > 0].items():
        print(f"    {col}: {count} ({100*count/len(df):.1f}%)")
    
    if 'converged' in df.columns:
        conv_rate = df['converged'].mean()
        print(f"  Convergence rate: {100*conv_rate:.1f}%")
    
    print(f"\n📊 VARIABILITY CHECK")
    # Check if there's enough variance in target variables
    if 'final_energy' in df.columns:
        energy_std = df['final_energy'].std()
        energy_range = df['final_energy'].max() - df['final_energy'].min()
        print(f"  Final energy range: {energy_range:.3f} (std: {energy_std:.3f})")
        
        # Check if energy varies within same graph
        if 'graph_id' in df.columns:
            within_graph_std = df.groupby('graph_id')['final_energy'].std().mean()
            print(f"  Avg energy std within graph: {within_graph_std:.4f}")
            if within_graph_std < 0.01:
                print(f"  ⚠️  WARNING: Very low variance within graphs!")
                print(f"     Parameters may not affect outcomes meaningfully.")
    
    if 'approximation_ratio' in df.columns:
        ratio_std = df['approximation_ratio'].std()
        print(f"  Approximation ratio std: {ratio_std:.4f}")
        if ratio_std < 0.05:
            print(f"  ⚠️  WARNING: Low variance in approximation ratios!")
    
    print(f"\n🔗 CORRELATIONS (top absolute correlations with final_energy)")
    if 'final_energy' in df.columns:
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()['final_energy'].abs().sort_values(ascending=False)
            print(corr.head(10).to_string())
    
    print(f"\n💡 RECOMMENDATIONS")
    
    # Check dataset size
    if len(df) < 500:
        print(f"  ⚠️  Dataset is small ({len(df)} samples)")
        print(f"     Recommend: Generate at least 1000-5000 samples")
    
    # Check graph diversity
    if 'num_nodes' in df.columns:
        node_range = df['num_nodes'].max() - df['num_nodes'].min()
        if node_range < 4:
            print(f"  ⚠️  Limited node size range ({node_range})")
            print(f"     Recommend: Increase n_min/n_max spread")
    
    # Check density diversity
    if 'density' in df.columns:
        density_unique = df['density'].nunique()
        if density_unique < 5:
            print(f"  ⚠️  Low density diversity ({density_unique} unique values)")
            print(f"     Recommend: Use wider density_range")
    
    # Check if using real QAOA
    if 'approximation_ratio' in df.columns:
        if df['approximation_ratio'].std() < 0.05:
            print(f"  💡 Low variance suggests simulated data")
            print(f"     Recommend: Use --use_qiskit for real QAOA runs")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate QAOA dataset quality")
    parser.add_argument("dataset", type=str, help="Path to dataset (.parquet or .csv)")
    args = parser.parse_args()
    
    try:
        df = load_dataset(args.dataset)
        analyze_dataset(df)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
