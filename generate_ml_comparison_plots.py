#!/usr/bin/env python3
"""
ML Model Comparison Visualization
==================================
Generates bar and line graphs comparing different ML models for UHI prediction.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    ExtraTreesRegressor, AdaBoostRegressor
)
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings('ignore')

# Try importing advanced ML libraries
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


def load_and_prepare_data():
    """Load and prepare UHI data for model comparison."""
    # Try to load actual data
    data_paths = [
        "/Users/harshita/Downloads/UHI_d_NZFL (1).csv",
        "/Users/harshita/Downloads/UHI_d_NZFL.csv",
        "data/UHI_d.csv"
    ]
    
    df = None
    for path in data_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"✓ Loaded data from {path}")
            break
    
    if df is None:
        # Generate synthetic data for demonstration
        print("⚠ No data file found. Generating synthetic data for demonstration...")
        np.random.seed(42)
        n_samples = 1000
        
        df = pd.DataFrame({
            'asphalt_ratio': np.random.uniform(0, 0.5, n_samples),
            'park_grass_ratio': np.random.uniform(0, 0.3, n_samples),
            'GnPR': np.random.uniform(0, 1, n_samples),
            'greenroof_ratio': np.random.uniform(0, 0.2, n_samples),
            'roadDensity': np.random.uniform(0, 0.6, n_samples),
            'bldDensity': np.random.uniform(0, 0.8, n_samples),
            'treeDensity': np.random.uniform(0, 0.4, n_samples),
            'avg_BH': np.random.uniform(5, 50, n_samples),
        })
        
        # Create UHI as function of features
        df['UHI_d'] = (
            0.15 + 
            0.08 * df['asphalt_ratio'] +
            0.05 * df['roadDensity'] +
            0.03 * df['bldDensity'] -
            0.06 * df['GnPR'] -
            0.04 * df['treeDensity'] -
            0.03 * df['park_grass_ratio'] +
            np.random.normal(0, 0.01, n_samples)
        )
    
    return df


def prepare_features(df, target_column='UHI_d'):
    """Prepare features and target for training."""
    feature_cols = [
        'asphalt_ratio', 'park_grass_ratio', 'GnPR', 'greenroof_ratio',
        'roadDensity', 'bldDensity', 'treeDensity', 'avg_BH',
        'parcel_grass_ratio', 'podium_grass_ratio', 'parcel_fp_ratio',
        'avg_GPR', 'parkRadius'
    ]
    
    available_cols = [c for c in feature_cols if c in df.columns]
    
    if not available_cols:
        available_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != target_column and 'UHI' not in c.upper()
        ]
    
    X = df[available_cols].copy().fillna(df[available_cols].median())
    y = df[target_column].copy()
    
    return X, y, available_cols


def get_models():
    """Get dictionary of models to compare."""
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        ),
        'Extra Trees': ExtraTreesRegressor(
            n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
        ),
        'AdaBoost': AdaBoostRegressor(
            n_estimators=100, learning_rate=0.1, random_state=42
        ),
        'Ridge': Ridge(alpha=1.0),
        'Bayesian Ridge': BayesianRidge(),
        'SVR': SVR(kernel='rbf', C=10, epsilon=0.01),
        'KNN': KNeighborsRegressor(n_neighbors=5),
        'Neural Network': MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=42
        ),
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBRegressor(
            n_estimators=100, max_depth=8, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbosity=0
        )
    
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = LGBMRegressor(
            n_estimators=100, max_depth=8, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbose=-1
        )
    
    return models


def train_and_evaluate_models(X, y, test_size=0.2):
    """Train all models and collect metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = get_models()
    results = []
    predictions = {}
    
    print("\n" + "="*60)
    print("Training and Evaluating Models...")
    print("="*60)
    
    for name, model in models.items():
        try:
            print(f"\nTraining {name}...", end=" ")
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            results.append({
                'Model': name,
                'R²': r2,
                'RMSE': rmse,
                'MAE': mae
            })
            
            predictions[name] = y_pred
            print(f"✓ R²={r2:.4f}, RMSE={rmse:.6f}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    results_df = pd.DataFrame(results).sort_values('R²', ascending=False)
    return results_df, predictions, y_test, scaler


def plot_bar_comparison(results_df, output_dir='output/plots'):
    """Generate bar chart comparing model metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    # Sort by R² for consistent ordering
    results_sorted = results_df.sort_values('R²', ascending=True)
    models = results_sorted['Model'].values
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
    
    # Bar chart 1: R² Score
    bars1 = axes[0].barh(models, results_sorted['R²'], color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_xlabel('R² Score', fontsize=12, fontweight='bold')
    axes[0].set_title('Model Comparison: R² Score', fontsize=14, fontweight='bold')
    axes[0].set_xlim(0, 1)
    axes[0].axvline(x=results_sorted['R²'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {results_sorted["R²"].mean():.3f}')
    axes[0].legend(loc='lower right')
    
    # Add value labels
    for bar, val in zip(bars1, results_sorted['R²']):
        axes[0].text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                     f'{val:.3f}', va='center', fontsize=9)
    
    # Bar chart 2: RMSE (lower is better)
    colors_rmse = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(models)))
    results_rmse = results_df.sort_values('RMSE', ascending=False)
    bars2 = axes[1].barh(results_rmse['Model'], results_rmse['RMSE'], 
                         color=colors_rmse, edgecolor='black', linewidth=0.5)
    axes[1].set_xlabel('RMSE (°C)', fontsize=12, fontweight='bold')
    axes[1].set_title('Model Comparison: RMSE (Lower is Better)', fontsize=14, fontweight='bold')
    axes[1].axvline(x=results_rmse['RMSE'].mean(), color='red', linestyle='--',
                    label=f'Mean: {results_rmse["RMSE"].mean():.5f}')
    axes[1].legend(loc='lower right')
    
    for bar, val in zip(bars2, results_rmse['RMSE']):
        axes[1].text(val + 0.0001, bar.get_y() + bar.get_height()/2,
                     f'{val:.5f}', va='center', fontsize=9)
    
    # Bar chart 3: MAE (lower is better)
    results_mae = results_df.sort_values('MAE', ascending=False)
    bars3 = axes[2].barh(results_mae['Model'], results_mae['MAE'],
                         color=plt.cm.YlOrRd(np.linspace(0.2, 0.8, len(models))),
                         edgecolor='black', linewidth=0.5)
    axes[2].set_xlabel('MAE (°C)', fontsize=12, fontweight='bold')
    axes[2].set_title('Model Comparison: MAE (Lower is Better)', fontsize=14, fontweight='bold')
    axes[2].axvline(x=results_mae['MAE'].mean(), color='red', linestyle='--',
                    label=f'Mean: {results_mae["MAE"].mean():.5f}')
    axes[2].legend(loc='lower right')
    
    for bar, val in zip(bars3, results_mae['MAE']):
        axes[2].text(val + 0.0001, bar.get_y() + bar.get_height()/2,
                     f'{val:.5f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'ml_model_comparison_bar.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ Bar chart saved to {filepath}")
    return filepath


def plot_line_comparison(predictions, y_test, results_df, output_dir='output/plots'):
    """Generate line chart comparing model predictions."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort predictions by actual values for visualization
    sort_idx = np.argsort(y_test.values)
    y_actual_sorted = y_test.values[sort_idx]
    
    # Select top 5 models by R² for cleaner visualization
    top_models = results_df.head(5)['Model'].values
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Line chart 1: Predictions vs Actual (sorted)
    sample_size = min(200, len(y_actual_sorted))
    sample_idx = np.linspace(0, len(y_actual_sorted)-1, sample_size, dtype=int)
    
    x_axis = np.arange(sample_size)
    
    # Plot actual values
    axes[0].plot(x_axis, y_actual_sorted[sample_idx], 'k-', linewidth=2.5, 
                 label='Actual UHI', alpha=0.8)
    
    # Plot predictions for top models
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_models)))
    for i, model_name in enumerate(top_models):
        if model_name in predictions:
            pred_sorted = predictions[model_name][sort_idx]
            axes[0].plot(x_axis, pred_sorted[sample_idx], '--', 
                        color=colors[i], linewidth=1.5, label=model_name, alpha=0.7)
    
    axes[0].set_xlabel('Sample Index (sorted by actual UHI)', fontsize=12)
    axes[0].set_ylabel('UHI Intensity (°C)', fontsize=12)
    axes[0].set_title('Model Predictions vs Actual Values (Top 5 Models)', 
                      fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper left', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Line chart 2: Prediction Error across samples
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    for i, model_name in enumerate(top_models):
        if model_name in predictions:
            pred_sorted = predictions[model_name][sort_idx]
            error = pred_sorted[sample_idx] - y_actual_sorted[sample_idx]
            axes[1].plot(x_axis, error, '-', color=colors[i], 
                        linewidth=1.2, label=model_name, alpha=0.7)
    
    axes[1].set_xlabel('Sample Index (sorted by actual UHI)', fontsize=12)
    axes[1].set_ylabel('Prediction Error (°C)', fontsize=12)
    axes[1].set_title('Prediction Error by Model (Predicted - Actual)', 
                      fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper left', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].fill_between(x_axis, -0.01, 0.01, alpha=0.2, color='green', label='±0.01°C band')
    
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'ml_model_comparison_line.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Line chart saved to {filepath}")
    return filepath


def plot_combined_metrics(results_df, output_dir='output/plots'):
    """Generate combined metrics visualization."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    models = results_df['Model'].values
    x = np.arange(len(models))
    width = 0.25
    
    # Normalize metrics for comparison (scale to 0-1)
    r2_vals = results_df['R²'].values
    rmse_norm = 1 - (results_df['RMSE'] / results_df['RMSE'].max()).values  # Invert so higher is better
    mae_norm = 1 - (results_df['MAE'] / results_df['MAE'].max()).values  # Invert so higher is better
    
    bars1 = ax.bar(x - width, r2_vals, width, label='R² Score', color='#2ecc71', edgecolor='black')
    bars2 = ax.bar(x, rmse_norm, width, label='RMSE Score (inverted)', color='#3498db', edgecolor='black')
    bars3 = ax.bar(x + width, mae_norm, width, label='MAE Score (inverted)', color='#e74c3c', edgecolor='black')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (0-1, higher is better)', fontsize=12, fontweight='bold')
    ax.set_title('ML Model Comparison: All Metrics Normalized', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add a horizontal line at mean R²
    ax.axhline(y=r2_vals.mean(), color='green', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'ml_model_comparison_combined.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Combined metrics chart saved to {filepath}")
    return filepath


def plot_scatter_actual_vs_predicted(predictions, y_test, results_df, output_dir='output/plots'):
    """Generate scatter plots of actual vs predicted for top models."""
    os.makedirs(output_dir, exist_ok=True)
    
    top_models = results_df.head(6)['Model'].values
    n_models = len(top_models)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, model_name in enumerate(top_models):
        if model_name in predictions:
            ax = axes[i]
            y_pred = predictions[model_name]
            
            ax.scatter(y_test, y_pred, alpha=0.5, s=20, c='blue')
            
            # Perfect prediction line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
            
            r2 = results_df[results_df['Model'] == model_name]['R²'].values[0]
            ax.set_xlabel('Actual UHI (°C)', fontsize=10)
            ax.set_ylabel('Predicted UHI (°C)', fontsize=10)
            ax.set_title(f'{model_name}\nR² = {r2:.4f}', fontsize=11, fontweight='bold')
            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Actual vs Predicted UHI Values by Model', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'ml_model_scatter_comparison.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Scatter comparison saved to {filepath}")
    return filepath


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("   ML MODEL COMPARISON VISUALIZATION")
    print("="*60)
    
    # Load data
    df = load_and_prepare_data()
    print(f"✓ Data loaded: {len(df)} samples")
    
    # Prepare features
    X, y, feature_cols = prepare_features(df)
    print(f"✓ Features prepared: {len(feature_cols)} features")
    
    # Train and evaluate models
    results_df, predictions, y_test, scaler = train_and_evaluate_models(X, y)
    
    # Print results table
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    bar_path = plot_bar_comparison(results_df)
    line_path = plot_line_comparison(predictions, y_test, results_df)
    combined_path = plot_combined_metrics(results_df)
    scatter_path = plot_scatter_actual_vs_predicted(predictions, y_test, results_df)
    
    # Save results to CSV
    results_df.to_csv('output/plots/model_comparison_results.csv', index=False)
    print(f"✓ Results saved to output/plots/model_comparison_results.csv")
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print(f"  • {bar_path}")
    print(f"  • {line_path}")
    print(f"  • {combined_path}")
    print(f"  • {scatter_path}")
    print(f"  • output/plots/model_comparison_results.csv")
    
    # Identify best model
    best_model = results_df.iloc[0]['Model']
    best_r2 = results_df.iloc[0]['R²']
    print(f"\n🏆 Best Model: {best_model} with R² = {best_r2:.4f}")


if __name__ == "__main__":
    main()
