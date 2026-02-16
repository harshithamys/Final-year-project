#!/usr/bin/env python3
"""
Urban Heat Island (UHI) Analysis System - Main Entry Point
==========================================================

This script demonstrates the complete UHI analysis workflow including:
1. Data loading and preprocessing
2. Hotspot detection (DBSCAN + Grid-based)
3. Mitigation strategy recommendations
4. UHI prediction modeling
5. AR/VR visualization generation

Usage:
    python main.py

Author: UHI Analysis Team
"""

import os
import sys
import logging
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import UHI Analysis modules
from uhi_analysis.core import UHIDataLoader, HotspotDetector, GridBasedAnalyzer
from uhi_analysis.mitigation import MitigationRecommender
from uhi_analysis.visualization import VisualizationManager, ThreeJSGenerator
from uhi_analysis.prediction import (
    UHIPredictionModel, EnsembleUHIModel,
    AdvancedUHIModel, StackingUHIModel, ModelComparison, AutoMLUHI
)
from uhi_analysis.urban_generator import UrbanLandscapeGenerator


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def demonstrate_data_loading():
    """Demonstrate data loading functionality."""
    print_header("1. DATA LOADING & PREPROCESSING")
    
    # Initialize loader
    loader = UHIDataLoader()
    
    # Define data paths
    daytime_path = "/Users/harshita/Downloads/UHI_d_NZFL (1).csv"
    nighttime_path = "/Users/harshita/Downloads/UHI_n_NZFL (1).csv"
    
    # Load data
    print("Loading UHI datasets...")
    df_day, df_night = loader.load_multiple(daytime_path, nighttime_path)
    
    print(f"✓ Daytime data loaded: {len(df_day)} records")
    print(f"✓ Nighttime data loaded: {len(df_night)} records")
    
    # Show data statistics
    print(f"\nDaytime UHI Statistics:")
    print(f"  - Mean: {df_day['UHI_d'].mean():.4f}°C")
    print(f"  - Max:  {df_day['UHI_d'].max():.4f}°C")
    print(f"  - Min:  {df_day['UHI_d'].min():.4f}°C")
    print(f"  - Std:  {df_day['UHI_d'].std():.4f}°C")
    
    # Add synthetic coordinates
    df_day = loader.add_synthetic_coordinates(df_day)
    print(f"\n✓ Synthetic coordinates added")
    
    # Preprocess
    df_day = loader.preprocess(df_day)
    print(f"✓ Data preprocessed")
    
    return df_day, df_night


def demonstrate_hotspot_detection(df: pd.DataFrame):
    """Demonstrate DBSCAN and grid-based hotspot detection."""
    print_header("2. HOTSPOT DETECTION")
    
    # DBSCAN Detection
    print("Running DBSCAN clustering...")
    detector = HotspotDetector(
        eps=0.5,
        min_samples=3,
        uhi_threshold_percentile=75
    )
    
    result = detector.detect(df, uhi_column='UHI_d')
    
    print(f"\nDBSCAN Results:")
    print(f"  - Clusters found: {result.n_clusters}")
    print(f"  - Hotspot points: {result.statistics.get('hotspot_points', 0)}")
    print(f"  - Hotspot percentage: {result.statistics.get('hotspot_percentage', 0):.1f}%")
    print(f"  - Mean UHI (hotspots): {result.statistics.get('mean_uhi_hotspots', 0):.4f}°C")
    
    # Grid-based Analysis
    print("\nRunning Grid-based analysis...")
    grid_analyzer = GridBasedAnalyzer(
        grid_rows=10,
        grid_cols=10,
        hotspot_threshold_percentile=75
    )
    
    hotspot_cells = grid_analyzer.analyze(df, uhi_column='UHI_d')
    summary = grid_analyzer.get_hotspot_summary()
    
    print(f"\nGrid Analysis Results:")
    print(f"  - Hotspot cells: {summary['n_hotspot_cells']}")
    print(f"  - Total cells: {summary['total_cells']}")
    print(f"  - Hotspot area: {summary['hotspot_percentage']:.1f}%")
    print(f"  - Avg hotspot UHI: {summary['avg_hotspot_uhi']:.4f}°C")
    
    return result, grid_analyzer


def demonstrate_mitigation_recommendations(df: pd.DataFrame):
    """Demonstrate mitigation strategy recommendations."""
    print_header("3. MITIGATION RECOMMENDATIONS")
    
    # Initialize recommender
    recommender = MitigationRecommender()
    
    # Analyze conditions
    conditions = recommender.analyze_conditions(df)
    print("Analyzed Urban Conditions:")
    print(f"  - Green coverage: {conditions.get('green_coverage', 0):.2%}")
    print(f"  - Road density: {conditions.get('road_density', 0):.2%}")
    print(f"  - Building density: {conditions.get('building_density', 0):.1f}")
    print(f"  - Tree density: {conditions.get('tree_density', 0):.2%}")
    
    # Generate recommendations
    print("\nGenerating recommendations...")
    recommendations = recommender.recommend(df, max_recommendations=5)
    
    print(f"\nTop 5 Recommendations:")
    print("-" * 70)
    
    for i, strategy in enumerate(recommendations, 1):
        print(f"\n{i}. {strategy.name}")
        print(f"   Category: {strategy.category.value}")
        print(f"   Priority: {strategy.priority.name}")
        print(f"   Cost: ${strategy.cost_per_sqm:.0f}/m²")
        print(f"   Timeline: {strategy.timeline_months} months")
        print(f"   Cooling Impact: {strategy.cooling_impact_celsius}°C reduction")
    
    # Export to DataFrame
    rec_df = recommender.to_dataframe()
    rec_df.to_csv('output/recommendations.csv', index=False)
    print(f"\n✓ Recommendations saved to output/recommendations.csv")
    
    return recommendations


def demonstrate_prediction_model(df_day: pd.DataFrame, df_night: pd.DataFrame):
    """Demonstrate UHI prediction modeling."""
    print_header("4. UHI PREDICTION MODEL")
    
    # Initialize model
    print("Training Random Forest model on daytime data...")
    model = UHIPredictionModel(model_type='random_forest')
    
    # Train on daytime data
    metrics = model.train(df_day, target_column='UHI_d')
    
    print(f"\nModel Performance (Daytime UHI):")
    print(f"  - R² Score: {metrics.r2:.4f}")
    print(f"  - RMSE: {metrics.rmse:.6f}°C")
    print(f"  - MAE: {metrics.mae:.6f}°C")
    
    if metrics.cv_scores is not None:
        print(f"  - CV R² Mean: {np.mean(metrics.cv_scores):.4f} ± {np.std(metrics.cv_scores):.4f}")
    
    # Feature importance
    print("\nTop 5 Important Features:")
    for i, (feature, importance) in enumerate(list(model.feature_importance.items())[:5], 1):
        print(f"  {i}. {feature}: {importance:.4f}")
    
    # Train ensemble model
    print("\n\nTraining Ensemble Model...")
    ensemble = EnsembleUHIModel()
    ensemble_metrics = ensemble.train(df_day, target_column='UHI_d')
    
    print(f"\nEnsemble Model Performance:")
    print(f"  - Random Forest R²: {ensemble_metrics['random_forest'].r2:.4f}")
    print(f"  - Gradient Boosting R²: {ensemble_metrics['gradient_boosting'].r2:.4f}")
    
    # Save model
    model.save_model('output/uhi_model.pkl')
    print(f"\n✓ Model saved to output/uhi_model.pkl")
    
    # Scenario analysis
    print("\n\nScenario Analysis:")
    scenarios = {
        'increased_green': {'GnPR': 0.9, 'park_grass_ratio': 0.2},
        'cool_roofs': {'greenroof_ratio': 0.3},
        'reduced_asphalt': {'asphalt_ratio': 0.05, 'roadDensity': 0.2}
    }
    
    scenario_results = model.predict_with_scenarios(df_day.head(50), scenarios)
    print(scenario_results.to_string(index=False))
    
    return model


def demonstrate_advanced_ml(df_day: pd.DataFrame):
    """Demonstrate advanced ML models."""
    print_header("4B. ADVANCED ML MODELS")
    
    # Model comparison
    print("Comparing multiple ML models...")
    comparison = ModelComparison()
    results_df = comparison.compare_models(df_day, target_column='UHI_d')
    
    print("\nModel Comparison Results:")
    print("-" * 60)
    print(results_df.to_string(index=False))
    
    # Get best model
    best_name, best_model = comparison.get_best_model()
    print(f"\n✓ Best Model: {best_name}")
    
    # Train advanced model (XGBoost or fallback)
    print("\n\nTraining Advanced Model (XGBoost/Extra Trees)...")
    advanced = AdvancedUHIModel(model_type='xgboost')
    adv_metrics = advanced.train(df_day, target_column='UHI_d')
    
    print(f"\nAdvanced Model Performance ({advanced.model_type}):")
    print(f"  - R² Score: {adv_metrics.r2:.4f}")
    print(f"  - RMSE: {adv_metrics.rmse:.6f}°C")
    
    # Train stacking ensemble
    print("\n\nTraining Stacking Ensemble...")
    stacking = StackingUHIModel(use_advanced=True)
    stack_metrics = stacking.train(df_day, target_column='UHI_d')
    
    print(f"\nStacking Ensemble Performance:")
    print(f"  - R² Score: {stack_metrics.r2:.4f}")
    print(f"  - RMSE: {stack_metrics.rmse:.6f}°C")
    
    # Show base model comparison
    print("\nBase Model Scores:")
    model_comp = stacking.get_model_comparison()
    print(model_comp.to_string(index=False))
    
    # Save advanced model
    advanced.save('output/advanced_uhi_model.pkl')
    print(f"\n✓ Advanced model saved to output/advanced_uhi_model.pkl")
    
    return advanced, stacking


def demonstrate_urban_landscape(df: pd.DataFrame, hotspot_result):
    """Demonstrate enhanced urban landscape visualization with buildings."""
    print_header("5B. URBAN LANDSCAPE VISUALIZATION")
    
    # Get hotspot data
    hotspot_df = hotspot_result.hotspot_data if len(hotspot_result.hotspot_data) > 0 else df.head(100)
    
    print("Generating procedural urban landscape...")
    
    # Generate urban landscape
    generator = UrbanLandscapeGenerator(seed=42)
    landscape = generator.generate_from_dataframe(
        df,
        hotspot_df=hotspot_df,
        terrain_size=(200, 200)
    )
    
    print(f"\nUrban Landscape Generated:")
    print(f"  - Buildings: {len(landscape.buildings)}")
    print(f"  - Trees: {len(landscape.trees)}")
    print(f"  - Roads: {len(landscape.roads)}")
    print(f"  - Hotspot Zones: {len(landscape.hotspot_zones)}")
    
    # Building type distribution
    from collections import Counter
    building_types = Counter(b.building_type.value for b in landscape.buildings)
    print("\nBuilding Type Distribution:")
    for btype, count in building_types.most_common():
        print(f"  - {btype}: {count}")
    
    # Generate enhanced visualization
    print("\nGenerating enhanced AR/VR visualization...")
    viz_manager = VisualizationManager()
    
    output_paths = viz_manager.generate_urban_landscape(
        df=df,
        hotspot_df=hotspot_df,
        output_dir='output/urban_visualization',
        prefix='uhi_city',
        terrain_size=(200, 200)
    )
    
    print(f"\nGenerated Enhanced Visualization Files:")
    for format_name, path in output_paths.items():
        print(f"  ✓ {format_name}: {path}")
    
    print("\n🎮 Open the HTML file in a browser for immersive 3D visualization")
    print("🥽 VR mode available on compatible devices")
    print("🌙 Toggle day/night mode for different viewing experiences")
    
    return landscape, output_paths


def demonstrate_visualization(df: pd.DataFrame, hotspot_result):
    """Demonstrate AR/VR visualization generation."""
    print_header("5. AR/VR VISUALIZATION GENERATION")
    
    # Get hotspot data
    hotspot_df = hotspot_result.hotspot_data if len(hotspot_result.hotspot_data) > 0 else df.head(50)
    
    # Initialize visualization manager
    viz_manager = VisualizationManager()
    
    # Generate all outputs
    print("Generating visualization files...")
    
    output_paths = viz_manager.generate_all(
        hotspot_df,
        output_dir='output/visualizations',
        prefix='uhi_hotspots',
        uhi_column='UHI_d'
    )
    
    print(f"\nGenerated Files:")
    for format_name, path in output_paths.items():
        print(f"  ✓ {format_name}: {path}")
    
    return output_paths


def generate_analysis_plots(df: pd.DataFrame, grid_analyzer):
    """Generate analysis visualization plots."""
    print_header("6. GENERATING ANALYSIS PLOTS")
    
    # Create output directory
    os.makedirs('output/plots', exist_ok=True)
    
    # Plot 1: UHI Distribution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # UHI histogram
    axes[0, 0].hist(df['UHI_d'], bins=30, color='coral', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('UHI Intensity (°C)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('UHI Distribution')
    axes[0, 0].axvline(df['UHI_d'].mean(), color='red', linestyle='--', label=f'Mean: {df["UHI_d"].mean():.3f}')
    axes[0, 0].legend()
    
    # UHI vs Green Coverage
    if 'GnPR' in df.columns:
        axes[0, 1].scatter(df['GnPR'], df['UHI_d'], alpha=0.5, c='green', s=20)
        axes[0, 1].set_xlabel('Green Coverage Ratio')
        axes[0, 1].set_ylabel('UHI Intensity (°C)')
        axes[0, 1].set_title('UHI vs Green Coverage')
    
    # UHI vs Road Density
    if 'roadDensity' in df.columns:
        axes[1, 0].scatter(df['roadDensity'], df['UHI_d'], alpha=0.5, c='gray', s=20)
        axes[1, 0].set_xlabel('Road Density')
        axes[1, 0].set_ylabel('UHI Intensity (°C)')
        axes[1, 0].set_title('UHI vs Road Density')
    
    # Grid heatmap
    grid_matrix = grid_analyzer.get_grid_matrix('mean_uhi')
    im = axes[1, 1].imshow(grid_matrix, cmap='YlOrRd', aspect='auto')
    axes[1, 1].set_title('Grid-based UHI Heatmap')
    axes[1, 1].set_xlabel('Grid Column')
    axes[1, 1].set_ylabel('Grid Row')
    plt.colorbar(im, ax=axes[1, 1], label='Mean UHI (°C)')
    
    plt.tight_layout()
    plt.savefig('output/plots/uhi_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Analysis plots saved to output/plots/uhi_analysis.png")
    
    # Feature correlation plot
    feature_cols = ['asphalt_ratio', 'park_grass_ratio', 'GnPR', 'roadDensity', 
                   'bldDensity', 'treeDensity', 'avg_BH']
    available_cols = [c for c in feature_cols if c in df.columns]
    
    if available_cols:
        corr_data = df[available_cols + ['UHI_d']].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_data, cmap='RdBu_r', vmin=-1, vmax=1)
        
        ax.set_xticks(range(len(corr_data.columns)))
        ax.set_yticks(range(len(corr_data.columns)))
        ax.set_xticklabels(corr_data.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr_data.columns)
        
        for i in range(len(corr_data)):
            for j in range(len(corr_data)):
                ax.text(j, i, f'{corr_data.iloc[i, j]:.2f}', 
                       ha='center', va='center', fontsize=8)
        
        plt.colorbar(im, label='Correlation')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('output/plots/correlation_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Correlation matrix saved to output/plots/correlation_matrix.png")


def main():
    """Main execution function."""
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "URBAN HEAT ISLAND ANALYSIS SYSTEM v2.0" + " " * 13 + "║")
    print("║" + " " * 8 + "Advanced ML + Urban Landscape Visualization" + " " * 13 + "║")
    print("╚" + "═" * 68 + "╝")
    
    # Create output directories
    os.makedirs('output/visualizations', exist_ok=True)
    os.makedirs('output/plots', exist_ok=True)
    os.makedirs('output/urban_visualization', exist_ok=True)
    
    try:
        # 1. Load and preprocess data
        df_day, df_night = demonstrate_data_loading()
        
        # 2. Detect hotspots
        hotspot_result, grid_analyzer = demonstrate_hotspot_detection(df_day)
        
        # 3. Generate mitigation recommendations
        recommendations = demonstrate_mitigation_recommendations(df_day)
        
        # 4. Train prediction model (basic)
        model = demonstrate_prediction_model(df_day, df_night)
        
        # 4B. Train advanced ML models
        advanced_model, stacking_model = demonstrate_advanced_ml(df_day)
        
        # 5. Generate basic visualizations
        viz_paths = demonstrate_visualization(df_day, hotspot_result)
        
        # 5B. Generate enhanced urban landscape visualization
        landscape, urban_viz_paths = demonstrate_urban_landscape(df_day, hotspot_result)
        
        # 6. Generate analysis plots
        generate_analysis_plots(df_day, grid_analyzer)
        
        # Summary
        print_header("ANALYSIS COMPLETE")
        print("Generated outputs:")
        print("  • output/recommendations.csv - Mitigation strategies")
        print("  • output/uhi_model.pkl - Basic prediction model")
        print("  • output/advanced_uhi_model.pkl - Advanced ML model")
        print("  • output/visualizations/ - Basic AR/VR files")
        print("  • output/urban_visualization/ - Enhanced 3D city visualization")
        print("  • output/plots/ - Analysis visualizations")
        print("\n✓ All modules executed successfully!")
        print("\n🏙️  Open output/urban_visualization/uhi_city_visualization.html")
        print("   for immersive 3D urban heat island visualization!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
