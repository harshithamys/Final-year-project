"""
Urban Heat Island (UHI) Analysis System v2.0
=============================================

A comprehensive Python system for analyzing urban heat islands,
detecting hotspots, recommending mitigation strategies, and
generating immersive AR/VR visualization outputs with procedural
urban landscape generation.

Modules:
    - core: Data loading, preprocessing, and hotspot detection
    - mitigation: Rule-based mitigation strategy recommender
    - visualization: AR/VR output generators (Three.js, Unity, Blender)
    - prediction: Advanced UHI prediction with XGBoost, LightGBM, Stacking
    - urban_generator: Procedural building and city generation

New in v2.0:
    - Advanced ML models (XGBoost, LightGBM, CatBoost, Neural Networks)
    - Stacking ensemble and AutoML
    - SHAP-based model explainability
    - Procedural urban landscape generation
    - Enhanced 3D visualization with buildings, trees, roads
    - WebXR VR support
    - Day/night mode toggle
"""

from .core import UHIDataLoader, HotspotDetector, GridBasedAnalyzer
from .mitigation import MitigationRecommender, Strategy
from .visualization import (
    ThreeJSGenerator, UnityExporter, BlenderScriptGenerator, 
    VisualizationManager
)
from .prediction import (
    UHIPredictionModel, EnsembleUHIModel,
    AdvancedUHIModel, StackingUHIModel, ModelComparison, AutoMLUHI
)
from .urban_generator import (
    UrbanLandscapeGenerator, UrbanLandscape, 
    Building, Tree, Road, HotspotZone
)

__version__ = "2.0.0"
__author__ = "UHI Analysis Team"

__all__ = [
    # Core
    "UHIDataLoader",
    "HotspotDetector", 
    "GridBasedAnalyzer",
    # Mitigation
    "MitigationRecommender",
    "Strategy",
    # Visualization
    "ThreeJSGenerator",
    "UnityExporter",
    "BlenderScriptGenerator",
    "VisualizationManager",
    # Prediction
    "UHIPredictionModel",
    "EnsembleUHIModel",
    "AdvancedUHIModel",
    "StackingUHIModel",
    "ModelComparison",
    "AutoMLUHI",
    # Urban Generator
    "UrbanLandscapeGenerator",
    "UrbanLandscape",
    "Building",
    "Tree",
    "Road",
    "HotspotZone",
]
