"""
UHI Prediction Model - Advanced ML Suite
=========================================

Comprehensive machine learning module for predicting Urban Heat Island intensity
based on urban characteristics. Features:

- Multiple ML algorithms: XGBoost, LightGBM, CatBoost, Neural Networks
- Ensemble methods: Random Forest, Gradient Boosting, Stacking
- AutoML with hyperparameter optimization
- SHAP-based model explainability
- Feature importance analysis
- Scenario prediction and what-if analysis
"""

import logging
import pickle
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    RandomizedSearchCV, KFold
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    AdaBoostRegressor, ExtraTreesRegressor, StackingRegressor,
    VotingRegressor, BaggingRegressor
)
from sklearn.linear_model import (
    Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin

# Try importing advanced ML libraries
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBRegressor = None
    
try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    CatBoostRegressor = None

# SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

# Optuna for hyperparameter optimization
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class ModelMetrics:
    """Stores model evaluation metrics."""
    mse: float
    rmse: float
    mae: float
    r2: float
    cv_scores: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'mse': self.mse,
            'rmse': self.rmse,
            'mae': self.mae,
            'r2': self.r2,
            'cv_mean': float(np.mean(self.cv_scores)) if self.cv_scores is not None else None,
            'cv_std': float(np.std(self.cv_scores)) if self.cv_scores is not None else None
        }


class UHIPredictionModel:
    """
    Machine learning model for UHI prediction.
    
    Supports multiple regression algorithms and provides
    feature importance analysis for understanding UHI drivers.
    """
    
    # Default feature columns based on the dataset structure
    DEFAULT_FEATURES = [
        'asphalt_ratio', 'park_grass_ratio', 'parcel_grass_ratio',
        'podium_grass_ratio', 'GnPR', 'greenroof_ratio', 
        'parcel_fp_ratio', 'roadDensity', 'bldDensity',
        'treeDensity', 'avg_BH', 'avg_GPR', 'parkRadius'
    ]
    
    AVAILABLE_MODELS = {
        'random_forest': RandomForestRegressor,
        'gradient_boosting': GradientBoostingRegressor,
        'ridge': Ridge
    }
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the prediction model.
        
        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting', 'ridge')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.target_column: str = ''
        self.is_trained = False
        self.metrics: Optional[ModelMetrics] = None
        self.feature_importance: Optional[Dict[str, float]] = None
        
    def _get_model(self, **kwargs) -> Any:
        """Get the model instance based on type."""
        if self.model_type not in self.AVAILABLE_MODELS:
            logger.warning(f"Unknown model type: {self.model_type}, using random_forest")
            self.model_type = 'random_forest'
            
        model_class = self.AVAILABLE_MODELS[self.model_type]
        
        # Default hyperparameters
        defaults = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'min_samples_split': 5,
                'random_state': 42
            },
            'ridge': {
                'alpha': 1.0
            }
        }
        
        params = defaults.get(self.model_type, {})
        params.update(kwargs)
        
        return model_class(**params)
    
    def prepare_features(self, df: pd.DataFrame,
                        feature_columns: List[str] = None,
                        target_column: str = 'UHI_d') -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Prepare features and target for training/prediction.
        
        Args:
            df: Input DataFrame
            feature_columns: List of feature column names
            target_column: Name of target column
            
        Returns:
            Tuple of (features DataFrame, target Series or None)
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided")
            return pd.DataFrame(), None
            
        # Determine feature columns
        if feature_columns is None:
            feature_columns = [c for c in self.DEFAULT_FEATURES if c in df.columns]
            
        if not feature_columns:
            logger.warning("No valid feature columns found")
            # Use all numeric columns except target
            feature_columns = [c for c in df.select_dtypes(include=[np.number]).columns 
                             if c != target_column and 'UHI' not in c.upper()]
                             
        self.feature_columns = feature_columns
        self.target_column = target_column
        
        # Extract features
        X = df[feature_columns].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Extract target if available
        y = None
        if target_column in df.columns:
            y = df[target_column].copy()
            
        return X, y
    
    def train(self, df: pd.DataFrame,
              feature_columns: List[str] = None,
              target_column: str = 'UHI_d',
              test_size: float = 0.2,
              perform_cv: bool = True,
              **model_kwargs) -> ModelMetrics:
        """
        Train the prediction model.
        
        Args:
            df: Training DataFrame
            feature_columns: Feature columns to use
            target_column: Target column name
            test_size: Fraction for test split
            perform_cv: Whether to perform cross-validation
            **model_kwargs: Additional model parameters
            
        Returns:
            ModelMetrics with evaluation results
        """
        try:
            # Prepare data
            X, y = self.prepare_features(df, feature_columns, target_column)
            
            if X.empty or y is None:
                logger.error("Failed to prepare training data")
                return self._get_default_metrics()
                
            logger.info(f"Training with {len(X)} samples and {len(self.feature_columns)} features")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Initialize and train model
            self.model = self._get_model(**model_kwargs)
            self.model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = None
            if perform_cv:
                X_scaled = self.scaler.transform(X)
                cv_scores = cross_val_score(
                    self._get_model(**model_kwargs), 
                    X_scaled, y, 
                    cv=5, 
                    scoring='r2'
                )
                
            self.metrics = ModelMetrics(
                mse=mse,
                rmse=rmse,
                mae=mae,
                r2=r2,
                cv_scores=cv_scores
            )
            
            # Calculate feature importance
            self._calculate_feature_importance()
            
            self.is_trained = True
            logger.info(f"Training complete. R² = {r2:.4f}, RMSE = {rmse:.6f}")
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return self._get_default_metrics()
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions for new data.
        
        Args:
            df: DataFrame with feature values
            
        Returns:
            Array of predicted UHI values
        """
        if not self.is_trained:
            logger.warning("Model not trained, returning default predictions")
            return np.full(len(df), 0.15)
            
        try:
            X, _ = self.prepare_features(df, self.feature_columns, self.target_column)
            
            if X.empty:
                return np.full(len(df), 0.15)
                
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return np.full(len(df), 0.15)
    
    def _calculate_feature_importance(self):
        """Calculate and store feature importance."""
        if self.model is None or not self.feature_columns:
            self.feature_importance = {}
            return
            
        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                importances = np.abs(self.model.coef_)
            else:
                importances = np.ones(len(self.feature_columns))
                
            self.feature_importance = dict(zip(self.feature_columns, importances))
            
            # Sort by importance
            self.feature_importance = dict(
                sorted(self.feature_importance.items(), 
                       key=lambda x: x[1], 
                       reverse=True)
            )
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            self.feature_importance = {}
    
    def _get_default_metrics(self) -> ModelMetrics:
        """Return default metrics when training fails."""
        return ModelMetrics(mse=0, rmse=0, mae=0, r2=0)
    
    def get_feature_importance_report(self) -> str:
        """Generate a text report of feature importance."""
        if not self.feature_importance:
            return "No feature importance data available."
            
        lines = [
            "=" * 50,
            "FEATURE IMPORTANCE REPORT",
            "=" * 50,
            ""
        ]
        
        total = sum(self.feature_importance.values())
        
        for i, (feature, importance) in enumerate(self.feature_importance.items(), 1):
            pct = (importance / total * 100) if total > 0 else 0
            bar = '█' * int(pct / 2) + '░' * (50 - int(pct / 2))
            lines.append(f"{i:2}. {feature:25} {pct:5.1f}% {bar[:30]}")
            
        return "\n".join(lines)
    
    def save_model(self, filepath: str) -> bool:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
            
        Returns:
            True if successful
        """
        if not self.is_trained:
            logger.warning("Cannot save untrained model")
            return False
            
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'model_type': self.model_type,
                'metrics': self.metrics,
                'feature_importance': self.feature_importance
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
                
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            True if successful
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
                
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.target_column = model_data['target_column']
            self.model_type = model_data['model_type']
            self.metrics = model_data.get('metrics')
            self.feature_importance = model_data.get('feature_importance')
            self.is_trained = True
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict_with_scenarios(self, base_df: pd.DataFrame,
                               scenarios: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Predict UHI for different intervention scenarios.
        
        Args:
            base_df: Base urban configuration
            scenarios: Dict of scenario_name -> {feature: new_value}
            
        Returns:
            DataFrame with predictions for each scenario
        """
        results = []
        
        # Base prediction
        base_pred = self.predict(base_df)
        results.append({
            'scenario': 'baseline',
            'mean_uhi': np.mean(base_pred),
            'max_uhi': np.max(base_pred),
            'reduction': 0
        })
        
        # Scenario predictions
        for scenario_name, changes in scenarios.items():
            scenario_df = base_df.copy()
            
            for feature, new_value in changes.items():
                if feature in scenario_df.columns:
                    scenario_df[feature] = new_value
                    
            pred = self.predict(scenario_df)
            reduction = np.mean(base_pred) - np.mean(pred)
            
            results.append({
                'scenario': scenario_name,
                'mean_uhi': np.mean(pred),
                'max_uhi': np.max(pred),
                'reduction': reduction
            })
            
        return pd.DataFrame(results)


class EnsembleUHIModel:
    """
    Ensemble model combining multiple prediction approaches.
    
    Uses weighted averaging of Random Forest and Gradient Boosting
    for more robust predictions.
    """
    
    def __init__(self, weights: Tuple[float, float] = (0.6, 0.4)):
        """
        Initialize ensemble model.
        
        Args:
            weights: Tuple of (rf_weight, gb_weight)
        """
        self.rf_model = UHIPredictionModel('random_forest')
        self.gb_model = UHIPredictionModel('gradient_boosting')
        self.weights = weights
        self.is_trained = False
        
    def train(self, df: pd.DataFrame, **kwargs) -> Dict[str, ModelMetrics]:
        """Train both models."""
        rf_metrics = self.rf_model.train(df, **kwargs)
        gb_metrics = self.gb_model.train(df, **kwargs)
        
        self.is_trained = True
        
        return {
            'random_forest': rf_metrics,
            'gradient_boosting': gb_metrics
        }
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make weighted ensemble predictions."""
        if not self.is_trained:
            return np.full(len(df), 0.15)
            
        rf_pred = self.rf_model.predict(df)
        gb_pred = self.gb_model.predict(df)
        
        ensemble_pred = (
            self.weights[0] * rf_pred + 
            self.weights[1] * gb_pred
        )
        
        return ensemble_pred
    
    def get_combined_feature_importance(self) -> Dict[str, float]:
        """Get weighted feature importance from both models."""
        rf_imp = self.rf_model.feature_importance or {}
        gb_imp = self.gb_model.feature_importance or {}
        
        all_features = set(rf_imp.keys()) | set(gb_imp.keys())
        
        combined = {}
        for feature in all_features:
            combined[feature] = (
                self.weights[0] * rf_imp.get(feature, 0) +
                self.weights[1] * gb_imp.get(feature, 0)
            )
            
        return dict(sorted(combined.items(), key=lambda x: x[1], reverse=True))


# ============================================================================
# ADVANCED ML MODELS
# ============================================================================

class AdvancedUHIModel:
    """
    Advanced UHI prediction model with state-of-the-art ML algorithms.
    
    Supports:
    - XGBoost (gradient boosting with regularization)
    - LightGBM (fast gradient boosting)
    - CatBoost (handles categorical features)
    - Neural Networks (MLP with configurable architecture)
    - Extra Trees (extremely randomized trees)
    - SVR (Support Vector Regression)
    - Bayesian Ridge Regression
    """
    
    ADVANCED_MODELS = {
        'xgboost': {
            'available': XGBOOST_AVAILABLE,
            'class': XGBRegressor,
            'params': {
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0
            }
        },
        'lightgbm': {
            'available': LIGHTGBM_AVAILABLE,
            'class': LGBMRegressor,
            'params': {
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
        },
        'catboost': {
            'available': CATBOOST_AVAILABLE,
            'class': CatBoostRegressor,
            'params': {
                'iterations': 200,
                'depth': 8,
                'learning_rate': 0.05,
                'l2_leaf_reg': 3,
                'random_state': 42,
                'verbose': False
            }
        },
        'neural_network': {
            'available': True,
            'class': MLPRegressor,
            'params': {
                'hidden_layer_sizes': (128, 64, 32),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.001,
                'batch_size': 32,
                'learning_rate': 'adaptive',
                'learning_rate_init': 0.001,
                'max_iter': 500,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'n_iter_no_change': 20,
                'random_state': 42
            }
        },
        'extra_trees': {
            'available': True,
            'class': ExtraTreesRegressor,
            'params': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            }
        },
        'svr': {
            'available': True,
            'class': SVR,
            'params': {
                'kernel': 'rbf',
                'C': 10,
                'epsilon': 0.01,
                'gamma': 'scale'
            }
        },
        'bayesian_ridge': {
            'available': True,
            'class': BayesianRidge,
            'params': {
                'alpha_1': 1e-6,
                'alpha_2': 1e-6,
                'lambda_1': 1e-6,
                'lambda_2': 1e-6,
                'compute_score': True
            }
        },
        'huber': {
            'available': True,
            'class': HuberRegressor,
            'params': {
                'epsilon': 1.35,
                'alpha': 0.001,
                'max_iter': 200
            }
        }
    }
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        Initialize advanced model.
        
        Args:
            model_type: Type of model to use
        """
        self.model_type = model_type
        self.model = None
        self.scaler = RobustScaler()  # More robust to outliers
        self.feature_columns: List[str] = []
        self.target_column: str = ''
        self.is_trained = False
        self.metrics: Optional[ModelMetrics] = None
        self.feature_importance: Dict[str, float] = {}
        self.training_history: List[Dict] = []
        
        # Validate model type
        if model_type not in self.ADVANCED_MODELS:
            logger.warning(f"Unknown model type: {model_type}, defaulting to random_forest")
            self.model_type = 'extra_trees'
        elif not self.ADVANCED_MODELS[model_type]['available']:
            logger.warning(f"{model_type} not available, defaulting to extra_trees")
            self.model_type = 'extra_trees'
    
    def _create_model(self, **override_params) -> Any:
        """Create model instance with parameters."""
        config = self.ADVANCED_MODELS[self.model_type]
        params = config['params'].copy()
        params.update(override_params)
        
        return config['class'](**params)
    
    def train(self, df: pd.DataFrame,
              feature_columns: List[str] = None,
              target_column: str = 'UHI_d',
              test_size: float = 0.2,
              perform_cv: bool = True,
              cv_folds: int = 5,
              **model_params) -> ModelMetrics:
        """
        Train the advanced model.
        
        Args:
            df: Training DataFrame
            feature_columns: Feature columns to use
            target_column: Target column name
            test_size: Fraction for test split
            perform_cv: Whether to perform cross-validation
            cv_folds: Number of CV folds
            **model_params: Override model parameters
            
        Returns:
            ModelMetrics with evaluation results
        """
        try:
            # Prepare features
            X, y = self._prepare_features(df, feature_columns, target_column)
            
            if X.empty or y is None:
                logger.error("Failed to prepare training data")
                return ModelMetrics(mse=0, rmse=0, mae=0, r2=0)
            
            logger.info(f"Training {self.model_type} with {len(X)} samples")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Create and train model
            self.model = self._create_model(**model_params)
            self.model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = None
            if perform_cv:
                kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                X_scaled = self.scaler.transform(X)
                cv_scores = cross_val_score(
                    self._create_model(**model_params),
                    X_scaled, y,
                    cv=kf,
                    scoring='r2'
                )
            
            self.metrics = ModelMetrics(
                mse=mse, rmse=rmse, mae=mae, r2=r2, cv_scores=cv_scores
            )
            
            # Calculate feature importance
            self._calculate_feature_importance()
            
            self.is_trained = True
            self.training_history.append({
                'model_type': self.model_type,
                'r2': r2,
                'rmse': rmse,
                'n_samples': len(X)
            })
            
            logger.info(f"Training complete. R² = {r2:.4f}, RMSE = {rmse:.6f}")
            return self.metrics
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return ModelMetrics(mse=0, rmse=0, mae=0, r2=0)
    
    def _prepare_features(self, df: pd.DataFrame,
                          feature_columns: List[str] = None,
                          target_column: str = 'UHI_d') -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Prepare features for training/prediction."""
        if df is None or df.empty:
            return pd.DataFrame(), None
        
        # Default feature columns
        default_features = [
            'asphalt_ratio', 'park_grass_ratio', 'parcel_grass_ratio',
            'podium_grass_ratio', 'GnPR', 'greenroof_ratio',
            'parcel_fp_ratio', 'roadDensity', 'bldDensity',
            'treeDensity', 'avg_BH', 'avg_GPR', 'parkRadius'
        ]
        
        if feature_columns is None:
            feature_columns = [c for c in default_features if c in df.columns]
        
        if not feature_columns:
            feature_columns = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c != target_column and 'UHI' not in c.upper()
            ]
        
        self.feature_columns = feature_columns
        self.target_column = target_column
        
        X = df[feature_columns].copy().fillna(df[feature_columns].median())
        y = df[target_column].copy() if target_column in df.columns else None
        
        return X, y
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            logger.warning("Model not trained")
            return np.full(len(df), 0.15)
        
        try:
            X, _ = self._prepare_features(df, self.feature_columns, self.target_column)
            if X.empty:
                return np.full(len(df), 0.15)
            
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return np.full(len(df), 0.15)
    
    def _calculate_feature_importance(self):
        """Calculate feature importance."""
        if self.model is None:
            return
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                importances = np.abs(self.model.coef_)
            else:
                importances = np.ones(len(self.feature_columns)) / len(self.feature_columns)
            
            self.feature_importance = dict(
                sorted(
                    zip(self.feature_columns, importances),
                    key=lambda x: x[1],
                    reverse=True
                )
            )
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
    
    def save(self, filepath: str) -> bool:
        """Save model to disk."""
        try:
            data = {
                'model': self.model,
                'scaler': self.scaler,
                'model_type': self.model_type,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'metrics': self.metrics,
                'feature_importance': self.feature_importance
            }
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load(self, filepath: str) -> bool:
        """Load model from disk."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.model_type = data['model_type']
            self.feature_columns = data['feature_columns']
            self.target_column = data['target_column']
            self.metrics = data.get('metrics')
            self.feature_importance = data.get('feature_importance', {})
            self.is_trained = True
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False


class StackingUHIModel:
    """
    Stacking ensemble that combines multiple base models with a meta-learner.
    
    Base models: Random Forest, Gradient Boosting, Extra Trees, XGBoost
    Meta-learner: Ridge Regression (to avoid overfitting)
    """
    
    def __init__(self, use_advanced: bool = True):
        """
        Initialize stacking model.
        
        Args:
            use_advanced: Whether to include XGBoost/LightGBM if available
        """
        self.use_advanced = use_advanced
        self.model = None
        self.scaler = RobustScaler()
        self.feature_columns: List[str] = []
        self.is_trained = False
        self.metrics: Optional[ModelMetrics] = None
        self.base_model_scores: Dict[str, float] = {}
        
    def _create_base_models(self) -> List[Tuple[str, Any]]:
        """Create list of base models."""
        base_models = [
            ('rf', RandomForestRegressor(
                n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
            )),
            ('gb', GradientBoostingRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
            )),
            ('et', ExtraTreesRegressor(
                n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
            )),
        ]
        
        if self.use_advanced:
            if XGBOOST_AVAILABLE:
                base_models.append(('xgb', XGBRegressor(
                    n_estimators=100, max_depth=8, learning_rate=0.1,
                    random_state=42, n_jobs=-1, verbosity=0
                )))
            if LIGHTGBM_AVAILABLE:
                base_models.append(('lgbm', LGBMRegressor(
                    n_estimators=100, max_depth=8, learning_rate=0.1,
                    random_state=42, n_jobs=-1, verbose=-1
                )))
        
        return base_models
    
    def train(self, df: pd.DataFrame,
              feature_columns: List[str] = None,
              target_column: str = 'UHI_d',
              test_size: float = 0.2) -> ModelMetrics:
        """
        Train stacking ensemble.
        """
        try:
            # Prepare data
            X, y = self._prepare_features(df, feature_columns, target_column)
            if X.empty or y is None:
                return ModelMetrics(mse=0, rmse=0, mae=0, r2=0)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Create stacking model
            base_models = self._create_base_models()
            self.model = StackingRegressor(
                estimators=base_models,
                final_estimator=Ridge(alpha=1.0),
                cv=5,
                n_jobs=-1
            )
            
            logger.info(f"Training Stacking Ensemble with {len(base_models)} base models")
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            
            mse = mean_squared_error(y_test, y_pred)
            self.metrics = ModelMetrics(
                mse=mse,
                rmse=np.sqrt(mse),
                mae=mean_absolute_error(y_test, y_pred),
                r2=r2_score(y_test, y_pred)
            )
            
            # Get individual model scores
            for name, model in base_models:
                model.fit(X_train_scaled, y_train)
                score = r2_score(y_test, model.predict(X_test_scaled))
                self.base_model_scores[name] = score
            
            self.is_trained = True
            logger.info(f"Stacking R² = {self.metrics.r2:.4f}")
            return self.metrics
            
        except Exception as e:
            logger.error(f"Stacking training error: {e}")
            return ModelMetrics(mse=0, rmse=0, mae=0, r2=0)
    
    def _prepare_features(self, df, feature_columns, target_column):
        """Prepare features."""
        default_features = [
            'asphalt_ratio', 'park_grass_ratio', 'parcel_grass_ratio',
            'podium_grass_ratio', 'GnPR', 'greenroof_ratio',
            'roadDensity', 'bldDensity', 'treeDensity', 'avg_BH'
        ]
        
        if feature_columns is None:
            feature_columns = [c for c in default_features if c in df.columns]
        
        if not feature_columns:
            feature_columns = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c != target_column and 'UHI' not in c.upper()
            ]
        
        self.feature_columns = feature_columns
        X = df[feature_columns].copy().fillna(df[feature_columns].median())
        y = df[target_column] if target_column in df.columns else None
        
        return X, y
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            return np.full(len(df), 0.15)
        
        X, _ = self._prepare_features(df, self.feature_columns, 'UHI_d')
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Get comparison of base model scores."""
        if not self.base_model_scores:
            return pd.DataFrame()
        
        data = [{'model': k, 'r2_score': v} for k, v in self.base_model_scores.items()]
        data.append({'model': 'stacking_ensemble', 'r2_score': self.metrics.r2})
        
        return pd.DataFrame(data).sort_values('r2_score', ascending=False)


class ModelExplainer:
    """
    SHAP-based model explainability for UHI predictions.
    
    Provides:
    - Feature importance visualization
    - SHAP summary plots
    - Partial dependence analysis
    - Feature interaction analysis
    """
    
    def __init__(self, model: Any, X_train: np.ndarray, feature_names: List[str]):
        """
        Initialize explainer.
        
        Args:
            model: Trained sklearn-compatible model
            X_train: Training data for background
            feature_names: List of feature names
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Install with: pip install shap")
    
    def compute_shap_values(self, X_explain: np.ndarray = None,
                            max_samples: int = 100) -> Optional[np.ndarray]:
        """
        Compute SHAP values for model explanations.
        
        Args:
            X_explain: Data to explain (defaults to sample of X_train)
            max_samples: Max samples for background
            
        Returns:
            SHAP values array
        """
        if not SHAP_AVAILABLE:
            return None
        
        try:
            # Sample background data
            if len(self.X_train) > max_samples:
                bg_indices = np.random.choice(len(self.X_train), max_samples, replace=False)
                background = self.X_train[bg_indices]
            else:
                background = self.X_train
            
            # Create explainer based on model type
            if hasattr(self.model, 'feature_importances_'):
                self.explainer = shap.TreeExplainer(self.model)
            else:
                self.explainer = shap.KernelExplainer(
                    self.model.predict, background
                )
            
            # Compute SHAP values
            X_to_explain = X_explain if X_explain is not None else background
            self.shap_values = self.explainer.shap_values(X_to_explain)
            
            logger.info(f"Computed SHAP values for {len(X_to_explain)} samples")
            return self.shap_values
            
        except Exception as e:
            logger.error(f"Error computing SHAP values: {e}")
            return None
    
    def get_feature_importance_shap(self) -> Dict[str, float]:
        """
        Get SHAP-based feature importance.
        
        Returns:
            Dictionary of feature -> mean |SHAP| importance
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        if self.shap_values is None:
            return {}
        
        # Mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(self.shap_values), axis=0)
        
        importance = dict(zip(self.feature_names, mean_abs_shap))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    def get_feature_interactions(self, top_n: int = 5) -> pd.DataFrame:
        """
        Analyze feature interactions.
        
        Returns:
            DataFrame with top feature interactions
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        if self.shap_values is None:
            return pd.DataFrame()
        
        # Compute interaction matrix
        n_features = len(self.feature_names)
        interactions = np.zeros((n_features, n_features))
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                # Correlation of SHAP values as proxy for interaction
                corr = np.corrcoef(self.shap_values[:, i], self.shap_values[:, j])[0, 1]
                interactions[i, j] = abs(corr)
                interactions[j, i] = abs(corr)
        
        # Get top interactions
        results = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                results.append({
                    'feature_1': self.feature_names[i],
                    'feature_2': self.feature_names[j],
                    'interaction_strength': interactions[i, j]
                })
        
        return pd.DataFrame(results).sort_values(
            'interaction_strength', ascending=False
        ).head(top_n)
    
    def explain_single_prediction(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Explain a single prediction.
        
        Args:
            x: Single sample to explain (1D array)
            
        Returns:
            Dictionary with prediction explanation
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            return {'error': 'SHAP not available or explainer not initialized'}
        
        try:
            x = x.reshape(1, -1)
            shap_vals = self.explainer.shap_values(x)[0]
            prediction = self.model.predict(x)[0]
            
            # Get contributions
            contributions = dict(zip(self.feature_names, shap_vals))
            contributions = dict(sorted(
                contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            ))
            
            return {
                'prediction': prediction,
                'base_value': self.explainer.expected_value if hasattr(
                    self.explainer, 'expected_value'
                ) else None,
                'feature_contributions': contributions,
                'top_positive_contributors': {
                    k: v for k, v in contributions.items() if v > 0
                },
                'top_negative_contributors': {
                    k: v for k, v in contributions.items() if v < 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error explaining prediction: {e}")
            return {'error': str(e)}


class AutoMLUHI:
    """
    AutoML for UHI prediction with automatic model selection
    and hyperparameter optimization.
    
    Uses Optuna for Bayesian optimization of hyperparameters.
    """
    
    def __init__(self, n_trials: int = 50, timeout: int = 300):
        """
        Initialize AutoML.
        
        Args:
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
        """
        self.n_trials = n_trials
        self.timeout = timeout
        self.best_model = None
        self.best_params = None
        self.best_score = None
        self.study = None
        self.scaler = RobustScaler()
        self.feature_columns: List[str] = []
        self.is_trained = False
        
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available. Using GridSearchCV instead.")
    
    def _objective(self, trial, X, y, cv_folds=5):
        """Optuna objective function."""
        # Select model type
        model_type = trial.suggest_categorical(
            'model_type', 
            ['random_forest', 'gradient_boosting', 'extra_trees', 'xgboost', 'lightgbm']
        )
        
        if model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=trial.suggest_int('n_estimators', 50, 300),
                max_depth=trial.suggest_int('max_depth', 5, 20),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 5),
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(
                n_estimators=trial.suggest_int('n_estimators', 50, 300),
                max_depth=trial.suggest_int('max_depth', 3, 10),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                random_state=42
            )
        elif model_type == 'extra_trees':
            model = ExtraTreesRegressor(
                n_estimators=trial.suggest_int('n_estimators', 50, 300),
                max_depth=trial.suggest_int('max_depth', 5, 20),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
            model = XGBRegressor(
                n_estimators=trial.suggest_int('n_estimators', 50, 300),
                max_depth=trial.suggest_int('max_depth', 3, 12),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                subsample=trial.suggest_float('subsample', 0.6, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
        elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            model = LGBMRegressor(
                n_estimators=trial.suggest_int('n_estimators', 50, 300),
                max_depth=trial.suggest_int('max_depth', 3, 12),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                num_leaves=trial.suggest_int('num_leaves', 20, 100),
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        else:
            # Fallback to Random Forest
            model = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
        
        # Cross-validation
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
        
        return scores.mean()
    
    def train(self, df: pd.DataFrame,
              feature_columns: List[str] = None,
              target_column: str = 'UHI_d') -> Dict[str, Any]:
        """
        Run AutoML optimization and train best model.
        
        Returns:
            Dictionary with best model info and metrics
        """
        # Prepare data
        X, y = self._prepare_features(df, feature_columns, target_column)
        if X.empty or y is None:
            return {'error': 'Failed to prepare data'}
        
        X_scaled = self.scaler.fit_transform(X)
        
        if OPTUNA_AVAILABLE:
            # Optuna optimization
            logger.info(f"Starting AutoML with {self.n_trials} trials...")
            
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            self.study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42)
            )
            
            self.study.optimize(
                lambda trial: self._objective(trial, X_scaled, y),
                n_trials=self.n_trials,
                timeout=self.timeout,
                show_progress_bar=False
            )
            
            self.best_params = self.study.best_params
            self.best_score = self.study.best_value
            
            # Train best model
            model_type = self.best_params.get('model_type', 'random_forest')
            logger.info(f"Best model: {model_type} with R² = {self.best_score:.4f}")
            
        else:
            # Fallback to GridSearchCV
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            }
            
            grid_search = GridSearchCV(
                RandomForestRegressor(random_state=42),
                param_grid,
                cv=5,
                scoring='r2',
                n_jobs=-1
            )
            grid_search.fit(X_scaled, y)
            
            self.best_params = grid_search.best_params_
            self.best_score = grid_search.best_score_
            self.best_model = grid_search.best_estimator_
        
        # Train final model with best params
        if self.best_model is None:
            self.best_model = self._create_best_model(self.best_params)
            self.best_model.fit(X_scaled, y)
        
        self.is_trained = True
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': self.n_trials if OPTUNA_AVAILABLE else 'grid_search',
            'model_type': self.best_params.get('model_type', 'random_forest')
        }
    
    def _create_best_model(self, params: Dict) -> Any:
        """Create model from best parameters."""
        model_type = params.get('model_type', 'random_forest')
        
        if model_type == 'xgboost' and XGBOOST_AVAILABLE:
            return XGBRegressor(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 8),
                learning_rate=params.get('learning_rate', 0.1),
                random_state=42, n_jobs=-1, verbosity=0
            )
        elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            return LGBMRegressor(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 8),
                learning_rate=params.get('learning_rate', 0.1),
                random_state=42, n_jobs=-1, verbose=-1
            )
        elif model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 6),
                learning_rate=params.get('learning_rate', 0.1),
                random_state=42
            )
        else:
            return RandomForestRegressor(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 10),
                random_state=42, n_jobs=-1
            )
    
    def _prepare_features(self, df, feature_columns, target_column):
        """Prepare features."""
        default_features = [
            'asphalt_ratio', 'park_grass_ratio', 'GnPR', 'greenroof_ratio',
            'roadDensity', 'bldDensity', 'treeDensity', 'avg_BH'
        ]
        
        if feature_columns is None:
            feature_columns = [c for c in default_features if c in df.columns]
        
        if not feature_columns:
            feature_columns = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c != target_column and 'UHI' not in c.upper()
            ]
        
        self.feature_columns = feature_columns
        X = df[feature_columns].copy().fillna(df[feature_columns].median())
        y = df[target_column] if target_column in df.columns else None
        
        return X, y
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions with best model."""
        if not self.is_trained:
            return np.full(len(df), 0.15)
        
        X, _ = self._prepare_features(df, self.feature_columns, 'UHI_d')
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict(X_scaled)


class ModelComparison:
    """
    Compare multiple ML models on UHI prediction task.
    """
    
    def __init__(self):
        self.results: List[Dict] = []
        self.trained_models: Dict[str, Any] = {}
        
    def compare_models(self, df: pd.DataFrame,
                       target_column: str = 'UHI_d',
                       test_size: float = 0.2) -> pd.DataFrame:
        """
        Compare all available models.
        
        Returns:
            DataFrame with model comparison results
        """
        models_to_test = [
            ('Random Forest', RandomForestRegressor(
                n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
            )),
            ('Gradient Boosting', GradientBoostingRegressor(
                n_estimators=100, max_depth=6, random_state=42
            )),
            ('Extra Trees', ExtraTreesRegressor(
                n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
            )),
            ('Ridge', Ridge(alpha=1.0)),
            ('Bayesian Ridge', BayesianRidge()),
            ('SVR', SVR(kernel='rbf', C=10)),
            ('KNN', KNeighborsRegressor(n_neighbors=5)),
            ('Neural Network', MLPRegressor(
                hidden_layer_sizes=(64, 32), max_iter=500, random_state=42
            )),
        ]
        
        if XGBOOST_AVAILABLE:
            models_to_test.append(('XGBoost', XGBRegressor(
                n_estimators=100, max_depth=8, random_state=42, verbosity=0
            )))
        
        if LIGHTGBM_AVAILABLE:
            models_to_test.append(('LightGBM', LGBMRegressor(
                n_estimators=100, max_depth=8, random_state=42, verbose=-1
            )))
        
        if CATBOOST_AVAILABLE:
            models_to_test.append(('CatBoost', CatBoostRegressor(
                iterations=100, depth=8, random_state=42, verbose=False
            )))
        
        # Prepare data
        feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != target_column and 'UHI' not in c.upper()
        ]
        
        X = df[feature_cols].fillna(df[feature_cols].median())
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.results = []
        
        for name, model in models_to_test:
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                result = {
                    'Model': name,
                    'R²': r2_score(y_test, y_pred),
                    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'MAE': mean_absolute_error(y_test, y_pred)
                }
                
                self.results.append(result)
                self.trained_models[name] = model
                
            except Exception as e:
                logger.warning(f"Error training {name}: {e}")
        
        return pd.DataFrame(self.results).sort_values('R²', ascending=False)
    
    def get_best_model(self) -> Tuple[str, Any]:
        """Get the best performing model."""
        if not self.results:
            return None, None
        
        best = max(self.results, key=lambda x: x['R²'])
        return best['Model'], self.trained_models.get(best['Model'])
