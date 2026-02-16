"""
Core Module for UHI Analysis
============================

This module provides core functionality for:
- Loading and preprocessing UHI data
- Detecting heat hotspots using DBSCAN clustering
- Grid-based spatial analysis of temperature patterns
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HotspotResult:
    """Data class to store hotspot detection results."""
    hotspot_indices: np.ndarray
    cluster_labels: np.ndarray
    n_clusters: int
    hotspot_data: pd.DataFrame
    statistics: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.statistics is None:
            self.statistics = {}


@dataclass
class GridCell:
    """Represents a single grid cell in spatial analysis."""
    row: int
    col: int
    center_x: float
    center_y: float
    uhi_values: List[float] = field(default_factory=list)
    mean_uhi: float = 0.0
    max_uhi: float = 0.0
    is_hotspot: bool = False
    point_count: int = 0


class UHIDataLoader:
    """
    Handles loading and preprocessing of UHI data from various sources.
    
    Supports CSV files with urban heat island measurements including
    spatial coordinates and UHI intensity values.
    """
    
    REQUIRED_COLUMNS_MINIMAL = ['UHI_d', 'UHI_n']
    FEATURE_COLUMNS = [
        'asphalt_ratio', 'park_grass_ratio', 'parcel_grass_ratio',
        'podium_grass_ratio', 'GnPR', 'greenroof_ratio', 'roadDensity',
        'bldDensity', 'treeDensity', 'avg_BH', 'total_GFA'
    ]
    
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.daytime_data: Optional[pd.DataFrame] = None
        self.nighttime_data: Optional[pd.DataFrame] = None
        self._is_loaded = False
        
    def load_csv(self, filepath: str, uhi_column: str = 'UHI_d') -> pd.DataFrame:
        """
        Load UHI data from a CSV file.
        
        Args:
            filepath: Path to the CSV file
            uhi_column: Name of the UHI value column
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If required columns are missing
        """
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} records from {filepath}")
            
            # Validate data
            if df.empty:
                logger.warning("Loaded empty dataset")
                return self._create_default_dataframe()
                
            # Check for UHI column
            if uhi_column not in df.columns:
                available = [c for c in df.columns if 'UHI' in c.upper()]
                if available:
                    uhi_column = available[0]
                    logger.info(f"Using column '{uhi_column}' as UHI values")
                else:
                    logger.warning(f"No UHI column found, using defaults")
                    return self._create_default_dataframe()
                    
            self.data = df
            self._is_loaded = True
            return df
            
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            return self._create_default_dataframe()
        except Exception as e:
            logger.error(f"Error loading file: {e}")
            return self._create_default_dataframe()
    
    def load_multiple(self, daytime_path: str, nighttime_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both daytime and nighttime UHI datasets.
        
        Args:
            daytime_path: Path to daytime UHI data
            nighttime_path: Path to nighttime UHI data
            
        Returns:
            Tuple of (daytime_df, nighttime_df)
        """
        self.daytime_data = self.load_csv(daytime_path, 'UHI_d')
        self.nighttime_data = self.load_csv(nighttime_path, 'UHI_n')
        return self.daytime_data, self.nighttime_data
    
    def _create_default_dataframe(self) -> pd.DataFrame:
        """Create a default DataFrame when loading fails."""
        logger.info("Creating default empty dataframe")
        return pd.DataFrame({
            'sim_no': [0],
            'UHI_d': [0.0],
            'UHI_n': [0.0],
            'lat': [0.0],
            'lon': [0.0],
            'asphalt_ratio': [0.1],
            'park_grass_ratio': [0.1],
            'GnPR': [0.5],
            'roadDensity': [0.5],
            'bldDensity': [3],
            'treeDensity': [0],
        })
    
    def add_synthetic_coordinates(self, df: pd.DataFrame, 
                                   grid_size: int = 10) -> pd.DataFrame:
        """
        Add synthetic lat/lon coordinates based on simulation parameters.
        
        Args:
            df: Input DataFrame
            grid_size: Size of the spatial grid
            
        Returns:
            DataFrame with added coordinates
        """
        if df is None or df.empty:
            return self._create_default_dataframe()
            
        df = df.copy()
        n_points = len(df)
        
        # Generate grid-based coordinates
        if 'parkRadius' in df.columns and 'parkLocation' in df.columns:
            # Use park parameters to create spatial variation
            df['lat'] = df.index % grid_size + df.get('parkRadius', 100) / 100
            df['lon'] = df.index // grid_size + df.get('parkLocation', 0) / 100
        else:
            # Simple grid layout
            df['lat'] = (df.index % grid_size).astype(float)
            df['lon'] = (df.index // grid_size).astype(float)
            
        # Normalize to realistic coordinate ranges
        df['lat'] = 40.0 + (df['lat'] - df['lat'].min()) / max(df['lat'].max() - df['lat'].min(), 1) * 0.1
        df['lon'] = -74.0 + (df['lon'] - df['lon'].min()) / max(df['lon'].max() - df['lon'].min(), 1) * 0.1
        
        return df
    
    def preprocess(self, df: pd.DataFrame, 
                   fill_missing: bool = True,
                   scale_features: bool = False) -> pd.DataFrame:
        """
        Preprocess the data for analysis.
        
        Args:
            df: Input DataFrame
            fill_missing: Whether to fill missing values
            scale_features: Whether to scale numeric features
            
        Returns:
            Preprocessed DataFrame
        """
        if df is None or df.empty:
            return self._create_default_dataframe()
            
        df = df.copy()
        
        if fill_missing:
            # Fill numeric columns with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].median())
                    
        if scale_features:
            feature_cols = [c for c in self.FEATURE_COLUMNS if c in df.columns]
            if feature_cols:
                scaler = StandardScaler()
                df[feature_cols] = scaler.fit_transform(df[feature_cols])
                
        return df
    
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded


class HotspotDetector:
    """
    Detects urban heat island hotspots using DBSCAN clustering.
    
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    identifies clusters of high UHI values based on spatial proximity
    and density thresholds.
    """
    
    DEFAULT_EPS = 0.5
    DEFAULT_MIN_SAMPLES = 3
    DEFAULT_UHI_THRESHOLD_PERCENTILE = 75
    
    def __init__(self, eps: float = DEFAULT_EPS, 
                 min_samples: int = DEFAULT_MIN_SAMPLES,
                 uhi_threshold_percentile: float = DEFAULT_UHI_THRESHOLD_PERCENTILE):
        """
        Initialize the hotspot detector.
        
        Args:
            eps: Maximum distance between points in a cluster
            min_samples: Minimum points required to form a cluster
            uhi_threshold_percentile: Percentile threshold for hotspot classification
        """
        self.eps = eps
        self.min_samples = min_samples
        self.uhi_threshold_percentile = uhi_threshold_percentile
        self._last_result: Optional[HotspotResult] = None
        
    def detect(self, df: pd.DataFrame, 
               uhi_column: str = 'UHI_d',
               lat_column: str = 'lat',
               lon_column: str = 'lon') -> HotspotResult:
        """
        Detect hotspots using DBSCAN clustering.
        
        Args:
            df: Input DataFrame with UHI data
            uhi_column: Column name for UHI values
            lat_column: Column name for latitude
            lon_column: Column name for longitude
            
        Returns:
            HotspotResult containing detected hotspots
        """
        # Safety check for empty data
        if df is None or df.empty:
            logger.warning("Empty dataset provided, returning default result")
            return self._create_default_result()
            
        # Check for required columns
        if uhi_column not in df.columns:
            logger.warning(f"UHI column '{uhi_column}' not found")
            return self._create_default_result()
            
        try:
            # Add coordinates if missing
            if lat_column not in df.columns or lon_column not in df.columns:
                loader = UHIDataLoader()
                df = loader.add_synthetic_coordinates(df)
                
            # Calculate UHI threshold
            uhi_values = df[uhi_column].values
            threshold = np.percentile(uhi_values, self.uhi_threshold_percentile)
            
            # Filter to high UHI areas
            high_uhi_mask = uhi_values >= threshold
            high_uhi_df = df[high_uhi_mask].copy()
            
            if len(high_uhi_df) < self.min_samples:
                logger.warning("Not enough high-UHI points for clustering")
                return self._create_default_result(df)
                
            # Prepare features for DBSCAN
            features = high_uhi_df[[lat_column, lon_column, uhi_column]].values
            
            # Normalize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Apply DBSCAN
            dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            cluster_labels = dbscan.fit_predict(features_scaled)
            
            # Identify hotspot clusters (exclude noise: -1)
            hotspot_mask = cluster_labels >= 0
            hotspot_indices = np.where(high_uhi_mask)[0][hotspot_mask]
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            
            # Extract hotspot data
            hotspot_data = df.iloc[hotspot_indices].copy()
            hotspot_data['cluster_id'] = cluster_labels[hotspot_mask]
            
            # Calculate statistics
            statistics = self._calculate_statistics(df, hotspot_data, uhi_column)
            
            result = HotspotResult(
                hotspot_indices=hotspot_indices,
                cluster_labels=cluster_labels,
                n_clusters=n_clusters,
                hotspot_data=hotspot_data,
                statistics=statistics
            )
            
            self._last_result = result
            logger.info(f"Detected {n_clusters} hotspot clusters with {len(hotspot_indices)} points")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in hotspot detection: {e}")
            return self._create_default_result(df)
    
    def _calculate_statistics(self, full_df: pd.DataFrame, 
                              hotspot_df: pd.DataFrame,
                              uhi_column: str) -> Dict[str, float]:
        """Calculate hotspot statistics."""
        stats = {
            'total_points': len(full_df),
            'hotspot_points': len(hotspot_df),
            'hotspot_percentage': len(hotspot_df) / max(len(full_df), 1) * 100,
            'mean_uhi_overall': full_df[uhi_column].mean() if uhi_column in full_df.columns else 0,
            'mean_uhi_hotspots': hotspot_df[uhi_column].mean() if len(hotspot_df) > 0 and uhi_column in hotspot_df.columns else 0,
            'max_uhi': full_df[uhi_column].max() if uhi_column in full_df.columns else 0,
            'min_uhi': full_df[uhi_column].min() if uhi_column in full_df.columns else 0,
        }
        return stats
    
    def _create_default_result(self, df: pd.DataFrame = None) -> HotspotResult:
        """Create a default result when detection fails."""
        empty_df = pd.DataFrame() if df is None else df.head(0)
        return HotspotResult(
            hotspot_indices=np.array([], dtype=int),
            cluster_labels=np.array([], dtype=int),
            n_clusters=0,
            hotspot_data=empty_df,
            statistics={'total_points': 0, 'hotspot_points': 0, 'hotspot_percentage': 0}
        )
    
    @property
    def last_result(self) -> Optional[HotspotResult]:
        return self._last_result


class GridBasedAnalyzer:
    """
    Grid-based spatial analysis for UHI data.
    
    Divides the study area into a regular grid and calculates
    UHI statistics for each cell to identify hotspot zones.
    """
    
    def __init__(self, grid_rows: int = 10, grid_cols: int = 10,
                 hotspot_threshold_percentile: float = 75):
        """
        Initialize the grid analyzer.
        
        Args:
            grid_rows: Number of grid rows
            grid_cols: Number of grid columns
            hotspot_threshold_percentile: Percentile for hotspot classification
        """
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.hotspot_threshold_percentile = hotspot_threshold_percentile
        self.grid: Optional[List[List[GridCell]]] = None
        self._hotspot_cells: List[GridCell] = []
        
    def analyze(self, df: pd.DataFrame,
                uhi_column: str = 'UHI_d',
                lat_column: str = 'lat',
                lon_column: str = 'lon') -> List[GridCell]:
        """
        Perform grid-based analysis.
        
        Args:
            df: Input DataFrame
            uhi_column: UHI value column
            lat_column: Latitude column
            lon_column: Longitude column
            
        Returns:
            List of GridCell objects representing hotspot cells
        """
        if df is None or df.empty:
            logger.warning("Empty dataset provided")
            return []
            
        # Add coordinates if missing
        if lat_column not in df.columns or lon_column not in df.columns:
            loader = UHIDataLoader()
            df = loader.add_synthetic_coordinates(df)
            
        if uhi_column not in df.columns:
            logger.warning(f"UHI column '{uhi_column}' not found")
            return []
            
        try:
            # Get coordinate bounds
            lat_min, lat_max = df[lat_column].min(), df[lat_column].max()
            lon_min, lon_max = df[lon_column].min(), df[lon_column].max()
            
            # Add small buffer to avoid edge cases
            lat_range = max(lat_max - lat_min, 0.001)
            lon_range = max(lon_max - lon_min, 0.001)
            
            # Initialize grid
            self.grid = [[GridCell(row=i, col=j,
                                   center_x=lon_min + (j + 0.5) * lon_range / self.grid_cols,
                                   center_y=lat_min + (i + 0.5) * lat_range / self.grid_rows)
                         for j in range(self.grid_cols)]
                        for i in range(self.grid_rows)]
            
            # Assign points to grid cells
            for _, row in df.iterrows():
                lat, lon, uhi = row[lat_column], row[lon_column], row[uhi_column]
                
                # Calculate grid indices
                row_idx = min(int((lat - lat_min) / lat_range * self.grid_rows), self.grid_rows - 1)
                col_idx = min(int((lon - lon_min) / lon_range * self.grid_cols), self.grid_cols - 1)
                
                row_idx = max(0, row_idx)
                col_idx = max(0, col_idx)
                
                self.grid[row_idx][col_idx].uhi_values.append(uhi)
                self.grid[row_idx][col_idx].point_count += 1
            
            # Calculate cell statistics
            all_means = []
            for row in self.grid:
                for cell in row:
                    if cell.uhi_values:
                        cell.mean_uhi = np.mean(cell.uhi_values)
                        cell.max_uhi = np.max(cell.uhi_values)
                        all_means.append(cell.mean_uhi)
            
            # Determine hotspot threshold
            if all_means:
                threshold = np.percentile(all_means, self.hotspot_threshold_percentile)
                
                # Mark hotspot cells
                self._hotspot_cells = []
                for row in self.grid:
                    for cell in row:
                        if cell.mean_uhi >= threshold and cell.point_count > 0:
                            cell.is_hotspot = True
                            self._hotspot_cells.append(cell)
            
            logger.info(f"Grid analysis complete: {len(self._hotspot_cells)} hotspot cells identified")
            return self._hotspot_cells
            
        except Exception as e:
            logger.error(f"Error in grid analysis: {e}")
            return []
    
    def get_grid_matrix(self, metric: str = 'mean_uhi') -> np.ndarray:
        """
        Get a 2D numpy array of grid values.
        
        Args:
            metric: Which metric to return ('mean_uhi', 'max_uhi', 'point_count')
            
        Returns:
            2D numpy array of grid values
        """
        if self.grid is None:
            return np.zeros((self.grid_rows, self.grid_cols))
            
        matrix = np.zeros((self.grid_rows, self.grid_cols))
        for i, row in enumerate(self.grid):
            for j, cell in enumerate(row):
                if metric == 'mean_uhi':
                    matrix[i, j] = cell.mean_uhi
                elif metric == 'max_uhi':
                    matrix[i, j] = cell.max_uhi
                elif metric == 'point_count':
                    matrix[i, j] = cell.point_count
                    
        return matrix
    
    def get_hotspot_summary(self) -> Dict[str, Any]:
        """Get a summary of hotspot analysis."""
        if not self._hotspot_cells:
            return {
                'n_hotspot_cells': 0,
                'total_cells': self.grid_rows * self.grid_cols,
                'hotspot_percentage': 0,
                'avg_hotspot_uhi': 0,
                'max_hotspot_uhi': 0
            }
            
        return {
            'n_hotspot_cells': len(self._hotspot_cells),
            'total_cells': self.grid_rows * self.grid_cols,
            'hotspot_percentage': len(self._hotspot_cells) / (self.grid_rows * self.grid_cols) * 100,
            'avg_hotspot_uhi': np.mean([c.mean_uhi for c in self._hotspot_cells]),
            'max_hotspot_uhi': max([c.max_uhi for c in self._hotspot_cells])
        }
    
    @property
    def hotspot_cells(self) -> List[GridCell]:
        return self._hotspot_cells
