"""
Urban Landscape Generator
=========================

Procedural generation of urban elements for AR/VR visualization:
- Buildings with varying heights, footprints, and types
- Street networks and road systems
- Green spaces, trees, and vegetation
- Heat influence zones around hotspots

This module creates realistic urban environments based on
UHI data parameters like building density, tree density, etc.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
import json
import logging
import random
import math

logger = logging.getLogger(__name__)


class BuildingType(Enum):
    """Types of buildings in the urban landscape."""
    RESIDENTIAL_LOW = "residential_low"      # 1-3 floors
    RESIDENTIAL_MID = "residential_mid"      # 4-8 floors
    RESIDENTIAL_HIGH = "residential_high"    # 9+ floors
    COMMERCIAL = "commercial"                # Offices, shops
    INDUSTRIAL = "industrial"                # Factories, warehouses
    MIXED_USE = "mixed_use"                  # Residential + Commercial
    PUBLIC = "public"                        # Schools, hospitals


class VegetationType(Enum):
    """Types of vegetation."""
    TREE_DECIDUOUS = "tree_deciduous"
    TREE_CONIFER = "tree_conifer"
    TREE_PALM = "tree_palm"
    SHRUB = "shrub"
    GRASS_PATCH = "grass_patch"
    PARK = "park"


class RoadType(Enum):
    """Types of roads."""
    HIGHWAY = "highway"
    MAIN_ROAD = "main_road"
    STREET = "street"
    ALLEY = "alley"
    PEDESTRIAN = "pedestrian"


@dataclass
class Building:
    """Represents a single building in the urban landscape."""
    id: str
    x: float
    y: float
    z: float = 0  # Ground level
    width: float = 10
    depth: float = 10
    height: float = 20
    building_type: BuildingType = BuildingType.RESIDENTIAL_MID
    rotation: float = 0  # Degrees
    floors: int = 4
    heat_exposure: float = 0  # 0-1, how much UHI affects this building
    color: Tuple[float, float, float] = (0.7, 0.7, 0.7)
    has_green_roof: bool = False
    roof_type: str = "flat"  # flat, pitched, dome
    window_density: float = 0.3  # 0-1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'position': {'x': self.x, 'y': self.y, 'z': self.z},
            'dimensions': {'width': self.width, 'depth': self.depth, 'height': self.height},
            'type': self.building_type.value,
            'rotation': self.rotation,
            'floors': self.floors,
            'heat_exposure': self.heat_exposure,
            'color': {'r': self.color[0], 'g': self.color[1], 'b': self.color[2]},
            'has_green_roof': self.has_green_roof,
            'roof_type': self.roof_type,
            'window_density': self.window_density
        }


@dataclass
class Tree:
    """Represents a tree or vegetation element."""
    id: str
    x: float
    y: float
    z: float = 0
    height: float = 8
    canopy_radius: float = 4
    trunk_radius: float = 0.3
    vegetation_type: VegetationType = VegetationType.TREE_DECIDUOUS
    color: Tuple[float, float, float] = (0.2, 0.6, 0.2)
    cooling_effect: float = 0.5  # Cooling radius multiplier
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'position': {'x': self.x, 'y': self.y, 'z': self.z},
            'height': self.height,
            'canopy_radius': self.canopy_radius,
            'trunk_radius': self.trunk_radius,
            'type': self.vegetation_type.value,
            'color': {'r': self.color[0], 'g': self.color[1], 'b': self.color[2]},
            'cooling_effect': self.cooling_effect
        }


@dataclass
class Road:
    """Represents a road segment."""
    id: str
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    width: float = 8
    road_type: RoadType = RoadType.STREET
    has_sidewalk: bool = True
    is_asphalt: bool = True
    heat_absorption: float = 0.8  # Asphalt absorbs more heat
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'start': {'x': self.start_x, 'y': self.start_y},
            'end': {'x': self.end_x, 'y': self.end_y},
            'width': self.width,
            'type': self.road_type.value,
            'has_sidewalk': self.has_sidewalk,
            'is_asphalt': self.is_asphalt,
            'heat_absorption': self.heat_absorption
        }


@dataclass
class HotspotZone:
    """Represents a heat hotspot zone for visualization."""
    id: str
    center_x: float
    center_y: float
    radius: float
    intensity: float  # 0-1
    uhi_value: float
    color: Tuple[float, float, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'center': {'x': self.center_x, 'y': self.center_y},
            'radius': self.radius,
            'intensity': self.intensity,
            'uhi_value': self.uhi_value,
            'color': {'r': self.color[0], 'g': self.color[1], 'b': self.color[2]}
        }


@dataclass
class UrbanLandscape:
    """Complete urban landscape data structure."""
    buildings: List[Building] = field(default_factory=list)
    trees: List[Tree] = field(default_factory=list)
    roads: List[Road] = field(default_factory=list)
    hotspot_zones: List[HotspotZone] = field(default_factory=list)
    terrain_size: Tuple[float, float] = (200, 200)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'buildings': [b.to_dict() for b in self.buildings],
            'trees': [t.to_dict() for t in self.trees],
            'roads': [r.to_dict() for r in self.roads],
            'hotspot_zones': [h.to_dict() for h in self.hotspot_zones],
            'terrain_size': {'width': self.terrain_size[0], 'depth': self.terrain_size[1]},
            'metadata': self.metadata,
            'statistics': {
                'building_count': len(self.buildings),
                'tree_count': len(self.trees),
                'road_count': len(self.roads),
                'hotspot_count': len(self.hotspot_zones)
            }
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class UrbanLandscapeGenerator:
    """
    Generates procedural urban landscapes based on UHI data parameters.
    
    Uses building density, tree density, road density, and other urban
    metrics to create realistic city environments for visualization.
    """
    
    # Default building parameters by type
    BUILDING_PARAMS = {
        BuildingType.RESIDENTIAL_LOW: {'height_range': (6, 12), 'floors': (2, 4), 'footprint': (8, 15)},
        BuildingType.RESIDENTIAL_MID: {'height_range': (15, 30), 'floors': (5, 10), 'footprint': (12, 20)},
        BuildingType.RESIDENTIAL_HIGH: {'height_range': (30, 80), 'floors': (10, 25), 'footprint': (15, 25)},
        BuildingType.COMMERCIAL: {'height_range': (20, 60), 'floors': (5, 15), 'footprint': (20, 40)},
        BuildingType.INDUSTRIAL: {'height_range': (8, 20), 'floors': (1, 3), 'footprint': (30, 60)},
        BuildingType.MIXED_USE: {'height_range': (15, 40), 'floors': (4, 12), 'footprint': (15, 30)},
        BuildingType.PUBLIC: {'height_range': (10, 25), 'floors': (2, 6), 'footprint': (25, 50)},
    }
    
    # Color palettes for buildings
    BUILDING_COLORS = {
        BuildingType.RESIDENTIAL_LOW: [(0.85, 0.8, 0.75), (0.9, 0.85, 0.8), (0.75, 0.7, 0.65)],
        BuildingType.RESIDENTIAL_MID: [(0.7, 0.7, 0.75), (0.8, 0.75, 0.7), (0.65, 0.65, 0.7)],
        BuildingType.RESIDENTIAL_HIGH: [(0.6, 0.65, 0.7), (0.55, 0.6, 0.65), (0.5, 0.55, 0.6)],
        BuildingType.COMMERCIAL: [(0.4, 0.5, 0.6), (0.45, 0.45, 0.5), (0.35, 0.4, 0.45)],
        BuildingType.INDUSTRIAL: [(0.5, 0.5, 0.5), (0.55, 0.5, 0.45), (0.45, 0.45, 0.45)],
        BuildingType.MIXED_USE: [(0.65, 0.6, 0.55), (0.7, 0.65, 0.6), (0.6, 0.55, 0.5)],
        BuildingType.PUBLIC: [(0.75, 0.7, 0.65), (0.8, 0.75, 0.7), (0.7, 0.65, 0.6)],
    }
    
    def __init__(self, seed: int = 42):
        """Initialize the generator with a random seed for reproducibility."""
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.landscape: Optional[UrbanLandscape] = None
        
    def generate_from_dataframe(self, 
                                df: pd.DataFrame,
                                hotspot_df: pd.DataFrame = None,
                                terrain_size: Tuple[float, float] = (200, 200),
                                scale_factor: float = 1.0) -> UrbanLandscape:
        """
        Generate urban landscape from UHI DataFrame.
        
        Args:
            df: DataFrame with urban parameters (bldDensity, treeDensity, etc.)
            hotspot_df: DataFrame with hotspot locations
            terrain_size: Size of terrain (width, depth)
            scale_factor: Scale multiplier for the scene
            
        Returns:
            UrbanLandscape with generated elements
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame, generating default landscape")
            return self._generate_default_landscape(terrain_size)
        
        # Extract parameters from DataFrame (use mean values)
        params = self._extract_parameters(df)
        
        # Initialize landscape
        self.landscape = UrbanLandscape(terrain_size=terrain_size)
        
        # Generate hotspot zones first (they influence building placement)
        if hotspot_df is not None and not hotspot_df.empty:
            self._generate_hotspot_zones(hotspot_df, terrain_size, scale_factor)
        
        # Generate road network
        self._generate_roads(params, terrain_size)
        
        # Generate buildings around hotspots
        self._generate_buildings(params, terrain_size, scale_factor)
        
        # Generate vegetation
        self._generate_vegetation(params, terrain_size)
        
        # Apply heat exposure to buildings based on hotspot proximity
        self._apply_heat_exposure()
        
        # Store metadata
        self.landscape.metadata = {
            'source_data_rows': len(df),
            'parameters': params,
            'seed': self.seed,
            'scale_factor': scale_factor
        }
        
        logger.info(f"Generated landscape: {len(self.landscape.buildings)} buildings, "
                   f"{len(self.landscape.trees)} trees, {len(self.landscape.roads)} roads")
        
        return self.landscape
    
    def _extract_parameters(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract urban parameters from DataFrame."""
        params = {
            'building_density': df['bldDensity'].mean() if 'bldDensity' in df.columns else 3.0,
            'tree_density': df['treeDensity'].mean() if 'treeDensity' in df.columns else 0.1,
            'road_density': df['roadDensity'].mean() if 'roadDensity' in df.columns else 0.3,
            'green_ratio': df['GnPR'].mean() if 'GnPR' in df.columns else 0.3,
            'avg_building_height': df['avg_BH'].mean() if 'avg_BH' in df.columns else 20,
            'asphalt_ratio': df['asphalt_ratio'].mean() if 'asphalt_ratio' in df.columns else 0.1,
            'greenroof_ratio': df['greenroof_ratio'].mean() if 'greenroof_ratio' in df.columns else 0.05,
            'park_grass_ratio': df['park_grass_ratio'].mean() if 'park_grass_ratio' in df.columns else 0.1,
        }
        return params
    
    def _generate_hotspot_zones(self, hotspot_df: pd.DataFrame, 
                                terrain_size: Tuple[float, float],
                                scale_factor: float):
        """Generate hotspot visualization zones."""
        uhi_col = 'UHI_d' if 'UHI_d' in hotspot_df.columns else None
        if uhi_col is None:
            for col in hotspot_df.columns:
                if 'UHI' in col.upper():
                    uhi_col = col
                    break
        
        if uhi_col is None:
            return
            
        uhi_min = hotspot_df[uhi_col].min()
        uhi_max = hotspot_df[uhi_col].max()
        uhi_range = max(uhi_max - uhi_min, 0.001)
        
        # Check for coordinate columns
        lat_col = 'lat' if 'lat' in hotspot_df.columns else None
        lon_col = 'lon' if 'lon' in hotspot_df.columns else None
        
        for idx, row in hotspot_df.iterrows():
            # Get position
            if lat_col and lon_col:
                x = (row[lon_col] - hotspot_df[lon_col].min()) * scale_factor * 100
                y = (row[lat_col] - hotspot_df[lat_col].min()) * scale_factor * 100
            else:
                x = (idx % 10) * (terrain_size[0] / 10) + random.uniform(-5, 5)
                y = (idx // 10) * (terrain_size[1] / 10) + random.uniform(-5, 5)
            
            # Normalize to terrain
            x = min(max(x, 0), terrain_size[0])
            y = min(max(y, 0), terrain_size[1])
            
            uhi_value = row[uhi_col]
            intensity = (uhi_value - uhi_min) / uhi_range
            
            # Color based on intensity
            color = self._heat_to_color(intensity)
            
            # Radius based on intensity
            radius = 10 + intensity * 30
            
            zone = HotspotZone(
                id=f"hotspot_{idx:04d}",
                center_x=x,
                center_y=y,
                radius=radius,
                intensity=intensity,
                uhi_value=uhi_value,
                color=color
            )
            self.landscape.hotspot_zones.append(zone)
    
    def _generate_roads(self, params: Dict[str, float], terrain_size: Tuple[float, float]):
        """Generate road network based on road density."""
        road_density = params.get('road_density', 0.3)
        
        # Number of roads based on density
        n_main_roads = max(2, int(road_density * 8))
        n_streets = max(4, int(road_density * 20))
        
        road_id = 0
        
        # Generate main roads (horizontal and vertical)
        for i in range(n_main_roads // 2):
            # Horizontal main road
            y = terrain_size[1] * (i + 1) / (n_main_roads // 2 + 1)
            road = Road(
                id=f"road_{road_id:04d}",
                start_x=0,
                start_y=y,
                end_x=terrain_size[0],
                end_y=y,
                width=12,
                road_type=RoadType.MAIN_ROAD
            )
            self.landscape.roads.append(road)
            road_id += 1
            
            # Vertical main road
            x = terrain_size[0] * (i + 1) / (n_main_roads // 2 + 1)
            road = Road(
                id=f"road_{road_id:04d}",
                start_x=x,
                start_y=0,
                end_x=x,
                end_y=terrain_size[1],
                width=12,
                road_type=RoadType.MAIN_ROAD
            )
            self.landscape.roads.append(road)
            road_id += 1
        
        # Generate smaller streets
        for i in range(n_streets):
            if random.random() < 0.5:
                # Horizontal street
                y = random.uniform(10, terrain_size[1] - 10)
                x_start = random.uniform(0, terrain_size[0] / 2)
                x_end = random.uniform(x_start + 20, terrain_size[0])
            else:
                # Vertical street
                x = random.uniform(10, terrain_size[0] - 10)
                y_start = random.uniform(0, terrain_size[1] / 2)
                y_end = random.uniform(y_start + 20, terrain_size[1])
                x_start, x_end = x, x
                y = y_start
                y_end_temp = y_end
                
            road = Road(
                id=f"road_{road_id:04d}",
                start_x=x_start if random.random() < 0.5 else x,
                start_y=y if 'y_start' not in dir() else y_start,
                end_x=x_end if random.random() < 0.5 else x,
                end_y=y if 'y_end_temp' not in dir() else y_end_temp,
                width=6 + random.random() * 2,
                road_type=RoadType.STREET
            )
            self.landscape.roads.append(road)
            road_id += 1
    
    def _generate_buildings(self, params: Dict[str, float], 
                           terrain_size: Tuple[float, float],
                           scale_factor: float):
        """Generate buildings based on density and height parameters."""
        building_density = params.get('building_density', 3.0)
        avg_height = params.get('avg_building_height', 20)
        greenroof_ratio = params.get('greenroof_ratio', 0.05)
        
        # Number of buildings based on density (scale with terrain)
        n_buildings = int(building_density * terrain_size[0] * terrain_size[1] / 400)
        n_buildings = max(20, min(n_buildings, 500))  # Clamp
        
        # Determine building type distribution based on height
        type_weights = self._get_building_type_weights(avg_height, building_density)
        
        building_id = 0
        placed_positions = []  # Track placed buildings to avoid overlap
        
        for _ in range(n_buildings):
            # Select building type
            building_type = random.choices(
                list(type_weights.keys()),
                weights=list(type_weights.values())
            )[0]
            
            params_type = self.BUILDING_PARAMS[building_type]
            
            # Generate building dimensions
            base_height = random.uniform(*params_type['height_range'])
            # Scale height based on average
            height = base_height * (avg_height / 20)
            height = max(5, min(height, 100))  # Clamp height
            
            floors = max(1, int(height / 3))
            footprint_base = random.uniform(*params_type['footprint'])
            width = footprint_base * (0.8 + random.random() * 0.4)
            depth = footprint_base * (0.8 + random.random() * 0.4)
            
            # Find position (avoid overlaps and roads)
            for attempt in range(20):
                x = random.uniform(width/2, terrain_size[0] - width/2)
                y = random.uniform(depth/2, terrain_size[1] - depth/2)
                
                # Check overlap with existing buildings
                overlap = False
                for px, py, pw, pd in placed_positions:
                    if (abs(x - px) < (width + pw) / 2 + 3 and 
                        abs(y - py) < (depth + pd) / 2 + 3):
                        overlap = True
                        break
                
                if not overlap:
                    break
            else:
                continue  # Skip if can't place
            
            # Bias buildings toward hotspot areas for visual effect
            hotspot_boost = self._calculate_hotspot_influence(x, y)
            
            # Building color
            color = random.choice(self.BUILDING_COLORS.get(
                building_type, 
                [(0.7, 0.7, 0.7)]
            ))
            
            # Apply heat tint if near hotspot
            if hotspot_boost > 0.3:
                color = self._apply_heat_tint(color, hotspot_boost)
            
            building = Building(
                id=f"building_{building_id:04d}",
                x=x,
                y=y,
                z=0,
                width=width,
                depth=depth,
                height=height,
                building_type=building_type,
                rotation=random.choice([0, 90, 180, 270]),
                floors=floors,
                heat_exposure=hotspot_boost,
                color=color,
                has_green_roof=random.random() < greenroof_ratio,
                roof_type=random.choice(['flat', 'flat', 'pitched']) if building_type in 
                         [BuildingType.RESIDENTIAL_LOW, BuildingType.RESIDENTIAL_MID] else 'flat',
                window_density=0.2 + random.random() * 0.4
            )
            
            self.landscape.buildings.append(building)
            placed_positions.append((x, y, width, depth))
            building_id += 1
    
    def _generate_vegetation(self, params: Dict[str, float], terrain_size: Tuple[float, float]):
        """Generate trees and vegetation."""
        tree_density = params.get('tree_density', 0.1)
        green_ratio = params.get('green_ratio', 0.3)
        park_ratio = params.get('park_grass_ratio', 0.1)
        
        # Number of trees based on density
        n_trees = int((tree_density + green_ratio * 0.5) * terrain_size[0] * terrain_size[1] / 50)
        n_trees = max(10, min(n_trees, 300))
        
        tree_id = 0
        
        for _ in range(n_trees):
            x = random.uniform(2, terrain_size[0] - 2)
            y = random.uniform(2, terrain_size[1] - 2)
            
            # Check if position conflicts with buildings
            conflict = False
            for building in self.landscape.buildings:
                if (abs(x - building.x) < building.width/2 + 2 and 
                    abs(y - building.y) < building.depth/2 + 2):
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # Tree type based on location
            if random.random() < park_ratio:
                veg_type = VegetationType.TREE_DECIDUOUS
                height = random.uniform(8, 15)
                canopy = random.uniform(4, 8)
            else:
                veg_type = random.choice([
                    VegetationType.TREE_DECIDUOUS,
                    VegetationType.TREE_CONIFER,
                    VegetationType.SHRUB
                ])
                if veg_type == VegetationType.SHRUB:
                    height = random.uniform(1, 3)
                    canopy = random.uniform(1, 2)
                else:
                    height = random.uniform(5, 12)
                    canopy = random.uniform(2, 5)
            
            # Color variation
            green_var = random.uniform(-0.1, 0.1)
            color = (0.15 + green_var, 0.5 + green_var, 0.15 + green_var)
            
            tree = Tree(
                id=f"tree_{tree_id:04d}",
                x=x,
                y=y,
                z=0,
                height=height,
                canopy_radius=canopy,
                trunk_radius=height * 0.05,
                vegetation_type=veg_type,
                color=color,
                cooling_effect=0.3 + random.random() * 0.4
            )
            
            self.landscape.trees.append(tree)
            tree_id += 1
    
    def _get_building_type_weights(self, avg_height: float, density: float) -> Dict[BuildingType, float]:
        """Calculate building type distribution based on parameters."""
        if avg_height > 40:
            # High-rise area
            return {
                BuildingType.RESIDENTIAL_HIGH: 0.4,
                BuildingType.COMMERCIAL: 0.3,
                BuildingType.RESIDENTIAL_MID: 0.2,
                BuildingType.MIXED_USE: 0.08,
                BuildingType.PUBLIC: 0.02,
            }
        elif avg_height > 20:
            # Mid-rise area
            return {
                BuildingType.RESIDENTIAL_MID: 0.35,
                BuildingType.COMMERCIAL: 0.25,
                BuildingType.RESIDENTIAL_HIGH: 0.15,
                BuildingType.MIXED_USE: 0.15,
                BuildingType.PUBLIC: 0.05,
                BuildingType.RESIDENTIAL_LOW: 0.05,
            }
        else:
            # Low-rise area
            return {
                BuildingType.RESIDENTIAL_LOW: 0.4,
                BuildingType.RESIDENTIAL_MID: 0.25,
                BuildingType.INDUSTRIAL: 0.15,
                BuildingType.COMMERCIAL: 0.1,
                BuildingType.PUBLIC: 0.05,
                BuildingType.MIXED_USE: 0.05,
            }
    
    def _calculate_hotspot_influence(self, x: float, y: float) -> float:
        """Calculate how much a position is influenced by nearby hotspots."""
        if not self.landscape.hotspot_zones:
            return 0.0
        
        max_influence = 0.0
        for zone in self.landscape.hotspot_zones:
            distance = math.sqrt((x - zone.center_x)**2 + (y - zone.center_y)**2)
            if distance < zone.radius:
                influence = zone.intensity * (1 - distance / zone.radius)
                max_influence = max(max_influence, influence)
        
        return max_influence
    
    def _apply_heat_exposure(self):
        """Apply heat exposure values to buildings based on hotspot proximity."""
        for building in self.landscape.buildings:
            exposure = self._calculate_hotspot_influence(building.x, building.y)
            building.heat_exposure = exposure
            
            # Update color if high exposure
            if exposure > 0.5:
                building.color = self._apply_heat_tint(building.color, exposure)
    
    def _heat_to_color(self, intensity: float) -> Tuple[float, float, float]:
        """Convert heat intensity to RGB color."""
        intensity = max(0, min(1, intensity))
        
        if intensity < 0.25:
            r = 0
            g = intensity * 4
            b = 1
        elif intensity < 0.5:
            r = 0
            g = 1
            b = 1 - (intensity - 0.25) * 4
        elif intensity < 0.75:
            r = (intensity - 0.5) * 4
            g = 1
            b = 0
        else:
            r = 1
            g = 1 - (intensity - 0.75) * 4
            b = 0
        
        return (r, g, b)
    
    def _apply_heat_tint(self, color: Tuple[float, float, float], 
                         intensity: float) -> Tuple[float, float, float]:
        """Apply heat tint to a color based on intensity."""
        heat_color = self._heat_to_color(intensity)
        blend = min(intensity * 0.3, 0.3)  # Max 30% tint
        
        return (
            color[0] * (1 - blend) + heat_color[0] * blend,
            color[1] * (1 - blend) + heat_color[1] * blend,
            color[2] * (1 - blend) + heat_color[2] * blend
        )
    
    def _generate_default_landscape(self, terrain_size: Tuple[float, float]) -> UrbanLandscape:
        """Generate a default landscape when no data is available."""
        self.landscape = UrbanLandscape(terrain_size=terrain_size)
        
        # Add some default buildings
        for i in range(20):
            building = Building(
                id=f"building_{i:04d}",
                x=20 + (i % 5) * 35,
                y=20 + (i // 5) * 40,
                width=15 + random.random() * 10,
                depth=15 + random.random() * 10,
                height=15 + random.random() * 30,
                building_type=random.choice(list(BuildingType)),
                color=(0.6 + random.random() * 0.2,) * 3
            )
            self.landscape.buildings.append(building)
        
        # Add some trees
        for i in range(30):
            tree = Tree(
                id=f"tree_{i:04d}",
                x=random.uniform(5, terrain_size[0] - 5),
                y=random.uniform(5, terrain_size[1] - 5),
                height=5 + random.random() * 10
            )
            self.landscape.trees.append(tree)
        
        return self.landscape


class CityBlockGenerator:
    """
    Generates city blocks with organized building layouts.
    
    Creates more structured urban environments with proper
    block-based organization of buildings and streets.
    """
    
    def __init__(self, block_size: float = 40, street_width: float = 8):
        self.block_size = block_size
        self.street_width = street_width
        
    def generate_blocks(self, terrain_size: Tuple[float, float],
                       building_params: Dict[str, float]) -> UrbanLandscape:
        """Generate organized city blocks."""
        landscape = UrbanLandscape(terrain_size=terrain_size)
        
        # Calculate number of blocks
        n_blocks_x = int(terrain_size[0] / (self.block_size + self.street_width))
        n_blocks_y = int(terrain_size[1] / (self.block_size + self.street_width))
        
        building_id = 0
        
        for bx in range(n_blocks_x):
            for by in range(n_blocks_y):
                # Block origin
                origin_x = bx * (self.block_size + self.street_width) + self.street_width
                origin_y = by * (self.block_size + self.street_width) + self.street_width
                
                # Generate buildings within block
                buildings_in_block = random.randint(2, 6)
                
                for i in range(buildings_in_block):
                    local_x = random.uniform(2, self.block_size - 12)
                    local_y = random.uniform(2, self.block_size - 12)
                    
                    building = Building(
                        id=f"building_{building_id:04d}",
                        x=origin_x + local_x,
                        y=origin_y + local_y,
                        width=8 + random.random() * 8,
                        depth=8 + random.random() * 8,
                        height=10 + random.random() * 40,
                        building_type=random.choice(list(BuildingType)),
                        color=(0.5 + random.random() * 0.3,) * 3
                    )
                    landscape.buildings.append(building)
                    building_id += 1
        
        return landscape
