"""
AR/VR Visualization Module - Enhanced Urban Landscape
=====================================================

Generates immersive visualization outputs featuring:
- Procedural buildings with heat exposure coloring
- Street networks and infrastructure
- Trees and green spaces
- Heat hotspot zones with visual effects
- WebXR VR support for immersive viewing

Supported platforms:
- Three.js HTML for web-based 3D visualization (with WebXR VR)
- Unity JSON for game engine import
- Blender Python script for automated 3D scene creation
"""

import json
import logging
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict, field
import numpy as np
import pandas as pd
from pathlib import Path

# Import urban generator for building generation
try:
    from .urban_generator import (
        UrbanLandscapeGenerator, UrbanLandscape,
        Building, Tree, Road, HotspotZone
    )
    URBAN_GENERATOR_AVAILABLE = True
except ImportError:
    URBAN_GENERATOR_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class HotspotPoint:
    """Represents a 3D hotspot point for visualization."""
    x: float
    y: float
    z: float  # UHI intensity as height
    uhi_value: float
    intensity: float  # Normalized 0-1
    color: Tuple[float, float, float]  # RGB
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'position': {'x': self.x, 'y': self.y, 'z': self.z},
            'uhi_value': self.uhi_value,
            'intensity': self.intensity,
            'color': {'r': self.color[0], 'g': self.color[1], 'b': self.color[2]}
        }


class BaseVisualizationGenerator:
    """Base class for visualization generators."""
    
    def __init__(self):
        self.hotspot_points: List[HotspotPoint] = []
        self.terrain_data: np.ndarray = None
        self.metadata: Dict[str, Any] = {}
        
    def prepare_data(self, df: pd.DataFrame,
                     uhi_column: str = 'UHI_d',
                     lat_column: str = 'lat',
                     lon_column: str = 'lon',
                     scale_factor: float = 100.0,
                     height_multiplier: float = 50.0) -> List[HotspotPoint]:
        """
        Prepare hotspot data for 3D visualization.
        
        Args:
            df: DataFrame with hotspot data
            uhi_column: Column name for UHI values
            lat_column: Latitude column
            lon_column: Longitude column
            scale_factor: Scale for x/y coordinates
            height_multiplier: Multiplier for UHI -> height conversion
            
        Returns:
            List of HotspotPoint objects
        """
        if df is None or df.empty:
            logger.warning("Empty data provided, returning default points")
            return self._get_default_points()
            
        try:
            # Check required columns
            required = [uhi_column]
            for col in required:
                if col not in df.columns:
                    logger.warning(f"Missing column: {col}")
                    return self._get_default_points()
            
            # Add coordinates if missing
            if lat_column not in df.columns or lon_column not in df.columns:
                df = df.copy()
                df[lat_column] = np.arange(len(df)) % 10
                df[lon_column] = np.arange(len(df)) // 10
            
            # Get value ranges for normalization
            uhi_min = df[uhi_column].min()
            uhi_max = df[uhi_column].max()
            uhi_range = max(uhi_max - uhi_min, 0.001)
            
            lat_min = df[lat_column].min()
            lon_min = df[lon_column].min()
            
            self.hotspot_points = []
            
            for _, row in df.iterrows():
                # Normalize and scale coordinates
                x = (row[lon_column] - lon_min) * scale_factor
                y = (row[lat_column] - lat_min) * scale_factor
                
                uhi_value = row[uhi_column]
                intensity = (uhi_value - uhi_min) / uhi_range
                z = intensity * height_multiplier
                
                # Color based on intensity (blue -> yellow -> red)
                color = self._intensity_to_color(intensity)
                
                point = HotspotPoint(
                    x=float(x),
                    y=float(y),
                    z=float(z),
                    uhi_value=float(uhi_value),
                    intensity=float(intensity),
                    color=color
                )
                self.hotspot_points.append(point)
            
            # Store metadata
            self.metadata = {
                'point_count': len(self.hotspot_points),
                'uhi_min': float(uhi_min),
                'uhi_max': float(uhi_max),
                'scale_factor': scale_factor,
                'height_multiplier': height_multiplier
            }
            
            logger.info(f"Prepared {len(self.hotspot_points)} points for visualization")
            return self.hotspot_points
            
        except Exception as e:
            logger.error(f"Error preparing visualization data: {e}")
            return self._get_default_points()
    
    def _intensity_to_color(self, intensity: float) -> Tuple[float, float, float]:
        """Convert intensity (0-1) to RGB color (heat map)."""
        intensity = max(0, min(1, intensity))
        
        if intensity < 0.25:
            # Blue to Cyan
            r = 0
            g = intensity * 4
            b = 1
        elif intensity < 0.5:
            # Cyan to Green
            r = 0
            g = 1
            b = 1 - (intensity - 0.25) * 4
        elif intensity < 0.75:
            # Green to Yellow
            r = (intensity - 0.5) * 4
            g = 1
            b = 0
        else:
            # Yellow to Red
            r = 1
            g = 1 - (intensity - 0.75) * 4
            b = 0
            
        return (r, g, b)
    
    def _get_default_points(self) -> List[HotspotPoint]:
        """Return default points when data preparation fails."""
        return [
            HotspotPoint(x=0, y=0, z=10, uhi_value=0.15, intensity=0.5, color=(1, 0.5, 0)),
            HotspotPoint(x=10, y=0, z=15, uhi_value=0.18, intensity=0.7, color=(1, 0.3, 0)),
            HotspotPoint(x=5, y=10, z=20, uhi_value=0.2, intensity=0.9, color=(1, 0, 0)),
        ]


class ThreeJSGenerator(BaseVisualizationGenerator):
    """
    Generates Three.js HTML visualization for web-based 3D display.
    
    Creates an interactive 3D scene with:
    - Procedural buildings with heat exposure coloring
    - Trees and vegetation
    - Street networks
    - Heat hotspot pillars and zones
    - WebXR VR support
    - Day/night cycle
    - Interactive building info panels
    """
    
    # Enhanced template with buildings and VR support
    HTML_TEMPLATE_ENHANCED = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UHI Urban Landscape - Immersive 3D Visualization</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ overflow: hidden; font-family: 'Segoe UI', Arial, sans-serif; }}
        #info {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.85);
            color: white;
            padding: 20px;
            border-radius: 12px;
            z-index: 100;
            max-width: 320px;
            backdrop-filter: blur(10px);
        }}
        #info h2 {{ margin-bottom: 15px; color: #ff6b6b; font-size: 1.4em; }}
        #info p {{ margin: 8px 0; font-size: 13px; line-height: 1.4; }}
        #info .stat {{ color: #4ecdc4; font-weight: bold; }}
        #controls {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0,0,0,0.85);
            color: white;
            padding: 15px;
            border-radius: 12px;
            z-index: 100;
        }}
        #controls button {{
            display: block;
            width: 100%;
            margin: 8px 0;
            padding: 10px 15px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.3s;
        }}
        .btn-primary {{ background: #4ecdc4; color: #1a1a2e; }}
        .btn-primary:hover {{ background: #3dbdb5; }}
        .btn-primary.active {{ background: #2db8a6; box-shadow: 0 0 12px #00ff88; border: 2px solid #00ff88; }}
        .btn-secondary {{ background: #444; color: white; }}
        .btn-secondary:hover {{ background: #555; }}
        .btn-vr {{ background: #ff6b6b; color: white; }}
        .btn-vr:hover {{ background: #ff5252; }}
        #legend {{
            position: absolute;
            bottom: 20px;
            right: 20px;
            background: rgba(0,0,0,0.85);
            color: white;
            padding: 15px;
            border-radius: 12px;
        }}
        .legend-item {{ display: flex; align-items: center; margin: 6px 0; }}
        .legend-color {{ width: 24px; height: 24px; margin-right: 12px; border-radius: 4px; }}
        #tooltip {{
            position: absolute;
            background: rgba(0,0,0,0.9);
            color: white;
            padding: 12px 16px;
            border-radius: 8px;
            display: none;
            pointer-events: none;
            z-index: 200;
            font-size: 13px;
            border: 1px solid #4ecdc4;
        }}
        #stats {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(0,0,0,0.85);
            color: white;
            padding: 15px;
            border-radius: 12px;
            font-size: 12px;
        }}
        .toggle-label {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin: 8px 0;
            font-size: 13px;
        }}
        .toggle-switch {{
            position: relative;
            width: 44px;
            height: 24px;
            background: #444;
            border-radius: 12px;
            cursor: pointer;
        }}
        .toggle-switch.active {{ background: #4ecdc4; }}
        .toggle-switch::after {{
            content: '';
            position: absolute;
            top: 2px;
            left: 2px;
            width: 20px;
            height: 20px;
            background: white;
            border-radius: 50%;
            transition: 0.3s;
        }}
        .toggle-switch.active::after {{ left: 22px; }}
    </style>
</head>
<body>
    <div id="info">
        <h2>🏙️ UHI Urban Landscape</h2>
        <p><strong>Buildings:</strong> <span class="stat">{building_count}</span></p>
        <p><strong>Trees:</strong> <span class="stat">{tree_count}</span></p>
        <p><strong>Hotspots:</strong> <span class="stat">{hotspot_count}</span></p>
        <p><strong>UHI Range:</strong> <span class="stat">{uhi_min:.3f} - {uhi_max:.3f}°C</span></p>
        <hr style="border-color: #333; margin: 15px 0;">
        <p><strong>Controls:</strong></p>
        <p>🖱️ Left drag: Rotate</p>
        <p>🖱️ Right drag: Pan</p>
        <p>🔍 Scroll: Zoom</p>
    </div>
    
    <div id="controls">
        <button class="btn-vr" id="vrButton">🥽 Enter VR</button>
        <button class="btn-primary" id="resetCamera">📷 Reset View</button>
        <button class="btn-secondary" id="toggleBuildings">🏢 Toggle Buildings</button>
        <button class="btn-secondary" id="toggleTrees">🌳 Toggle Trees</button>
        <button class="btn-secondary" id="toggleHeatmap">🔥 Toggle Heatmap</button>
        <button class="btn-primary" id="toggleMitigation">🌱 Toggle Mitigation</button>
        <div class="toggle-label">
            <span>Day/Night</span>
            <div class="toggle-switch" id="dayNightToggle"></div>
        </div>
    </div>

    <div id="mitigationPanel" style="position: absolute; left: 10px; top: 350px; background: rgba(0,0,0,0.85); color: white; padding: 20px; border-radius: 12px; width: 320px; max-height: 300px; overflow-y: auto; z-index: 100; display: none; border: 1px solid #4ecdc4;">
        <h3 style="color: #ff6b6b; margin-bottom: 12px;">📋 Mitigation Strategies</h3>
        <div id="strategiesList"></div>
        <button class="btn-secondary" style="margin-top: 15px; width: 100%;" onclick="document.getElementById('mitigationPanel').style.display='none';">Close</button>
    </div>

    <div id="tempReduction" style="position: absolute; left: 50%; bottom: 240px; transform: translateX(-50%); background: rgba(0,0,0,0.85); color: white; padding: 15px; border-radius: 12px; z-index: 100; display: none; border: 1px solid #4ecdc4; font-size: 13px; white-space: nowrap;">
        <h4 style="color: #4ecdc4; margin-bottom: 8px;">✓ Mitigation Active</h4>
        <div id="tempStats"></div>
    </div>
    
    <div id="legend">
        <h3 style="margin-bottom: 12px;">Heat Intensity</h3>
        <div class="legend-item">
            <div class="legend-color" style="background: #ff0000;"></div>
            <span>Critical (>0.8)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #ff6600;"></div>
            <span>High (0.6-0.8)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #ffcc00;"></div>
            <span>Medium (0.4-0.6)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #66ff66;"></div>
            <span>Low (<0.4)</span>
        </div>
    </div>
    
    <div id="stats"></div>
    <div id="tooltip"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    
    <script>
        // Urban landscape data
        const urbanData = {urban_data_json};
        
        // Scene setup
        const scene = new THREE.Scene();
        let isNightMode = false;
        const dayBackground = new THREE.Color(0x87ceeb);
        const nightBackground = new THREE.Color(0x0a0a1a);
        scene.background = dayBackground;
        scene.fog = new THREE.Fog(0x87ceeb, 100, 500);
        
        // Camera
        const camera = new THREE.PerspectiveCamera(
            60, window.innerWidth / window.innerHeight, 0.1, 2000
        );
        camera.position.set(150, 120, 150);
        
        // Renderer
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        renderer.xr.enabled = true;
        document.body.appendChild(renderer.domElement);
        
        // Controls
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.maxPolarAngle = Math.PI / 2.1;
        
        // Groups for toggling
        const buildingGroup = new THREE.Group();
        const treeGroup = new THREE.Group();
        const hotspotGroup = new THREE.Group();
        const roadGroup = new THREE.Group();
        const vehicleGroup = new THREE.Group();
        scene.add(buildingGroup);
        scene.add(treeGroup);
        scene.add(hotspotGroup);
        scene.add(roadGroup);
        scene.add(vehicleGroup);
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
        scene.add(ambientLight);
        
        const sunLight = new THREE.DirectionalLight(0xffffff, 0.8);
        sunLight.position.set(100, 200, 100);
        sunLight.castShadow = true;
        sunLight.shadow.mapSize.width = 2048;
        sunLight.shadow.mapSize.height = 2048;
        sunLight.shadow.camera.near = 0.5;
        sunLight.shadow.camera.far = 500;
        sunLight.shadow.camera.left = -150;
        sunLight.shadow.camera.right = 150;
        sunLight.shadow.camera.top = 150;
        sunLight.shadow.camera.bottom = -150;
        scene.add(sunLight);
        
        // Hemisphere light for more natural lighting
        const hemiLight = new THREE.HemisphereLight(0x87ceeb, 0x3d5c5c, 0.3);
        scene.add(hemiLight);
        
        // Ground
        const terrainSize = urbanData.terrain_size || {{width: 200, depth: 200}};
        const groundGeometry = new THREE.PlaneGeometry(terrainSize.width, terrainSize.depth, 100, 100);
        const groundMaterial = new THREE.MeshStandardMaterial({{
            color: 0x3d5c3d,
            roughness: 0.9,
            metalness: 0.1
        }});
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.position.set(terrainSize.width/2, -0.1, terrainSize.depth/2);
        ground.receiveShadow = true;
        scene.add(ground);
        
        // Interactive objects for raycasting
        const interactiveObjects = [];
        
        // Create buildings
        function createBuildings() {{
            if (!urbanData.buildings) return;
            
            urbanData.buildings.forEach((building, idx) => {{
                const width = building.dimensions?.width || 10;
                const depth = building.dimensions?.depth || 10;
                const height = building.dimensions?.height || 20;
                
                // Building body
                const geometry = new THREE.BoxGeometry(width, height, depth);
                
                // Color based on heat exposure with smooth lerp
                const heatExposure = building.heat_exposure || 0;
                const baseColor = new THREE.Color(
                    building.color?.r || 0.7,
                    building.color?.g || 0.7,
                    building.color?.b || 0.7
                );
                const hotColor = new THREE.Color(1.0, 0.2, 0.0);
                const color = baseColor.clone().lerp(hotColor, heatExposure);

                const material = new THREE.MeshStandardMaterial({{
                    color: color,
                    emissive: color.clone(),
                    emissiveIntensity: 0.1 + heatExposure * 0.5,
                    roughness: 0.7,
                    metalness: 0.1
                }});
                
                const mesh = new THREE.Mesh(geometry, material);
                mesh.position.set(
                    building.position?.x || idx * 20,
                    height / 2,
                    building.position?.y || 0
                );
                mesh.castShadow = true;
                mesh.receiveShadow = true;
                mesh.userData = {{
                    type: 'building',
                    id: building.id,
                    buildingType: building.type,
                    height: height,
                    floors: building.floors || Math.floor(height/3),
                    heatExposure: heatExposure,
                    hasGreenRoof: building.has_green_roof
                }};
                
                buildingGroup.add(mesh);
                interactiveObjects.push(mesh);
                
                // Add windows (simplified)
                const windowDensity = building.window_density || 0.3;
                if (windowDensity > 0.2 && height > 10) {{
                    addWindowsToBuilding(mesh, width, height, depth, windowDensity);
                }}
                
                // Green roof (added to mitigation group - only visible when mitigation is ON)
                if (building.has_green_roof) {{
                    const roofGeometry = new THREE.BoxGeometry(width * 0.9, 0.5, depth * 0.9);
                    const roofMaterial = new THREE.MeshStandardMaterial({{color: 0x228b22}});
                    const roof = new THREE.Mesh(roofGeometry, roofMaterial);
                    roof.position.set(mesh.position.x, height + 0.25, mesh.position.z);
                    roof.userData = {{ type: 'greenroof', buildingId: building.id }};
                    mitigationGroup.add(roof);
                }}
            }});
        }}
        
        function addWindowsToBuilding(building, width, height, depth, density) {{
            const windowGeometry = new THREE.PlaneGeometry(1.2, 1.5);
            const windowMaterial = new THREE.MeshStandardMaterial({{
                color: 0x88ccff,
                emissive: 0x88ccff,
                emissiveIntensity: 0.2,
                metalness: 0.9,
                roughness: 0.1
            }});

            const doorGeometry = new THREE.PlaneGeometry(1.0, 2.5);
            const doorMaterial = new THREE.MeshStandardMaterial({{
                color: 0x663300,
                metalness: 0.3,
                roughness: 0.7
            }});

            const floors = Math.floor(height / 4);
            const windowsPerFloor = Math.floor(width / 3 * density);
            const windowsPerDepth = Math.floor(depth / 3 * density);

            // Windows on all 4 faces
            for (let floor = 0; floor < floors; floor++) {{
                const yOffset = floor * 4 + 2.5;

                // Front face (X-axis windows)
                for (let w = 0; w < windowsPerFloor; w++) {{
                    const windowMesh = new THREE.Mesh(windowGeometry, windowMaterial);
                    const xOffset = (w - windowsPerFloor/2) * 3;
                    windowMesh.position.set(
                        building.position.x + xOffset,
                        yOffset,
                        building.position.z + depth/2 + 0.05
                    );
                    buildingGroup.add(windowMesh);
                }}

                // Back face
                for (let w = 0; w < windowsPerFloor; w++) {{
                    const windowMesh = new THREE.Mesh(windowGeometry, windowMaterial);
                    windowMesh.rotation.y = Math.PI;
                    const xOffset = (w - windowsPerFloor/2) * 3;
                    windowMesh.position.set(
                        building.position.x + xOffset,
                        yOffset,
                        building.position.z - depth/2 - 0.05
                    );
                    buildingGroup.add(windowMesh);
                }}

                // Left face (Z-axis windows)
                for (let d = 0; d < windowsPerDepth; d++) {{
                    const windowMesh = new THREE.Mesh(windowGeometry, windowMaterial);
                    windowMesh.rotation.y = Math.PI / 2;
                    const zOffset = (d - windowsPerDepth/2) * 3;
                    windowMesh.position.set(
                        building.position.x - width/2 - 0.05,
                        yOffset,
                        building.position.z + zOffset
                    );
                    buildingGroup.add(windowMesh);
                }}

                // Right face
                for (let d = 0; d < windowsPerDepth; d++) {{
                    const windowMesh = new THREE.Mesh(windowGeometry, windowMaterial);
                    windowMesh.rotation.y = -Math.PI / 2;
                    const zOffset = (d - windowsPerDepth/2) * 3;
                    windowMesh.position.set(
                        building.position.x + width/2 + 0.05,
                        yOffset,
                        building.position.z + zOffset
                    );
                    buildingGroup.add(windowMesh);
                }}
            }}

            // Doors at ground level (front and back)
            const doorMesh1 = new THREE.Mesh(doorGeometry, doorMaterial);
            doorMesh1.position.set(
                building.position.x,
                1.2,
                building.position.z + depth/2 + 0.05
            );
            buildingGroup.add(doorMesh1);

            const doorMesh2 = new THREE.Mesh(doorGeometry, doorMaterial);
            doorMesh2.rotation.y = Math.PI;
            doorMesh2.position.set(
                building.position.x,
                1.2,
                building.position.z - depth/2 - 0.05
            );
            buildingGroup.add(doorMesh2);
        }}
        
        // Create trees
        function createTrees() {{
            if (!urbanData.trees) return;
            
            urbanData.trees.forEach((tree, idx) => {{
                const height = tree.height || 8;
                const canopyRadius = tree.canopy_radius || 3;
                const trunkRadius = tree.trunk_radius || 0.3;
                
                // Trunk
                const trunkGeometry = new THREE.CylinderGeometry(
                    trunkRadius, trunkRadius * 1.2, height * 0.4, 8
                );
                const trunkMaterial = new THREE.MeshStandardMaterial({{color: 0x4a3728}});
                const trunk = new THREE.Mesh(trunkGeometry, trunkMaterial);
                trunk.position.set(
                    tree.position?.x || idx * 10,
                    height * 0.2,
                    tree.position?.y || 0
                );
                trunk.castShadow = true;
                treeGroup.add(trunk);
                
                // Canopy
                const canopyGeometry = new THREE.SphereGeometry(canopyRadius, 8, 6);
                const canopyColor = new THREE.Color(
                    tree.color?.r || 0.2,
                    tree.color?.g || 0.6,
                    tree.color?.b || 0.2
                );
                const canopyMaterial = new THREE.MeshStandardMaterial({{
                    color: canopyColor,
                    roughness: 0.8
                }});
                const canopy = new THREE.Mesh(canopyGeometry, canopyMaterial);
                canopy.position.set(
                    trunk.position.x,
                    height * 0.6,
                    trunk.position.z
                );
                canopy.scale.y = 0.7;
                canopy.castShadow = true;
                canopy.userData = {{
                    type: 'tree',
                    id: tree.id,
                    treeType: tree.type,
                    height: height,
                    coolingEffect: tree.cooling_effect || 0.5
                }};
                treeGroup.add(canopy);
                interactiveObjects.push(canopy);
            }});
        }}
        
        // Create roads
        function createRoads() {{
            if (!urbanData.roads) return;
            
            urbanData.roads.forEach((road) => {{
                const start = new THREE.Vector3(road.start?.x || 0, 0.05, road.start?.y || 0);
                const end = new THREE.Vector3(road.end?.x || 100, 0.05, road.end?.y || 0);
                
                const direction = end.clone().sub(start);
                const length = direction.length();
                const width = road.width || 6;
                
                const roadGeometry = new THREE.PlaneGeometry(length, width);
                const roadMaterial = new THREE.MeshStandardMaterial({{
                    color: road.is_asphalt ? 0x333333 : 0x666666,
                    roughness: 0.9
                }});
                
                const roadMesh = new THREE.Mesh(roadGeometry, roadMaterial);
                roadMesh.rotation.x = -Math.PI / 2;
                
                const midpoint = start.clone().add(end).multiplyScalar(0.5);
                roadMesh.position.copy(midpoint);
                
                const angle = Math.atan2(direction.z, direction.x);
                roadMesh.rotation.z = angle;
                
                roadMesh.receiveShadow = true;
                roadGroup.add(roadMesh);
            }});
        }}

        // Create vehicles moving on roads
        function createVehicles() {{
            if (!urbanData.vehicles || !urbanData.roads) return;

            const roadMap = new Map();
            urbanData.roads.forEach(road => roadMap.set(road.id, road));

            urbanData.vehicles.forEach((vehicle) => {{
                const road = roadMap.get(vehicle.road_id);
                if (!road) return;

                // Vehicle dimensions
                const width = vehicle.width || 2.0;
                const height = vehicle.height || 1.5;
                const length = vehicle.length || 4.0;

                // Create vehicle body
                const bodyGeometry = new THREE.BoxGeometry(width, height, length);
                const bodyMaterial = new THREE.MeshStandardMaterial({{
                    color: new THREE.Color(
                        vehicle.color?.r || 0.2,
                        vehicle.color?.g || 0.2,
                        vehicle.color?.b || 0.2
                    ),
                    metalness: 0.7,
                    roughness: 0.3
                }});
                const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
                body.castShadow = true;
                body.userData = {{
                    type: 'vehicle',
                    id: vehicle.id,
                    road_id: vehicle.road_id,
                    position: vehicle.position,
                    speed: vehicle.speed,
                    roadData: road
                }};
                vehicleGroup.add(body);

                // Add windows
                const windowGeometry = new THREE.BoxGeometry(width * 0.3, height * 0.4, 0.05);
                const windowMaterial = new THREE.MeshStandardMaterial({{
                    color: 0x4488ff,
                    transparent: true,
                    opacity: 0.6,
                    metalness: 0.8
                }});

                // Front and rear windows
                for (let i = -1; i <= 1; i += 2) {{
                    const window = new THREE.Mesh(windowGeometry, windowMaterial);
                    window.position.z = i * (length / 2.5);
                    window.position.y = height * 0.1;
                    body.add(window);
                }}
            }});
        }}

        // Create mitigation visualizations (wind arrows + canopy spreads)
        const mitigationGroup = new THREE.Group();
        function createMitigation() {{
            if (!urbanData.hotspot_zones) return;

            urbanData.hotspot_zones.forEach((zone, idx) => {{
                const intensity = zone.intensity || 0.5;
                const radius = zone.radius || 20;
                const centerX = zone.center?.x || idx * 30;
                const centerY = zone.center?.y || idx * 30;

                // Canopy spread rings for trees near hotspot (shows tree cooling zones)
                if (urbanData.trees) {{
                    urbanData.trees.forEach((tree) => {{
                        const tx = tree.position?.x || 0;
                        const ty = tree.position?.y || 0;
                        const dist = Math.hypot(tx - centerX, ty - centerY);
                        if (dist < radius * 1.4 && dist > radius * 0.3) {{
                            const canopyR = tree.canopy_radius || 3;
                            const spreadGeo = new THREE.RingGeometry(canopyR, canopyR * 1.8, 32);
                            const spreadMat = new THREE.MeshBasicMaterial({{
                                color: 0x44aa44, transparent: true, opacity: 0.18,
                                side: THREE.DoubleSide
                            }});
                            const spread = new THREE.Mesh(spreadGeo, spreadMat);
                            spread.rotation.x = -Math.PI / 2;
                            spread.position.set(tx, 0.15, ty);
                            mitigationGroup.add(spread);

                            // Elevated shade disk
                            const shadeDisk = new THREE.CircleGeometry(canopyR * 1.4, 32);
                            const shadeMat = new THREE.MeshBasicMaterial({{
                                color: 0x228833, transparent: true, opacity: 0.12
                            }});
                            const shade = new THREE.Mesh(shadeDisk, shadeMat);
                            shade.rotation.x = -Math.PI / 2;
                            shade.position.set(tx, tree.height || 8, ty);
                            mitigationGroup.add(shade);
                        }}
                    }});
                }}
            }});

            scene.add(mitigationGroup);
            mitigationGroup.visible = false;
        }}

        function toggleMitigation() {{
            const btn = document.getElementById('toggleMitigation');
            mitigationGroup.visible = !mitigationGroup.visible;

            if (mitigationGroup.visible) {{
                // MITIGATION ON: Reduce heat, cool colors, shrink hotspots
                btn.classList.add('active');

                // Cool down buildings (maintain definition)
                buildingGroup.children.forEach(b => {{
                    if (b.material.emissiveIntensity) {{
                        b.userData.originalIntensity = b.material.emissiveIntensity;
                        b.userData.originalColor = b.material.color.clone();
                        b.material.emissiveIntensity -= 0.15;
                        // Slight color shift to maintain building definition
                        b.material.color.lerp(new THREE.Color(0x6ebdd4), 0.08);
                    }}
                }});

                // Cool down hotspots: reduce radius, glow, and change to cool colors
                hotspotHalos.forEach(halo => {{
                    if (halo.userData.phase !== undefined && halo.material) {{
                        // Scale down hotspot radius to 50%
                        halo.userData.originalScale = halo.scale.x || 1.0;
                        halo.scale.set(0.5, 0.5, 0.5);

                        halo.userData.originalOpacity = halo.material.opacity;
                        // Reduce opacity by 60% for cooling effect
                        halo.material.opacity *= 0.4;
                        // Change color to cool blue-cyan
                        if (halo.material.color) {{
                            halo.userData.originalColor = halo.material.color.clone();
                            halo.material.color.copy(new THREE.Color(0x00ccff));
                        }}
                    }}
                }});

                // Reduce point light intensity
                hotspotGroup.children.forEach(child => {{
                    if (child.isLight) {{
                        child.userData.originalIntensity = child.intensity;
                        child.intensity *= 0.3;
                        child.color.set(0x00ccff); // Cool cyan light
                    }}
                }});
            }} else {{
                // MITIGATION OFF: Restore heat visualization
                btn.classList.remove('active');

                buildingGroup.children.forEach(b => {{
                    if (b.userData.originalIntensity) {{
                        b.material.emissiveIntensity = b.userData.originalIntensity;
                        if (b.userData.originalColor) {{
                            b.material.color.copy(b.userData.originalColor);
                        }}
                    }}
                }});

                // Restore hotspot sizes, opacity, and colors
                hotspotHalos.forEach(halo => {{
                    // Restore original radius
                    halo.scale.set(halo.userData.originalScale, halo.userData.originalScale, halo.userData.originalScale);

                    if (halo.userData.originalOpacity) {{
                        halo.material.opacity = halo.userData.originalOpacity;
                        if (halo.userData.originalColor) {{
                            halo.material.color.copy(halo.userData.originalColor);
                        }}
                    }}
                }});

                hotspotGroup.children.forEach(child => {{
                    if (child.isLight && child.userData.originalIntensity) {{
                        child.intensity = child.userData.originalIntensity;
                        child.color.set(0xff4400); // Restore hot orange
                    }}
                }});
            }}
        }}

        // Track hotspot halos and rings for scaling on mitigation toggle
        const hotspotHalos = [];

        // Create hotspot zones with ground halos and pulsing glow rings
        function createHotspots() {{
            if (!urbanData.hotspot_zones) return;

            urbanData.hotspot_zones.forEach((zone, idx) => {{
                const intensity = zone.intensity || 0.5;
                const radius = zone.radius || 20;
                const centerX = zone.center?.x || idx * 30;
                const centerY = zone.center?.y || idx * 30;

                const color = new THREE.Color(1.0, 0.267, 0.0);

                // Ground-level heat halo (filled circle on ground)
                const haloGeometry = new THREE.CircleGeometry(radius, 64);
                const haloMaterial = new THREE.MeshBasicMaterial({{
                    color: color,
                    transparent: true,
                    opacity: 0.4,
                    side: THREE.DoubleSide
                }});
                const halo = new THREE.Mesh(haloGeometry, haloMaterial);
                halo.rotation.x = -Math.PI / 2;
                halo.position.set(centerX, 0.05, centerY);
                halo.userData = {{ type: 'hotspot', id: zone.id, intensity, phase: idx * 0.78, originalScale: 1.0 }};
                hotspotGroup.add(halo);
                hotspotHalos.push(halo);
                interactiveObjects.push(halo);

                // Pulsing glow ring (animated opacity)
                const ringGeometry = new THREE.RingGeometry(radius * 0.8, radius, 32);
                const ringMaterial = new THREE.MeshBasicMaterial({{
                    color: color,
                    transparent: true,
                    opacity: 0.3,
                    side: THREE.DoubleSide
                }});
                const ring = new THREE.Mesh(ringGeometry, ringMaterial);
                ring.rotation.x = -Math.PI / 2;
                ring.position.set(centerX, 0.06, centerY);
                ring.userData = {{ phase: idx * 0.78, originalScale: 1.0 }};
                hotspotGroup.add(ring);
                hotspotHalos.push(ring);

                // Point light at hotspot center for ambient glow
                const light = new THREE.PointLight(0xff4400, intensity * 2, radius * 1.5);
                light.position.set(centerX, 3, centerY);
                hotspotGroup.add(light);
            }});
        }}
        
        // Create street lights group
        const streetLightsGroup = new THREE.Group();
        scene.add(streetLightsGroup);

        function createStreetLights() {{
            if (!urbanData.roads) return;

            const LIGHT_SPACING = 25; // Uniform spacing: one light every 25 units
            const SIDE_OFFSET = 4; // Distance from center of road to lamp

            urbanData.roads.forEach((road) => {{
                const start = new THREE.Vector3(road.start?.x || 0, 0, road.start?.y || 0);
                const end = new THREE.Vector3(road.end?.x || 100, 0, road.end?.y || 0);

                const direction = end.clone().sub(start).normalize();
                const length = start.distanceTo(end);

                // Perpendicular direction (rotate 90 degrees)
                const perpendicular = new THREE.Vector3(-direction.z, 0, direction.x).normalize();

                const numLights = Math.max(2, Math.ceil(length / LIGHT_SPACING)); // Uniform interval

                for (let i = 0; i < numLights; i++) {{
                    const t = i / Math.max(1, numLights - 1);
                    const centerPos = start.clone().lerp(end, t);

                    // Place lights on alternating sides of road
                    const side = (i % 2 === 0) ? 1 : -1;
                    const offset = perpendicular.clone().multiplyScalar(side * SIDE_OFFSET);
                    const pos = centerPos.clone().add(offset);

                    // Light pole (post)
                    const poleGeometry = new THREE.CylinderGeometry(0.3, 0.35, 8, 8);
                    const poleMaterial = new THREE.MeshStandardMaterial({{
                        color: 0x333333,
                        metalness: 0.6,
                        roughness: 0.4
                    }});
                    const pole = new THREE.Mesh(poleGeometry, poleMaterial);
                    pole.position.set(pos.x, 4, pos.z);
                    streetLightsGroup.add(pole);

                    // Lamp head
                    const lampGeometry = new THREE.SphereGeometry(0.8, 16, 16);
                    const lampMaterial = new THREE.MeshStandardMaterial({{
                        color: 0xffff99,
                        emissive: 0xffff99,
                        emissiveIntensity: 0.3
                    }});
                    const lamp = new THREE.Mesh(lampGeometry, lampMaterial);
                    lamp.position.set(pos.x, 8, pos.z);
                    streetLightsGroup.add(lamp);

                    // Point light (illuminates at night)
                    const pointLight = new THREE.PointLight(0xffff99, 1.5, 40);
                    pointLight.position.set(pos.x, 7.5, pos.z);
                    streetLightsGroup.add(pointLight);
                }}
            }});

            streetLightsGroup.visible = false; // Hidden during day
        }}

        // Initialize scene
        createRoads();
        createBuildings();
        createTrees();
        createVehicles();
        createHotspots();
        createMitigation();
        createStreetLights();
        
        // Raycaster for interaction
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        const tooltip = document.getElementById('tooltip');
        
        function onMouseMove(event) {{
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
            
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObjects(interactiveObjects);
            
            if (intersects.length > 0) {{
                const obj = intersects[0].object;
                const data = obj.userData;
                
                tooltip.style.display = 'block';
                tooltip.style.left = event.clientX + 15 + 'px';
                tooltip.style.top = event.clientY + 15 + 'px';
                
                if (data.type === 'building') {{
                    tooltip.innerHTML = `
                        <strong>🏢 ${{data.buildingType || 'Building'}}</strong><br>
                        ID: ${{data.id}}<br>
                        Height: ${{data.height?.toFixed(1)}}m (${{data.floors}} floors)<br>
                        Heat Exposure: ${{(data.heatExposure * 100).toFixed(1)}}%<br>
                        ${{data.hasGreenRoof ? '🌿 Green Roof' : ''}}
                    `;
                }} else if (data.type === 'tree') {{
                    tooltip.innerHTML = `
                        <strong>🌳 ${{data.treeType || 'Tree'}}</strong><br>
                        ID: ${{data.id}}<br>
                        Height: ${{data.height?.toFixed(1)}}m<br>
                        Cooling Effect: ${{(data.coolingEffect * 100).toFixed(0)}}%
                    `;
                }}
            }} else {{
                tooltip.style.display = 'none';
            }}
        }}
        
        window.addEventListener('mousemove', onMouseMove);

        // Click handler for building mitigation panel
        window.addEventListener('click', (event) => {{
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObjects(interactiveObjects);

            if (intersects.length > 0) {{
                const obj = intersects[0].object;
                const data = obj.userData;

                if (data.type === 'building') {{
                    showMitigationPanel(data);
                }}
            }}
        }});

        function showMitigationPanel(buildingData) {{
            const strategies = [
                {{ name: 'Street Tree Planting', cooling: 2.5, cost: 45, timeline: 6 }},
                {{ name: 'Urban Parks & Green Spaces', cooling: 3.5, cost: 85, timeline: 18 }},
                {{ name: 'Green Roofs (Extensive)', cooling: 1.8, cost: 120, timeline: 4 }},
                {{ name: 'Cool Pavements', cooling: 1.5, cost: 35, timeline: 3 }}
            ];

            let html = `<div style="margin-bottom: 15px; padding-bottom: 12px; border-bottom: 1px solid #333;">
                <strong style="color: #4ecdc4;">Building Info</strong><br>
                <span style="font-size: 12px; color: #aaa;">Heat Exposure: ${{(buildingData.heatExposure * 100).toFixed(1)}}%</span><br>
                <span style="font-size: 12px; color: #aaa;">Est. Temp: ${{(28 + buildingData.heatExposure * 10).toFixed(1)}}°C</span>
            </div>`;

            html += '<strong style="color: #4ecdc4; display: block; margin-bottom: 10px;">Recommended Strategies</strong>';
            strategies.forEach(s => {{
                html += `<div style="background: rgba(78, 205, 196, 0.1); padding: 10px; margin: 8px 0; border-left: 3px solid #4ecdc4; border-radius: 4px; font-size: 12px;">
                    <div style="font-weight: bold; color: #4ecdc4;">${{s.name}}</div>
                    <div style="color: #ff6b6b; margin-top: 4px;">Cooling: −${{s.cooling}}°C</div>
                    <div style="color: #aaa; font-size: 11px;">Cost: $${{s.cost}}/m² | Timeline: ${{s.timeline}} months</div>
                </div>`;
            }});

            document.getElementById('strategiesList').innerHTML = html;
            document.getElementById('mitigationPanel').style.display = 'block';
        }}

        // UI Controls
        document.getElementById('resetCamera').onclick = () => {{
            camera.position.set(150, 120, 150);
            controls.target.set(terrainSize.width/2, 0, terrainSize.depth/2);
        }};
        
        document.getElementById('toggleBuildings').onclick = () => {{
            buildingGroup.visible = !buildingGroup.visible;
        }};
        
        document.getElementById('toggleTrees').onclick = () => {{
            treeGroup.visible = !treeGroup.visible;
        }};
        
        document.getElementById('toggleHeatmap').onclick = () => {{
            hotspotGroup.visible = !hotspotGroup.visible;
        }};

        document.getElementById('toggleMitigation').onclick = () => {{
            toggleMitigation();
            document.getElementById('tempReduction').style.display = mitigationGroup.visible ? 'block' : 'none';
            if (mitigationGroup.visible) {{
                updateTempStats();
            }}
        }};

        function updateTempStats() {{
            const strategies = [
                {{ name: 'Street Tree Planting', cooling: 2.5 }},
                {{ name: 'Urban Parks', cooling: 3.5 }},
                {{ name: 'Green Roofs', cooling: 1.8 }},
                {{ name: 'Cool Pavements', cooling: 1.5 }}
            ];
            const totalCooling = strategies.reduce((sum, s) => sum + s.cooling, 0);
            let html = '';
            strategies.forEach(s => {{
                html += `<div style="display: flex; justify-content: space-between; margin: 6px 0;"><span>${{s.name}}:</span><span style="color: #ff6b6b;">−${{s.cooling}}°C</span></div>`;
            }});
            html += `<div style="border-top: 1px solid #333; margin: 8px 0; padding-top: 8px; font-weight: bold;">Estimated Total: −${{totalCooling.toFixed(1)}}°C</div>`;
            document.getElementById('tempStats').innerHTML = html;
        }}

        document.getElementById('dayNightToggle').onclick = function() {{
            this.classList.toggle('active');
            isNightMode = !isNightMode;
            
            if (isNightMode) {{
                scene.background = nightBackground;
                scene.fog.color = nightBackground;
                sunLight.intensity = 0.2;
                ambientLight.intensity = 0.15;

                // Show street lights at night
                streetLightsGroup.visible = true;

                // Make windows glow at night
                buildingGroup.traverse((child) => {{
                    if (child.material && child.material.emissive) {{
                        child.material.emissiveIntensity = 0.5;
                    }}
                }});
            }} else {{
                scene.background = dayBackground;
                scene.fog.color = dayBackground;
                sunLight.intensity = 0.8;
                ambientLight.intensity = 0.4;

                // Hide street lights during day
                streetLightsGroup.visible = false;

                buildingGroup.traverse((child) => {{
                    if (child.material && child.material.emissive) {{
                        child.material.emissiveIntensity = 0.1;
                    }}
                }});
            }}
        }};
        
        // VR Button
        document.getElementById('vrButton').onclick = () => {{
            if (navigator.xr) {{
                navigator.xr.isSessionSupported('immersive-vr').then((supported) => {{
                    if (supported) {{
                        renderer.xr.enabled = true;
                        document.body.appendChild(document.createElement('button'));
                    }} else {{
                        alert('VR not supported on this device');
                    }}
                }});
            }} else {{
                alert('WebXR not available');
            }}
        }};
        
        // Window resize
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
        
        // Stats display
        const statsDiv = document.getElementById('stats');
        
        // Animation loop
        let lastTime = 0;
        function animate(time) {{
            requestAnimationFrame(animate);
            controls.update();

            // Animate hotspot glow rings and halos
            const t = time * 0.003;
            hotspotGroup.children.forEach((child) => {{
                if (child.userData.phase !== undefined && child.material.opacity !== undefined) {{
                    // Pulsing glow effect: 0.2 to 0.45 opacity
                    child.material.opacity = 0.2 + 0.25 * Math.sin(t + child.userData.phase);
                }}
            }});

            // Animate vehicles moving along roads
            vehicleGroup.children.forEach((vehicle) => {{
                if (vehicle.userData.type === 'vehicle' && vehicle.userData.roadData) {{
                    const road = vehicle.userData.roadData;
                    const speed = vehicle.userData.speed || 20;
                    const roadLength = Math.hypot(
                        (road.end?.x || 0) - (road.start?.x || 0),
                        (road.end?.y || 0) - (road.start?.y || 0)
                    );

                    // Update position along road (0-1)
                    vehicle.userData.position += (speed / roadLength) * (time - lastTime) / 1000;
                    if (vehicle.userData.position > 1) {{
                        vehicle.userData.position = 0; // Loop back
                    }}

                    // Interpolate position on road
                    const t = vehicle.userData.position;
                    const startX = road.start?.x || 0;
                    const startY = road.start?.y || 0;
                    const endX = road.end?.x || 100;
                    const endY = road.end?.y || 0;

                    vehicle.position.x = startX + (endX - startX) * t;
                    vehicle.position.z = startY + (endY - startY) * t;
                    vehicle.position.y = 0.75; // Height above ground

                    // Orient vehicle along road direction
                    const direction = new THREE.Vector3(endX - startX, 0, endY - startY).normalize();
                    vehicle.rotation.y = Math.atan2(direction.x, direction.z);
                }}
            }});

            // Update stats
            if (time - lastTime > 500) {{
                const fps = Math.round(1000 / (time - lastTime) * 2);
                statsDiv.innerHTML = `FPS: ~${{fps}}<br>Objects: ${{scene.children.length}}`;
                lastTime = time;
            }}

            renderer.render(scene, camera);
        }}
        
        animate(0);
    </script>
</body>
</html>'''
    
    # Original simple template
    HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UHI Hotspot Visualization - Three.js</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ overflow: hidden; font-family: Arial, sans-serif; }}
        #info {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 15px;
            border-radius: 8px;
            z-index: 100;
            max-width: 300px;
        }}
        #info h2 {{ margin-bottom: 10px; color: #ff6b6b; }}
        #info p {{ margin: 5px 0; font-size: 14px; }}
        #legend {{
            position: absolute;
            bottom: 20px;
            right: 20px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 15px;
            border-radius: 8px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            margin-right: 10px;
            border-radius: 3px;
        }}
        #tooltip {{
            position: absolute;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 10px;
            border-radius: 5px;
            display: none;
            pointer-events: none;
            z-index: 200;
        }}
    </style>
</head>
<body>
    <div id="info">
        <h2>🌡️ UHI Hotspot Map</h2>
        <p><strong>Points:</strong> {point_count}</p>
        <p><strong>UHI Range:</strong> {uhi_min:.3f} - {uhi_max:.3f}°C</p>
        <p><strong>Controls:</strong></p>
        <p>• Left click + drag: Rotate</p>
        <p>• Right click + drag: Pan</p>
        <p>• Scroll: Zoom</p>
    </div>
    
    <div id="legend">
        <h3 style="margin-bottom: 10px;">Heat Intensity</h3>
        <div class="legend-item">
            <div class="legend-color" style="background: #ff0000;"></div>
            <span>High (Critical)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #ffaa00;"></div>
            <span>Medium-High</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #ffff00;"></div>
            <span>Medium</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #00ff00;"></div>
            <span>Low</span>
        </div>
    </div>
    
    <div id="tooltip"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    
    <script>
        // Hotspot data
        const hotspotData = {hotspot_json};
        
        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a2e);
        
        // Camera
        const camera = new THREE.PerspectiveCamera(
            60, window.innerWidth / window.innerHeight, 0.1, 2000
        );
        camera.position.set(150, 150, 150);
        
        // Renderer
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.shadowMap.enabled = true;
        document.body.appendChild(renderer.domElement);
        
        // Controls
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
        scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(100, 200, 100);
        directionalLight.castShadow = true;
        scene.add(directionalLight);
        
        // Ground plane (terrain)
        const groundGeometry = new THREE.PlaneGeometry(200, 200, 50, 50);
        const groundMaterial = new THREE.MeshStandardMaterial({{
            color: 0x2d3436,
            roughness: 0.8,
            metalness: 0.2,
            wireframe: false
        }});
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.position.set(50, 0, 50);
        ground.receiveShadow = true;
        scene.add(ground);
        
        // Grid helper
        const gridHelper = new THREE.GridHelper(200, 20, 0x444444, 0x333333);
        gridHelper.position.set(50, 0.1, 50);
        scene.add(gridHelper);
        
        // Create hotspot pillars
        const pillars = [];
        hotspotData.forEach((point, index) => {{
            // Pillar geometry
            const height = Math.max(point.position.z, 2);
            const geometry = new THREE.CylinderGeometry(1.5, 2, height, 16);
            
            // Color based on intensity
            const color = new THREE.Color(
                point.color.r,
                point.color.g,
                point.color.b
            );
            
            const material = new THREE.MeshStandardMaterial({{
                color: color,
                emissive: color,
                emissiveIntensity: 0.3,
                roughness: 0.4,
                metalness: 0.6
            }});
            
            const pillar = new THREE.Mesh(geometry, material);
            pillar.position.set(
                point.position.x,
                height / 2,
                point.position.y
            );
            pillar.castShadow = true;
            pillar.userData = {{
                uhi: point.uhi_value,
                intensity: point.intensity,
                index: index
            }};
            
            scene.add(pillar);
            pillars.push(pillar);
            
            // Add glow effect for high intensity
            if (point.intensity > 0.7) {{
                const glowGeometry = new THREE.CylinderGeometry(2.5, 3, height * 0.8, 16);
                const glowMaterial = new THREE.MeshBasicMaterial({{
                    color: color,
                    transparent: true,
                    opacity: 0.2
                }});
                const glow = new THREE.Mesh(glowGeometry, glowMaterial);
                glow.position.copy(pillar.position);
                scene.add(glow);
            }}
        }});
        
        // Raycaster for interaction
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        const tooltip = document.getElementById('tooltip');
        
        function onMouseMove(event) {{
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
            
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObjects(pillars);
            
            if (intersects.length > 0) {{
                const obj = intersects[0].object;
                tooltip.style.display = 'block';
                tooltip.style.left = event.clientX + 15 + 'px';
                tooltip.style.top = event.clientY + 15 + 'px';
                tooltip.innerHTML = `
                    <strong>Hotspot #${{obj.userData.index + 1}}</strong><br>
                    UHI Value: ${{obj.userData.uhi.toFixed(4)}}°C<br>
                    Intensity: ${{(obj.userData.intensity * 100).toFixed(1)}}%
                `;
            }} else {{
                tooltip.style.display = 'none';
            }}
        }}
        
        window.addEventListener('mousemove', onMouseMove);
        
        // Handle window resize
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
        
        // Animation loop
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            
            // Subtle animation for pillars
            const time = Date.now() * 0.001;
            pillars.forEach((pillar, i) => {{
                if (pillar.userData.intensity > 0.6) {{
                    pillar.material.emissiveIntensity = 0.3 + Math.sin(time * 2 + i) * 0.1;
                }}
            }});
            
            renderer.render(scene, camera);
        }}
        
        animate();
    </script>
</body>
</html>'''

    def generate(self, df: pd.DataFrame = None,
                 output_path: str = 'uhi_visualization.html',
                 **kwargs) -> str:
        """
        Generate Three.js HTML visualization.
        
        Args:
            df: DataFrame with hotspot data
            output_path: Output file path
            **kwargs: Additional arguments for prepare_data
            
        Returns:
            Path to generated HTML file
        """
        try:
            # Prepare data
            if df is not None:
                self.prepare_data(df, **kwargs)
            elif not self.hotspot_points:
                self.hotspot_points = self._get_default_points()
                self.metadata = {'point_count': 3, 'uhi_min': 0.15, 'uhi_max': 0.2}
            
            # Convert points to JSON
            hotspot_json = json.dumps([p.to_dict() for p in self.hotspot_points])
            
            # Generate HTML
            html_content = self.HTML_TEMPLATE.format(
                hotspot_json=hotspot_json,
                point_count=self.metadata.get('point_count', len(self.hotspot_points)),
                uhi_min=self.metadata.get('uhi_min', 0),
                uhi_max=self.metadata.get('uhi_max', 1)
            )
            
            # Write to file
            with open(output_path, 'w') as f:
                f.write(html_content)
                
            logger.info(f"Generated Three.js visualization: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating Three.js visualization: {e}")
            return ""
    
    def generate_enhanced(self, df: pd.DataFrame = None,
                          hotspot_df: pd.DataFrame = None,
                          output_path: str = 'uhi_urban_landscape.html',
                          terrain_size: Tuple[float, float] = (200, 200),
                          **kwargs) -> str:
        """
        Generate enhanced Three.js visualization with buildings, trees, and roads.
        
        Args:
            df: DataFrame with urban parameters (bldDensity, treeDensity, etc.)
            hotspot_df: DataFrame with hotspot data (UHI values, coordinates)
            output_path: Output file path
            terrain_size: Size of terrain (width, depth)
            **kwargs: Additional arguments for urban generator
            
        Returns:
            Path to generated HTML file
        """
        try:
            # Generate urban landscape
            if URBAN_GENERATOR_AVAILABLE:
                generator = UrbanLandscapeGenerator(seed=kwargs.get('seed', 42))
                
                # Use hotspot_df if provided, otherwise use df
                source_df = df if df is not None else hotspot_df
                hotspot_source = hotspot_df if hotspot_df is not None else df
                
                landscape = generator.generate_from_dataframe(
                    source_df,
                    hotspot_df=hotspot_source,
                    terrain_size=terrain_size,
                    scale_factor=kwargs.get('scale_factor', 1.0)
                )
                
                urban_data = landscape.to_dict()
            else:
                # Fallback: generate basic data structure
                urban_data = self._generate_basic_urban_data(df, hotspot_df, terrain_size)
            
            # Get statistics
            building_count = len(urban_data.get('buildings', []))
            tree_count = len(urban_data.get('trees', []))
            hotspot_count = len(urban_data.get('hotspot_zones', []))
            
            # Get UHI range
            uhi_values = [z.get('uhi_value', 0) for z in urban_data.get('hotspot_zones', [])]
            uhi_min = min(uhi_values) if uhi_values else 0
            uhi_max = max(uhi_values) if uhi_values else 1
            
            # Generate HTML (use default=str to handle bool/numpy types)
            html_content = self.HTML_TEMPLATE_ENHANCED.format(
                urban_data_json=json.dumps(urban_data, default=str),
                building_count=building_count,
                tree_count=tree_count,
                hotspot_count=hotspot_count,
                uhi_min=uhi_min,
                uhi_max=uhi_max
            )
            
            # Write to file
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Generated enhanced Three.js visualization: {output_path}")
            logger.info(f"  Buildings: {building_count}, Trees: {tree_count}, Hotspots: {hotspot_count}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating enhanced visualization: {e}")
            return self.generate(df, output_path, **kwargs)  # Fallback to simple
    
    def _generate_basic_urban_data(self, df: pd.DataFrame, 
                                    hotspot_df: pd.DataFrame,
                                    terrain_size: Tuple[float, float]) -> Dict:
        """Generate basic urban data when urban_generator is not available."""
        import random
        random.seed(42)
        
        urban_data = {
            'buildings': [],
            'trees': [],
            'roads': [],
            'hotspot_zones': [],
            'terrain_size': {'width': terrain_size[0], 'depth': terrain_size[1]}
        }
        
        # Generate some buildings
        n_buildings = 30
        for i in range(n_buildings):
            urban_data['buildings'].append({
                'id': f'building_{i:04d}',
                'position': {'x': 20 + (i % 6) * 30, 'y': 20 + (i // 6) * 35},
                'dimensions': {
                    'width': 10 + random.random() * 10,
                    'depth': 10 + random.random() * 10,
                    'height': 15 + random.random() * 40
                },
                'type': random.choice(['residential_low', 'residential_mid', 'commercial']),
                'heat_exposure': random.random() * 0.5,
                'color': {'r': 0.7, 'g': 0.7, 'b': 0.7},
                'has_green_roof': random.random() < 0.1,
                'window_density': 0.3
            })
        
        # Generate trees
        n_trees = 40
        for i in range(n_trees):
            urban_data['trees'].append({
                'id': f'tree_{i:04d}',
                'position': {
                    'x': random.uniform(5, terrain_size[0] - 5),
                    'y': random.uniform(5, terrain_size[1] - 5)
                },
                'height': 5 + random.random() * 10,
                'canopy_radius': 2 + random.random() * 4,
                'trunk_radius': 0.3,
                'type': 'tree_deciduous',
                'color': {'r': 0.2, 'g': 0.5 + random.random() * 0.2, 'b': 0.2},
                'cooling_effect': 0.3 + random.random() * 0.4
            })
        
        # Generate hotspot zones from hotspot data
        if hotspot_df is not None and not hotspot_df.empty:
            uhi_col = 'UHI_d' if 'UHI_d' in hotspot_df.columns else None
            if uhi_col is None:
                for col in hotspot_df.columns:
                    if 'UHI' in col.upper():
                        uhi_col = col
                        break
            
            if uhi_col:
                uhi_min = hotspot_df[uhi_col].min()
                uhi_max = hotspot_df[uhi_col].max()
                uhi_range = max(uhi_max - uhi_min, 0.001)
                
                for idx, row in hotspot_df.iterrows():
                    uhi_val = row[uhi_col]
                    intensity = (uhi_val - uhi_min) / uhi_range
                    
                    # Color based on intensity
                    if intensity > 0.75:
                        color = {'r': 1, 'g': 0, 'b': 0}
                    elif intensity > 0.5:
                        color = {'r': 1, 'g': 0.5, 'b': 0}
                    elif intensity > 0.25:
                        color = {'r': 1, 'g': 1, 'b': 0}
                    else:
                        color = {'r': 0, 'g': 1, 'b': 0}
                    
                    urban_data['hotspot_zones'].append({
                        'id': f'hotspot_{idx:04d}',
                        'center': {
                            'x': (idx % 10) * (terrain_size[0] / 10) + 10,
                            'y': (idx // 10) * (terrain_size[1] / 10) + 10
                        },
                        'radius': 10 + intensity * 25,
                        'intensity': intensity,
                        'uhi_value': uhi_val,
                        'color': color
                    })
        
        return urban_data


class UnityExporter(BaseVisualizationGenerator):
    """
    Exports hotspot data in Unity-compatible JSON format.
    
    Generates a JSON file that can be imported into Unity for
    3D visualization and game engine integration.
    """
    
    def generate(self, df: pd.DataFrame = None,
                 output_path: str = 'uhi_unity_data.json',
                 **kwargs) -> str:
        """
        Generate Unity-compatible JSON export.
        
        Args:
            df: DataFrame with hotspot data
            output_path: Output file path
            **kwargs: Additional arguments for prepare_data
            
        Returns:
            Path to generated JSON file
        """
        try:
            # Prepare data
            if df is not None:
                self.prepare_data(df, **kwargs)
            elif not self.hotspot_points:
                self.hotspot_points = self._get_default_points()
            
            # Build Unity-compatible structure
            unity_data = {
                'version': '1.0',
                'type': 'UHI_HotspotData',
                'metadata': {
                    'point_count': len(self.hotspot_points),
                    'coordinate_system': 'Unity_LeftHanded',
                    'units': 'meters',
                    'uhi_range': {
                        'min': self.metadata.get('uhi_min', 0),
                        'max': self.metadata.get('uhi_max', 1)
                    }
                },
                'terrain': {
                    'width': 200,
                    'length': 200,
                    'subdivisions': 50
                },
                'hotspots': [],
                'prefab_settings': {
                    'pillar_prefab': 'Prefabs/HotspotPillar',
                    'base_radius': 1.5,
                    'max_height': 50,
                    'material': 'Materials/HotspotMaterial'
                }
            }
            
            # Add hotspot points
            for i, point in enumerate(self.hotspot_points):
                hotspot = {
                    'id': f'hotspot_{i:04d}',
                    'position': {
                        'x': point.x,
                        'y': 0,  # Ground level
                        'z': point.y  # Unity uses Y-up, Z-forward
                    },
                    'height': point.z,
                    'uhi_value': point.uhi_value,
                    'intensity': point.intensity,
                    'color': {
                        'r': point.color[0],
                        'g': point.color[1],
                        'b': point.color[2],
                        'a': 1.0
                    },
                    'metadata': {
                        'severity': self._get_severity(point.intensity),
                        'requires_attention': point.intensity > 0.7
                    }
                }
                unity_data['hotspots'].append(hotspot)
            
            # Add Unity C# script template
            unity_data['import_script'] = self._get_unity_script()
            
            # Write to file
            with open(output_path, 'w') as f:
                json.dump(unity_data, f, indent=2)
                
            logger.info(f"Generated Unity export: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating Unity export: {e}")
            return ""
    
    def _get_severity(self, intensity: float) -> str:
        """Get severity label from intensity."""
        if intensity > 0.8:
            return 'critical'
        elif intensity > 0.6:
            return 'high'
        elif intensity > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _get_unity_script(self) -> str:
        """Return Unity C# script template for importing data."""
        return '''
// UHIDataImporter.cs - Place in Assets/Scripts/
using UnityEngine;
using System.Collections.Generic;

[System.Serializable]
public class UHIHotspotData
{
    public string version;
    public Metadata metadata;
    public List<Hotspot> hotspots;
    
    [System.Serializable]
    public class Metadata
    {
        public int point_count;
        public string coordinate_system;
    }
    
    [System.Serializable]
    public class Hotspot
    {
        public string id;
        public Position position;
        public float height;
        public float uhi_value;
        public float intensity;
        public HotspotColor color;
    }
    
    [System.Serializable]
    public class Position
    {
        public float x, y, z;
    }
    
    [System.Serializable]
    public class HotspotColor
    {
        public float r, g, b, a;
    }
}

public class UHIDataImporter : MonoBehaviour
{
    public TextAsset jsonFile;
    public GameObject hotspotPrefab;
    
    void Start()
    {
        if (jsonFile != null)
        {
            ImportData();
        }
    }
    
    void ImportData()
    {
        UHIHotspotData data = JsonUtility.FromJson<UHIHotspotData>(jsonFile.text);
        
        foreach (var hotspot in data.hotspots)
        {
            Vector3 pos = new Vector3(hotspot.position.x, 0, hotspot.position.z);
            GameObject obj = Instantiate(hotspotPrefab, pos, Quaternion.identity);
            obj.transform.localScale = new Vector3(1.5f, hotspot.height, 1.5f);
            
            Renderer rend = obj.GetComponent<Renderer>();
            if (rend != null)
            {
                rend.material.color = new Color(
                    hotspot.color.r,
                    hotspot.color.g,
                    hotspot.color.b
                );
            }
        }
    }
}
'''


class BlenderScriptGenerator(BaseVisualizationGenerator):
    """
    Generates Blender Python script for automated 3D scene creation.
    
    Creates a Python script that can be run in Blender to generate
    a complete 3D visualization of UHI hotspots.
    """
    
    SCRIPT_TEMPLATE = '''# Blender Python Script for UHI Hotspot Visualization
# Run this script in Blender (Text Editor > Run Script)
# Requires Blender 2.8+

import bpy
import bmesh
import math
from mathutils import Vector

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Hotspot data
hotspot_data = {hotspot_data}

# Scene settings
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 128

# Create terrain plane
bpy.ops.mesh.primitive_plane_add(size=200, location=(50, 50, 0))
terrain = bpy.context.active_object
terrain.name = "UHI_Terrain"

# Create terrain material
terrain_mat = bpy.data.materials.new(name="TerrainMaterial")
terrain_mat.use_nodes = True
terrain_bsdf = terrain_mat.node_tree.nodes["Principled BSDF"]
terrain_bsdf.inputs["Base Color"].default_value = (0.15, 0.15, 0.18, 1)
terrain_bsdf.inputs["Roughness"].default_value = 0.8
terrain.data.materials.append(terrain_mat)

# Subdivide terrain for detail
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.subdivide(number_cuts=20)
bpy.ops.object.mode_set(mode='OBJECT')

# Create hotspot pillar material
def create_hotspot_material(name, color, emission_strength):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear default nodes
    for node in nodes:
        nodes.remove(node)
    
    # Add nodes
    output = nodes.new('ShaderNodeOutputMaterial')
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    emission = nodes.new('ShaderNodeEmission')
    mix = nodes.new('ShaderNodeMixShader')
    
    # Set node positions
    output.location = (400, 0)
    mix.location = (200, 0)
    principled.location = (0, 100)
    emission.location = (0, -100)
    
    # Configure nodes
    principled.inputs["Base Color"].default_value = (*color, 1)
    principled.inputs["Roughness"].default_value = 0.4
    principled.inputs["Metallic"].default_value = 0.6
    
    emission.inputs["Color"].default_value = (*color, 1)
    emission.inputs["Strength"].default_value = emission_strength
    
    mix.inputs["Fac"].default_value = 0.3
    
    # Link nodes
    links.new(principled.outputs["BSDF"], mix.inputs[1])
    links.new(emission.outputs["Emission"], mix.inputs[2])
    links.new(mix.outputs["Shader"], output.inputs["Surface"])
    
    return mat

# Create hotspot pillars
pillar_collection = bpy.data.collections.new("UHI_Hotspots")
bpy.context.scene.collection.children.link(pillar_collection)

for i, point in enumerate(hotspot_data):
    pos = point['position']
    color = (point['color']['r'], point['color']['g'], point['color']['b'])
    height = max(point['position']['z'], 1)
    intensity = point['intensity']
    
    # Create cylinder
    bpy.ops.mesh.primitive_cylinder_add(
        radius=1.5,
        depth=height,
        location=(pos['x'], pos['y'], height/2)
    )
    pillar = bpy.context.active_object
    pillar.name = f"Hotspot_{{i:04d}}"
    
    # Apply material
    mat_name = f"HotspotMat_{{i:04d}}"
    emission = 2.0 if intensity > 0.7 else 0.5
    mat = create_hotspot_material(mat_name, color, emission)
    pillar.data.materials.append(mat)
    
    # Move to collection
    bpy.context.scene.collection.objects.unlink(pillar)
    pillar_collection.objects.link(pillar)
    
    # Add glow ring for high intensity
    if intensity > 0.7:
        bpy.ops.mesh.primitive_torus_add(
            major_radius=2.5,
            minor_radius=0.2,
            location=(pos['x'], pos['y'], 0.2)
        )
        glow = bpy.context.active_object
        glow.name = f"Glow_{{i:04d}}"
        
        glow_mat = bpy.data.materials.new(name=f"GlowMat_{{i:04d}}")
        glow_mat.use_nodes = True
        glow_bsdf = glow_mat.node_tree.nodes["Principled BSDF"]
        glow_bsdf.inputs["Emission"].default_value = (*color, 1)
        glow_bsdf.inputs["Emission Strength"].default_value = 5
        glow.data.materials.append(glow_mat)
        
        bpy.context.scene.collection.objects.unlink(glow)
        pillar_collection.objects.link(glow)

# Add lighting
# Sun light
bpy.ops.object.light_add(type='SUN', location=(100, 100, 100))
sun = bpy.context.active_object
sun.name = "UHI_Sun"
sun.data.energy = 3

# Area light for fill
bpy.ops.object.light_add(type='AREA', location=(0, 0, 80))
area = bpy.context.active_object
area.name = "UHI_AreaLight"
area.data.energy = 500
area.data.size = 50

# Add camera
bpy.ops.object.camera_add(location=(150, -100, 100))
camera = bpy.context.active_object
camera.name = "UHI_Camera"
camera.rotation_euler = (math.radians(60), 0, math.radians(45))
bpy.context.scene.camera = camera

# Add HDRI world (optional - uses solid color if no HDRI)
world = bpy.data.worlds.new(name="UHI_World")
bpy.context.scene.world = world
world.use_nodes = True
world_nodes = world.node_tree.nodes
bg_node = world_nodes["Background"]
bg_node.inputs["Color"].default_value = (0.05, 0.05, 0.1, 1)
bg_node.inputs["Strength"].default_value = 0.5

# Set render settings
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.film_transparent = False

print(f"Created {{len(hotspot_data)}} hotspot pillars")
print("Scene setup complete! Press F12 to render.")
'''

    def generate(self, df: pd.DataFrame = None,
                 output_path: str = 'uhi_blender_script.py',
                 **kwargs) -> str:
        """
        Generate Blender Python script.
        
        Args:
            df: DataFrame with hotspot data
            output_path: Output file path
            **kwargs: Additional arguments for prepare_data
            
        Returns:
            Path to generated Python script
        """
        try:
            # Prepare data
            if df is not None:
                self.prepare_data(df, **kwargs)
            elif not self.hotspot_points:
                self.hotspot_points = self._get_default_points()
            
            # Convert points to Python list format
            hotspot_data = [p.to_dict() for p in self.hotspot_points]
            hotspot_str = repr(hotspot_data)
            
            # Generate script
            script_content = self.SCRIPT_TEMPLATE.format(
                hotspot_data=hotspot_str
            )
            
            # Write to file
            with open(output_path, 'w') as f:
                f.write(script_content)
                
            logger.info(f"Generated Blender script: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating Blender script: {e}")
            return ""


class VisualizationManager:
    """
    Unified manager for all visualization outputs.
    
    Provides a single interface to generate all visualization formats,
    including enhanced urban landscape visualizations with buildings.
    """
    
    def __init__(self):
        self.threejs = ThreeJSGenerator()
        self.unity = UnityExporter()
        self.blender = BlenderScriptGenerator()
        
    def generate_all(self, df: pd.DataFrame,
                     output_dir: str = '.',
                     prefix: str = 'uhi',
                     enhanced: bool = False,
                     **kwargs) -> Dict[str, str]:
        """
        Generate all visualization outputs.
        
        Args:
            df: DataFrame with hotspot data
            output_dir: Directory for output files
            prefix: Prefix for output filenames
            enhanced: Whether to generate enhanced urban landscape visualization
            **kwargs: Additional arguments for prepare_data
            
        Returns:
            Dictionary of format -> output path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Generate Three.js (enhanced or simple)
        if enhanced:
            threejs_path = self.threejs.generate_enhanced(
                df,
                hotspot_df=kwargs.get('hotspot_df', df),
                output_path=str(output_dir / f'{prefix}_urban_landscape.html'),
                terrain_size=kwargs.get('terrain_size', (200, 200)),
                **kwargs
            )
        else:
            threejs_path = self.threejs.generate(
                df,
                str(output_dir / f'{prefix}_threejs.html'),
                **kwargs
            )
        if threejs_path:
            results['threejs'] = threejs_path
            
        # Generate Unity
        unity_path = self.unity.generate(
            df,
            str(output_dir / f'{prefix}_unity.json'),
            **kwargs
        )
        if unity_path:
            results['unity'] = unity_path
            
        # Generate Blender
        blender_path = self.blender.generate(
            df,
            str(output_dir / f'{prefix}_blender.py'),
            **kwargs
        )
        if blender_path:
            results['blender'] = blender_path
            
        return results
    
    def generate_urban_landscape(self, df: pd.DataFrame,
                                  hotspot_df: pd.DataFrame = None,
                                  output_dir: str = '.',
                                  prefix: str = 'uhi_city',
                                  terrain_size: Tuple[float, float] = (200, 200),
                                  **kwargs) -> Dict[str, str]:
        """
        Generate full urban landscape visualization with buildings around hotspots.
        
        Args:
            df: DataFrame with urban parameters
            hotspot_df: DataFrame with hotspot data
            output_dir: Output directory
            prefix: File prefix
            terrain_size: Size of terrain
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of format -> output path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Generate enhanced Three.js with urban landscape
        threejs_path = self.threejs.generate_enhanced(
            df=df,
            hotspot_df=hotspot_df if hotspot_df is not None else df,
            output_path=str(output_dir / f'{prefix}_visualization.html'),
            terrain_size=terrain_size,
            **kwargs
        )
        if threejs_path:
            results['threejs_enhanced'] = threejs_path
        
        # Generate Unity data with buildings
        unity_path = self.unity.generate(
            hotspot_df if hotspot_df is not None else df,
            str(output_dir / f'{prefix}_unity.json'),
            **kwargs
        )
        if unity_path:
            results['unity'] = unity_path
        
        # Generate Blender script
        blender_path = self.blender.generate(
            hotspot_df if hotspot_df is not None else df,
            str(output_dir / f'{prefix}_blender.py'),
            **kwargs
        )
        if blender_path:
            results['blender'] = blender_path
        
        logger.info(f"Generated urban landscape visualization in {output_dir}")
        return results
