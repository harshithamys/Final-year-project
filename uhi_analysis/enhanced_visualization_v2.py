#!/usr/bin/env python3
"""
Advanced UHI Visualization with Mitigation Strategies
Generates an interactive 3D visualization with:
- Building surface hotspot integration
- Ground-level heat zones
- Dynamically placed cooling vegetation
- Wind visualization
- Interactive mitigation panel
- Temperature reduction estimates
"""

import json
from pathlib import Path
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.bool_, np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def generate_advanced_html(urban_data, output_path='output/urban_visualization/uhi_advanced_visualization.html'):
    """Generate advanced visualization HTML"""

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced UHI Analysis - 3D Visualization with Mitigation</title>
    <style>
        * {{ margin: 0; padding: 0; }}
        body {{ overflow: hidden; background: #0a0e27; font-family: 'Arial', sans-serif; }}
        canvas {{ display: block; width: 100%; height: 100%; }}

        #ui {{ position: absolute; user-select: none; pointer-events: none; font-size: 13px; }}

        .panel {{ background: rgba(10, 14, 39, 0.95); backdrop-filter: blur(8px);
                  border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 15px;
                  color: #fff; pointer-events: auto; }}

        #sidebar {{ position: absolute; top: 20px; left: 20px; width: 300px; max-height: 80vh;
                   overflow-y: auto; z-index: 100; }}

        #sidebar h2 {{ color: #4ecdc4; margin-bottom: 12px; font-size: 16px; }}

        .stat-row {{ display: flex; justify-content: space-between; margin: 8px 0; font-size: 12px; }}
        .stat-label {{ color: #888; }}
        .stat-value {{ color: #4ecdc4; font-weight: bold; }}

        button {{ width: 100%; padding: 8px; margin: 6px 0; background: #2a5f7a;
                 color: #4ecdc4; border: 1px solid #4ecdc4; border-radius: 4px;
                 cursor: pointer; font-size: 12px; font-weight: bold; transition: 0.2s; }}
        button:hover {{ background: #4ecdc4; color: #0a0e27; }}

        #mitigationPanel {{ position: absolute; right: 20px; top: 20px; width: 320px;
                           max-height: 80vh; overflow-y: auto; z-index: 101; display: none; }}
        #mitigationPanel h3 {{ color: #ff6b6b; margin-bottom: 10px; }}

        .strategy {{ background: rgba(42, 95, 122, 0.8); padding: 10px; margin: 8px 0;
                    border-left: 3px solid #4ecdc4; border-radius: 4px; }}
        .strategy-name {{ color: #4ecdc4; font-weight: bold; }}
        .strategy-detail {{ color: #aaa; font-size: 11px; margin: 4px 0; }}
        .strategy-cooling {{ color: #ff6b6b; }}

        #tempReduction {{ position: absolute; bottom: 20px; right: 20px; z-index: 100;
                         min-width: 250px; }}
        #tempReduction h4 {{ color: #4ecdc4; margin-bottom: 8px; }}
        .reduction-stat {{ display: flex; justify-content: space-between; margin: 6px 0; font-size: 12px; }}

        #tooltip {{ position: absolute; background: rgba(10, 14, 39, 0.98); padding: 10px;
                   border: 1px solid #4ecdc4; border-radius: 4px; display: none; font-size: 11px;
                   pointer-events: none; z-index: 200; max-width: 200px; }}
        .tooltip-title {{ color: #4ecdc4; font-weight: bold; margin-bottom: 4px; }}
        .tooltip-row {{ color: #aaa; margin: 2px 0; }}
    </style>
</head>
<body>
    <div id="ui">
        <div id="sidebar" class="panel">
            <h2>🏙️ UHI Analysis</h2>
            <div class="stat-row">
                <span class="stat-label">Buildings:</span>
                <span class="stat-value">{len(urban_data.get('buildings', []))}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Hotspots:</span>
                <span class="stat-value">{len(urban_data.get('hotspot_zones', []))}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Trees:</span>
                <span class="stat-value">{len(urban_data.get('trees', []))}</span>
            </div>

            <button onclick="resetCamera()">🔄 Reset Camera</button>
            <button onclick="toggleDay()">🌅 Day/Night</button>
            <button onclick="toggleMitigation()">🌱 Toggle Mitigation</button>
        </div>

        <div id="mitigationPanel" class="panel">
            <h3>📋 Recommended Strategies</h3>
            <div id="strategiesList"></div>
            <button onclick="closeMitigation()" style="margin-top: 10px;">Close</button>
        </div>

        <div id="tempReduction" class="panel" style="display: none;">
            <h4>✓ Mitigation Active</h4>
            <div id="reductionStats"></div>
        </div>
    </div>

    <div id="tooltip"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@r128/examples/js/controls/OrbitControls.js"></script>

    <script>
        const DATA = {json.dumps(urban_data, cls=NumpyEncoder)};

        let scene, camera, renderer, controls;
        let groups = {{ buildings: null, hotspots: null, trees: null, mitigation: null }};
        let isDayMode = true;
        let showMitigation = false;

        function init() {{
            // Scene setup
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1a2744);
            scene.fog = new THREE.Fog(0x1a2744, 300, 600);

            // Camera
            camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.1, 2000);
            camera.position.set(120, 80, 120);
            camera.lookAt(100, 0, 100);

            // Renderer
            renderer = new THREE.WebGLRenderer({{ antialias: true, precision: 'highp' }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowShadowMap;
            document.body.appendChild(renderer.domElement);

            // Lighting
            const ambLight = new THREE.AmbientLight(0xffffff, 0.65);
            scene.add(ambLight);

            const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
            dirLight.position.set(200, 200, 200);
            dirLight.castShadow = true;
            dirLight.shadow.mapSize.width = 4096;
            dirLight.shadow.mapSize.height = 4096;
            dirLight.shadow.camera.far = 500;
            dirLight.shadow.camera.left = -250;
            dirLight.shadow.camera.right = 250;
            dirLight.shadow.camera.top = 250;
            dirLight.shadow.camera.bottom = -250;
            scene.add(dirLight);

            const hemLight = new THREE.HemisphereLight(0x87ceeb, 0x4a3728, 0.4);
            scene.add(hemLight);

            // Ground
            const groundGeo = new THREE.PlaneGeometry(200, 200, 64, 64);
            const groundMat = new THREE.MeshStandardMaterial({{ color: 0x3d5a3d, roughness: 0.85 }});
            const ground = new THREE.Mesh(groundGeo, groundMat);
            ground.rotation.x = -Math.PI / 2;
            ground.position.y = 0;
            ground.receiveShadow = true;
            scene.add(ground);

            // Create all elements
            createBuildings();
            createHotspots();
            createTrees();
            createMitigation();

            // Controls
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.target.set(100, 20, 100);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.update();

            // Event listeners
            renderer.domElement.addEventListener('mousemove', onMouseMove);
            renderer.domElement.addEventListener('click', onBuildingClick);
            window.addEventListener('resize', onWindowResize);

            animate();
        }}

        function createBuildings() {{
            groups.buildings = new THREE.Group();

            DATA.buildings.forEach((b, i) => {{
                const exposure = b.heat_exposure || 0;
                const baseCol = new THREE.Color(b.color.r, b.color.g, b.color.b);
                const hotCol = new THREE.Color(1.0, 0.2, 0.0);
                const finalCol = baseCol.clone().lerp(hotCol, exposure);

                const geo = new THREE.BoxGeometry(b.dimensions.width, b.dimensions.height, b.dimensions.depth);
                const mat = new THREE.MeshStandardMaterial({{
                    color: finalCol,
                    emissive: finalCol.clone(),
                    emissiveIntensity: 0.1 + exposure * 0.5,
                    roughness: 0.7,
                    metalness: 0.05
                }});

                const mesh = new THREE.Mesh(geo, mat);
                mesh.position.set(b.position.x, b.dimensions.height / 2, b.position.y);
                mesh.castShadow = true;
                mesh.receiveShadow = true;
                mesh.userData = {{ type: 'building', id: b.id, exposure, temp: 28 + exposure * 10 }};

                groups.buildings.add(mesh);
            }});

            scene.add(groups.buildings);
        }}

        function createHotspots() {{
            groups.hotspots = new THREE.Group();

            DATA.hotspot_zones.forEach(z => {{
                // Ground circle halo
                const circleGeo = new THREE.CircleGeometry(z.radius, 64);
                const circleMat = new THREE.MeshBasicMaterial({{
                    color: new THREE.Color(z.color.r, z.color.g, z.color.b),
                    transparent: true,
                    opacity: 0.5,
                    side: THREE.DoubleSide
                }});
                const circle = new THREE.Mesh(circleGeo, circleMat);
                circle.rotation.x = -Math.PI / 2;
                circle.position.set(z.center.x, 0.05, z.center.y);
                groups.hotspots.add(circle);

                // Pulsing glow ring
                const ringGeo = new THREE.RingGeometry(z.radius * 0.8, z.radius, 64);
                const ringMat = new THREE.MeshBasicMaterial({{
                    color: 0xff6600,
                    transparent: true,
                    opacity: 0.3,
                    side: THREE.DoubleSide
                }});
                const ring = new THREE.Mesh(ringGeo, ringMat);
                ring.rotation.x = -Math.PI / 2;
                ring.position.set(z.center.x, 0.06, z.center.y);
                ring.userData = {{ phase: Math.random() * Math.PI * 2 }};
                groups.hotspots.add(ring);

                // Glow light
                const light = new THREE.PointLight(0xff6600, z.intensity * 2, z.radius * 1.5);
                light.position.set(z.center.x, 3, z.center.y);
                groups.hotspots.add(light);
            }});

            scene.add(groups.hotspots);
        }}

        function createTrees() {{
            groups.trees = new THREE.Group();

            DATA.trees.forEach(t => {{
                // Trunk
                const trunkGeo = new THREE.CylinderGeometry(t.canopy_radius * 0.12, t.canopy_radius * 0.15, t.height * 0.4, 8);
                const trunkMat = new THREE.MeshStandardMaterial({{ color: 0x4a3728, roughness: 0.8 }});
                const trunk = new THREE.Mesh(trunkGeo, trunkMat);
                trunk.position.set(t.x, t.height * 0.2, t.y);
                trunk.castShadow = true;
                groups.trees.add(trunk);

                // Foliage
                const foliageGeo = new THREE.SphereGeometry(t.canopy_radius, 16, 16);
                const foliageMat = new THREE.MeshStandardMaterial({{ color: 0x2d5a2d, roughness: 0.9 }});
                const foliage = new THREE.Mesh(foliageGeo, foliageMat);
                foliage.scale.y = 0.8;
                foliage.position.set(t.x, t.height * 0.7, t.y);
                foliage.castShadow = true;
                foliage.receiveShadow = true;
                groups.trees.add(foliage);
            }});

            scene.add(groups.trees);
        }}

        function createMitigation() {{
            groups.mitigation = new THREE.Group();

            DATA.hotspot_zones.forEach(z => {{
                // Wind arrows (8 radially)
                for (let a = 0; a < 8; a++) {{
                    const angle = (a / 8) * Math.PI * 2;
                    const ox = z.center.x + Math.cos(angle) * z.radius * 0.9;
                    const oz = z.center.y + Math.sin(angle) * z.radius * 0.9;

                    const dir = new THREE.Vector3(z.center.x - ox, 0, z.center.y - oz).normalize();
                    const arrow = new THREE.ArrowHelper(dir, new THREE.Vector3(ox, 8, oz), 10, 0x88ccff, 2, 1);
                    arrow.userData = {{ phase: a * 0.78 }};
                    groups.mitigation.add(arrow);
                }}
            }});

            // Tree canopy spreads
            DATA.trees.forEach(t => {{
                const spreadGeo = new THREE.RingGeometry(t.canopy_radius, t.canopy_radius * 1.8, 32);
                const spreadMat = new THREE.MeshBasicMaterial({{ color: 0x44aa44, transparent: true, opacity: 0.15 }});
                const spread = new THREE.Mesh(spreadGeo, spreadMat);
                spread.rotation.x = -Math.PI / 2;
                spread.position.set(t.x, 0.15, t.y);
                groups.mitigation.add(spread);

                const shadeDisk = new THREE.CircleGeometry(t.canopy_radius * 1.4, 32);
                const shadeMat = new THREE.MeshBasicMaterial({{ color: 0x228833, transparent: true, opacity: 0.12 }});
                const shade = new THREE.Mesh(shadeDisk, shadeMat);
                shade.rotation.x = -Math.PI / 2;
                shade.position.set(t.x, t.height + t.canopy_radius * 0.5, t.y);
                groups.mitigation.add(shade);
            }});

            groups.mitigation.visible = false;
            scene.add(groups.mitigation);
        }}

        function toggleMitigation() {{
            showMitigation = !showMitigation;
            groups.mitigation.visible = showMitigation;
            document.getElementById('tempReduction').style.display = showMitigation ? 'block' : 'none';

            if (showMitigation) {{
                groups.buildings.children.forEach(mesh => {{
                    if (mesh.material.emissiveIntensity) {{
                        mesh.userData.originalEmissive = mesh.material.emissiveIntensity;
                        mesh.material.emissiveIntensity -= 0.2;
                    }}
                }});
                updateReductionStats();
            }} else {{
                groups.buildings.children.forEach(mesh => {{
                    if (mesh.userData.originalEmissive) {{
                        mesh.material.emissiveIntensity = mesh.userData.originalEmissive;
                    }}
                }});
            }}
        }}

        function updateReductionStats() {{
            const html = `
                <div class="reduction-stat">
                    <span>Street Trees:</span>
                    <span class="strategy-cooling">−2.5°C</span>
                </div>
                <div class="reduction-stat">
                    <span>Green Roofs:</span>
                    <span class="strategy-cooling">−1.8°C</span>
                </div>
                <div class="reduction-stat">
                    <span>Wind Corridors:</span>
                    <span class="strategy-cooling">−1.5°C</span>
                </div>
                <hr style="margin: 8px 0; border: none; border-top: 1px solid #4ecdc4;">
                <div class="reduction-stat">
                    <strong>Total Cooling:</strong>
                    <span class="strategy-cooling"><strong>−4.3°C</strong></span>
                </div>
            `;
            document.getElementById('reductionStats').innerHTML = html;
        }}

        function toggleDay() {{
            isDayMode = !isDayMode;
            if (isDayMode) {{
                scene.background = new THREE.Color(0x1a2744);
                scene.fog.color = new THREE.Color(0x1a2744);
            }} else {{
                scene.background = new THREE.Color(0x0a0a15);
                scene.fog.color = new THREE.Color(0x0a0a15);
            }}
        }}

        function resetCamera() {{
            camera.position.set(120, 80, 120);
            controls.target.set(100, 20, 100);
            controls.update();
        }}

        function closeMitigation() {{
            document.getElementById('mitigationPanel').style.display = 'none';
        }}

        function onBuildingClick(event) {{
            const raycaster = new THREE.Raycaster();
            const mouse = new THREE.Vector2();
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

            raycaster.setFromCamera(mouse, camera);
            const hits = raycaster.intersectObjects(groups.buildings.children);

            if (hits.length > 0) {{
                const data = hits[0].object.userData;
                showMitigationPanel(data);
            }}
        }}

        function showMitigationPanel(building) {{
            const panel = document.getElementById('mitigationPanel');
            let html = `<h3>🏢 ${{building.id}}</h3>`;
            html += `<div class="stat-row"><span>Heat Level:</span><span>${{(building.exposure * 100).toFixed(0)}}%</span></div>`;
            html += `<div class="stat-row"><span>Est. Temp:</span><span>${{building.temp.toFixed(1)}}°C</span></div><hr style="margin: 10px 0; border: none; border-top: 1px solid #4ecdc4;">`;

            const strategies = [
                {{ name: 'Street Tree Planting', cooling: 2.5, cost: '$45/m²', time: '6 months' }},
                {{ name: 'Green Roofs', cooling: 1.8, cost: '$120/m²', time: '4 months' }},
                {{ name: 'Urban Parks', cooling: 3.5, cost: '$85/m²', time: '18 months' }},
                {{ name: 'Cool Pavements', cooling: 1.8, cost: '$35/m²', time: '2 months' }}
            ];

            strategies.forEach(s => {{
                html += `<div class="strategy">
                    <div class="strategy-name">${{s.name}}</div>
                    <div class="strategy-detail">Cooling: <span class="strategy-cooling">−${{s.cooling}}°C</span></div>
                    <div class="strategy-detail">Cost: ${{s.cost}}</div>
                    <div class="strategy-detail">Timeline: ${{s.time}}</div>
                </div>`;
            }});

            document.getElementById('strategiesList').innerHTML = html;
            panel.style.display = 'block';
        }}

        function onMouseMove(event) {{
            const raycaster = new THREE.Raycaster();
            const mouse = new THREE.Vector2();
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

            raycaster.setFromCamera(mouse, camera);
            const hits = raycaster.intersectObjects(groups.buildings.children);

            const tooltip = document.getElementById('tooltip');
            if (hits.length > 0) {{
                const data = hits[0].object.userData;
                tooltip.style.display = 'block';
                tooltip.style.left = (event.clientX + 10) + 'px';
                tooltip.style.top = (event.clientY + 10) + 'px';
                tooltip.innerHTML = `
                    <div class="tooltip-title">${{data.id}}</div>
                    <div class="tooltip-row">Heat Exposure: ${{(data.exposure * 100).toFixed(0)}}%</div>
                    <div class="tooltip-row">Temp: ${{data.temp.toFixed(1)}}°C</div>
                `;
            }} else {{
                tooltip.style.display = 'none';
            }}
        }}

        function onWindowResize() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }}

        function animate() {{
            requestAnimationFrame(animate);

            // Animate hotspot glows
            groups.hotspots.children.forEach((child, i) => {{
                if (child.userData.phase !== undefined) {{
                    child.material.opacity = 0.2 + 0.25 * Math.sin(Date.now() * 0.003 + child.userData.phase);
                }}
            }});

            controls.update();
            renderer.render(scene, camera);
        }}

        window.addEventListener('load', init);
    </script>
</body>
</html>'''

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)

    return output_path
