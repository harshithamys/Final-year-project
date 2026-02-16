# Urban Heat Island (UHI) Analysis System

A comprehensive Python-based system for analyzing Urban Heat Islands, detecting hotspots, recommending mitigation strategies, and generating AR/VR visualizations.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Features](#2-features)
3. [Installation](#3-installation)
4. [Software Requirements](#4-software-requirements)
5. [System Design](#5-system-design)
6. [Implementation and Demonstration](#6-implementation-and-demonstration)
7. [Usage](#7-usage)
8. [API Reference](#8-api-reference)
9. [Contributing](#9-contributing)

---

## 1. Project Overview

The Urban Heat Island (UHI) Analysis System is a final-year project that provides a complete solution for analyzing urban heat patterns, identifying critical hotspots, recommending evidence-based mitigation strategies, and generating immersive AR/VR visualizations for urban planners and policymakers.

### Problem Statement

Urban Heat Islands cause temperatures in cities to be significantly higher than surrounding rural areas, leading to:
- Increased energy consumption
- Health risks from heat stress
- Reduced air quality
- Environmental degradation

This system addresses these challenges through data-driven analysis and actionable recommendations.

---

## 2. Features

### 🔥 Hotspot Detection
- **DBSCAN Clustering**: Density-based spatial clustering to identify high-temperature zones
- **Grid-based Analysis**: Spatial grid analysis for comprehensive area coverage
- **Statistical Analysis**: Mean, max, percentile calculations for UHI intensity

### 🌱 Mitigation Strategies
- **Rule-based Recommender**: Matches urban conditions to appropriate interventions
- **18+ Strategies**: Including tree planting, green roofs, cool pavements, etc.
- **Cost-Benefit Analysis**: Cost per sqm, timeline, and cooling impact (°C reduction)

### 🎮 AR/VR Outputs
- **Three.js HTML**: Interactive 3D web visualization with red hotspot pillars
- **Unity JSON**: Import-ready coordinate data for game engines
- **Blender Python**: Automated 3D scene generation script

### 🤖 Machine Learning
- **Prediction Models**: Random Forest, Gradient Boosting, Ridge Regression
- **Ensemble Model**: Weighted combination for robust predictions
- **Feature Importance**: Understanding UHI drivers

### ⚡ Safety Features
- Handles empty data gracefully
- Missing column detection and fallbacks
- Default recommendations when analysis fails
- Comprehensive error logging

---

## 3. Installation

### Prerequisites

```bash
# Python 3.8+ required
python --version

# Create virtual environment
python -m venv uhi_env
source uhi_env/bin/activate  # Linux/Mac
# or
uhi_env\Scripts\activate  # Windows
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Quick Start

```bash
# Run the complete analysis
python main.py
```

---

## 4. Software Requirements

### 4.1 Runtime Environment

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.8+ | Core runtime |
| pip | 21.0+ | Package management |

### 4.2 Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥1.3.0 | Data manipulation |
| numpy | ≥1.21.0 | Numerical computing |
| scikit-learn | ≥1.0.0 | Machine learning (DBSCAN, Random Forest) |
| matplotlib | ≥3.4.0 | Plotting and visualization |

### 4.3 Optional Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| seaborn | ≥0.11.0 | Enhanced visualizations |
| plotly | ≥5.0.0 | Interactive plots |

### 4.4 Development Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pytest | ≥7.0.0 | Testing framework |
| black | ≥22.0.0 | Code formatting |
| mypy | ≥0.950 | Type checking |

### requirements.txt

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

---

## 5. System Design

### 5.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    UHI ANALYSIS SYSTEM ARCHITECTURE                  │
└─────────────────────────────────────────────────────────────────────┘

                              ┌──────────────┐
                              │   User/CLI   │
                              └──────┬───────┘
                                     │
                              ┌──────▼───────┐
                              │   main.py    │
                              │  (Entry Pt)  │
                              └──────┬───────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
        ▼                            ▼                            ▼
┌───────────────┐          ┌─────────────────┐          ┌─────────────────┐
│     CORE      │          │   MITIGATION    │          │  VISUALIZATION  │
│    MODULE     │          │     MODULE      │          │     MODULE      │
├───────────────┤          ├─────────────────┤          ├─────────────────┤
│ UHIDataLoader │          │ MitigationRec.  │          │ ThreeJSGenerator│
│ HotspotDetect │◄────────►│ StrategyDB      │◄────────►│ UnityExporter   │
│ GridAnalyzer  │          │ Strategy        │          │ BlenderScript   │
└───────────────┘          └─────────────────┘          └─────────────────┘
        │                            │                            │
        │                            │                            │
        │                    ┌───────▼───────┐                    │
        │                    │  PREDICTION   │                    │
        └───────────────────►│    MODULE     │◄───────────────────┘
                             ├───────────────┤
                             │ UHIPrediction │
                             │ EnsembleModel │
                             └───────────────┘
                                     │
                              ┌──────▼───────┐
                              │   OUTPUT     │
                              │  (Files/DB)  │
                              └──────────────┘
```

### Module Interactions

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA FLOW DIAGRAM                            │
└─────────────────────────────────────────────────────────────────┘

    CSV Files                Processing Pipeline              Outputs
    ─────────                ───────────────────              ───────

┌──────────────┐       ┌─────────────────────┐       ┌──────────────┐
│ UHI_d.csv    │──────►│   UHIDataLoader     │       │ Hotspot CSV  │
│ UHI_n.csv    │       │   - load_csv()      │       └──────────────┘
└──────────────┘       │   - preprocess()    │              ▲
                       │   - add_coords()    │              │
                       └─────────┬───────────┘              │
                                 │                          │
                       ┌─────────▼───────────┐              │
                       │  HotspotDetector    │──────────────┘
                       │  - DBSCAN cluster   │
                       │  - threshold filter │       ┌──────────────┐
                       └─────────┬───────────┘       │ Recommend CSV│
                                 │                   └──────────────┘
                       ┌─────────▼───────────┐              ▲
                       │  GridBasedAnalyzer  │              │
                       │  - spatial grid     │              │
                       │  - cell statistics  │              │
                       └─────────┬───────────┘              │
                                 │                          │
                       ┌─────────▼───────────┐              │
                       │ MitigationRecommend │──────────────┘
                       │  - analyze_cond()   │
                       │  - recommend()      │       ┌──────────────┐
                       └─────────┬───────────┘       │ Model .pkl   │
                                 │                   └──────────────┘
                       ┌─────────▼───────────┐              ▲
                       │  UHIPredictionModel │──────────────┘
                       │  - train()          │
                       │  - predict()        │       ┌──────────────┐
                       └─────────┬───────────┘       │ Three.js HTML│
                                 │                   ├──────────────┤
                       ┌─────────▼───────────┐       │ Unity JSON   │
                       │ VisualizationManager│──────►├──────────────┤
                       │  - generate_all()   │       │ Blender .py  │
                       └─────────────────────┘       └──────────────┘
```

### 5.2 Proposed System

#### Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SYSTEM WORKFLOW                               │
└─────────────────────────────────────────────────────────────────────┘

    ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
    │  LOAD   │────►│ DETECT  │────►│ ANALYZE │────►│ PREDICT │
    │  DATA   │     │ HOTSPOT │     │CONDITIONS│    │  UHI    │
    └─────────┘     └─────────┘     └─────────┘     └─────────┘
         │               │               │               │
         ▼               ▼               ▼               ▼
    ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
    │Preproc- │     │ DBSCAN  │     │ Rule-   │     │ Random  │
    │essing   │     │ Grid    │     │ Based   │     │ Forest  │
    │ Clean   │     │ Analysis│     │ Matching│     │ GB/Ridge│
    └─────────┘     └─────────┘     └─────────┘     └─────────┘
                          │               │               │
                          └───────────────┴───────────────┘
                                          │
                                          ▼
                               ┌─────────────────────┐
                               │     RECOMMEND       │
                               │  MITIGATION PLANS   │
                               └─────────────────────┘
                                          │
                                          ▼
                               ┌─────────────────────┐
                               │    VISUALIZE        │
                               │  (3D/AR/VR Output)  │
                               └─────────────────────┘
```

#### Comparison with Existing Systems

| Feature | Traditional GIS | Our System |
|---------|----------------|------------|
| Hotspot Detection | Manual threshold | DBSCAN + Grid automated |
| Mitigation Advice | General guidelines | Context-aware recommendations |
| Cost Analysis | Separate calculation | Integrated with strategies |
| 3D Visualization | Requires plugins | Native Three.js/Unity/Blender |
| ML Prediction | Not available | Random Forest ensemble |
| Error Handling | Crashes on bad data | Graceful fallbacks |

### 5.3 Detailed Design

#### Module-wise Explanation

##### Core Module (`core.py`)

```
┌─────────────────────────────────────────────────────────────────┐
│                       CORE MODULE                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────┐   ┌────────────────┐   ┌────────────────┐   │
│  │ UHIDataLoader  │   │ HotspotDetector│   │GridBasedAnalyz │   │
│  ├────────────────┤   ├────────────────┤   ├────────────────┤   │
│  │ +load_csv()    │   │ +detect()      │   │ +analyze()     │   │
│  │ +load_multiple │   │ +_calculate_   │   │ +get_grid_     │   │
│  │ +preprocess()  │   │   statistics() │   │   matrix()     │   │
│  │ +add_synth_    │   │ +_create_      │   │ +get_hotspot_  │   │
│  │   coordinates()│   │   default()    │   │   summary()    │   │
│  └────────────────┘   └────────────────┘   └────────────────┘   │
│                                                                  │
│  Data Classes:                                                   │
│  • HotspotResult: indices, labels, n_clusters, statistics        │
│  • GridCell: row, col, center_x/y, uhi_values, is_hotspot        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

##### Mitigation Module (`mitigation.py`)

```
┌─────────────────────────────────────────────────────────────────┐
│                     MITIGATION MODULE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────┐   ┌────────────────┐   ┌────────────────┐   │
│  │   Strategy     │   │ StrategyDB     │   │MitigationRecom │   │
│  ├────────────────┤   ├────────────────┤   ├────────────────┤   │
│  │ +name          │   │ +get_all_      │   │ +analyze_cond()│   │
│  │ +category      │   │   strategies() │   │ +recommend()   │   │
│  │ +cost_per_sqm  │   │                │   │ +_calculate_   │   │
│  │ +timeline      │   │ Contains 18+   │   │   applicability│   │
│  │ +cooling_impact│   │ strategies:    │   │ +_determine_   │   │
│  │ +co_benefits   │   │ • Tree plant   │   │   priority()   │   │
│  │ +total_5yr_cost│   │ • Green roofs  │   │ +to_dataframe()│   │
│  └────────────────┘   │ • Cool pave    │   └────────────────┘   │
│                       │ • Water feat.  │                        │
│  Enums:               │ • Planning     │                        │
│  • StrategyCategory   └────────────────┘                        │
│  • Priority                                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

##### Visualization Module (`visualization.py`)

```
┌─────────────────────────────────────────────────────────────────┐
│                   VISUALIZATION MODULE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │             BaseVisualizationGenerator                   │    │
│  │  +prepare_data()  +_intensity_to_color()  +_get_default │    │
│  └───────────────────────────┬─────────────────────────────┘    │
│                              │                                   │
│         ┌────────────────────┼────────────────────┐             │
│         │                    │                    │             │
│         ▼                    ▼                    ▼             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │ ThreeJS     │     │  Unity      │     │  Blender    │       │
│  │ Generator   │     │  Exporter   │     │  ScriptGen  │       │
│  ├─────────────┤     ├─────────────┤     ├─────────────┤       │
│  │ +generate() │     │ +generate() │     │ +generate() │       │
│  │ HTML_TEMPL  │     │ +_get_unity │     │ SCRIPT_TEMPL│       │
│  │             │     │   _script() │     │             │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
│                                                                  │
│  Output Formats:                                                 │
│  • Three.js: Interactive HTML with WebGL                         │
│  • Unity: JSON with C# import script                             │
│  • Blender: Python script for Cycles rendering                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

##### Prediction Module (`prediction.py`)

```
┌─────────────────────────────────────────────────────────────────┐
│                    PREDICTION MODULE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐     │
│  │                UHIPredictionModel                       │     │
│  ├────────────────────────────────────────────────────────┤     │
│  │ +train(df, features, target)                            │     │
│  │ +predict(df) -> np.ndarray                              │     │
│  │ +save_model(filepath)                                   │     │
│  │ +load_model(filepath)                                   │     │
│  │ +predict_with_scenarios(base_df, scenarios)             │     │
│  │ +get_feature_importance_report()                        │     │
│  └────────────────────────────────────────────────────────┘     │
│                              │                                   │
│                              │ inherits/uses                     │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────┐     │
│  │               EnsembleUHIModel                          │     │
│  │  Combines: Random Forest (60%) + Gradient Boost (40%)   │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                  │
│  Supported Models:                                               │
│  • RandomForestRegressor                                         │
│  • GradientBoostingRegressor                                     │
│  • Ridge Regression                                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Data Flow / Sequence Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    SEQUENCE DIAGRAM                              │
└─────────────────────────────────────────────────────────────────┘

  User          main.py       DataLoader    HotspotDet    Mitigation
   │               │              │             │             │
   │──run()───────►│              │             │             │
   │               │──load_csv()─►│             │             │
   │               │              │───────┐     │             │
   │               │              │  read │     │             │
   │               │              │◄──────┘     │             │
   │               │◄─DataFrame───│             │             │
   │               │              │             │             │
   │               │──detect()────────────────►│             │
   │               │              │             │────┐        │
   │               │              │             │DBSCAN       │
   │               │              │             │◄───┘        │
   │               │◄─HotspotResult────────────│             │
   │               │              │             │             │
   │               │──recommend()─────────────────────────────►│
   │               │              │             │             │─┐
   │               │              │             │             │analyze
   │               │              │             │             │◄┘
   │               │◄─List[Strategy]──────────────────────────│
   │               │              │             │             │
   │               │──generate_all()───────────────────────────────►│
   │               │              │             │             │     │
   │               │◄─output_paths────────────────────────────────────│
   │◄──results─────│              │             │             │
   │               │              │             │             │

                              Prediction     Visualization
                                  │              │
   (continued...)                 │              │
   │               │──train()────►│              │
   │               │              │──────┐       │
   │               │              │ fit  │       │
   │               │              │◄─────┘       │
   │               │◄─ModelMetrics│              │
   │               │              │              │
   │               │──generate_all()────────────►│
   │               │              │              │──┐
   │               │              │              │  │write files
   │               │              │              │◄─┘
   │               │◄─paths───────────────────────│
```

---

## 6. Implementation and Demonstration

### 6.1 Tools and Technologies

| Category | Tool/Technology | Version | Purpose |
|----------|-----------------|---------|---------|
| **Language** | Python | 3.8+ | Core development |
| **Data Processing** | pandas | 1.3+ | DataFrame operations |
| **Numerical** | NumPy | 1.21+ | Array computations |
| **Machine Learning** | scikit-learn | 1.0+ | DBSCAN, RF, GB |
| **Visualization** | matplotlib | 3.4+ | 2D plots |
| **3D Web** | Three.js | r128 | WebGL visualization |
| **Game Engine** | Unity | 2021+ | JSON import |
| **3D Modeling** | Blender | 2.8+ | Python scripting |
| **IDE** | VS Code/PyCharm | Latest | Development |
| **Version Control** | Git | 2.30+ | Source management |

### 6.2 Module Implementation

#### Module 1: Hotspot Detection (DBSCAN + Grid)

**Implementation:**

```python
from uhi_analysis.core import HotspotDetector, GridBasedAnalyzer

# DBSCAN-based detection
detector = HotspotDetector(
    eps=0.5,              # Maximum distance between points
    min_samples=3,        # Minimum cluster size
    uhi_threshold_percentile=75  # Only analyze top 25%
)

result = detector.detect(dataframe, uhi_column='UHI_d')
print(f"Found {result.n_clusters} hotspot clusters")
print(f"Statistics: {result.statistics}")

# Grid-based analysis
grid = GridBasedAnalyzer(grid_rows=10, grid_cols=10)
hotspot_cells = grid.analyze(dataframe, uhi_column='UHI_d')

# Get heatmap matrix
heatmap = grid.get_grid_matrix('mean_uhi')
```

**Sample Output:**
```
DBSCAN Results:
  - Clusters found: 12
  - Hotspot points: 245
  - Hotspot percentage: 24.5%
  - Mean UHI (hotspots): 0.1687°C

Grid Analysis Results:
  - Hotspot cells: 25
  - Total cells: 100
  - Hotspot area: 25.0%
```

#### Module 2: Mitigation Recommender

**Implementation:**

```python
from uhi_analysis.mitigation import MitigationRecommender

recommender = MitigationRecommender()

# Analyze urban conditions
conditions = recommender.analyze_conditions(dataframe)

# Get recommendations
recommendations = recommender.recommend(
    dataframe,
    max_recommendations=5,
    budget_limit=100  # Max $100/sqm
)

# Display recommendations
for strategy in recommendations:
    print(f"{strategy.name}")
    print(f"  Cost: ${strategy.cost_per_sqm}/m²")
    print(f"  Cooling: {strategy.cooling_impact_celsius}°C")
    print(f"  Timeline: {strategy.timeline_months} months")
```

**Sample Output:**
```
Top 5 Recommendations:
----------------------------------------------------------------------
1. Street Tree Planting
   Category: VEGETATION
   Priority: CRITICAL
   Cost: $45/m²
   Timeline: 6 months
   Cooling Impact: 2.5°C reduction

2. Cool Roofs (Reflective Coating)
   Category: BUILDING
   Priority: HIGH
   Cost: $25/m²
   Timeline: 1 months
   Cooling Impact: 1.5°C reduction
```

### 6.3 Screenshots / Prototype

#### Console Output

```
╔════════════════════════════════════════════════════════════════════╗
║               URBAN HEAT ISLAND ANALYSIS SYSTEM                    ║
║                    Complete Workflow Demo                          ║
╚════════════════════════════════════════════════════════════════════╝

======================================================================
  1. DATA LOADING & PREPROCESSING
======================================================================

Loading UHI datasets...
✓ Daytime data loaded: 1000 records
✓ Nighttime data loaded: 1000 records

Daytime UHI Statistics:
  - Mean: 0.1512°C
  - Max:  0.1780°C
  - Min:  0.1180°C
  - Std:  0.0156°C

✓ Synthetic coordinates added
✓ Data preprocessed

======================================================================
  4. UHI PREDICTION MODEL
======================================================================

Training Random Forest model on daytime data...

Model Performance (Daytime UHI):
  - R² Score: 0.9234
  - RMSE: 0.004521°C
  - MAE: 0.003102°C
  - CV R² Mean: 0.9156 ± 0.0234

Top 5 Important Features:
  1. GnPR: 0.2845
  2. asphalt_ratio: 0.1923
  3. bldDensity: 0.1456
  4. roadDensity: 0.1234
  5. avg_BH: 0.0987
```

#### Three.js Visualization Preview

The generated HTML file creates an interactive 3D scene:
- **Terrain**: Dark gray ground plane
- **Hotspot Pillars**: Color-coded cylinders (green→yellow→red)
- **Controls**: Orbit camera with zoom/pan
- **Tooltips**: Hover to see UHI values

#### Generated File Structure

```
output/
├── recommendations.csv          # Mitigation strategies
├── uhi_model.pkl               # Trained ML model
├── visualizations/
│   ├── uhi_hotspots_threejs.html    # Web 3D viewer
│   ├── uhi_hotspots_unity.json      # Unity import data
│   └── uhi_hotspots_blender.py      # Blender script
└── plots/
    ├── uhi_analysis.png        # Distribution & heatmap
    └── correlation_matrix.png  # Feature correlations
```

---

## 7. Usage

### Basic Usage

```python
from uhi_analysis import (
    UHIDataLoader, HotspotDetector, 
    MitigationRecommender, UHIPredictionModel
)

# Load data
loader = UHIDataLoader()
df = loader.load_csv('path/to/uhi_data.csv')
df = loader.add_synthetic_coordinates(df)

# Detect hotspots
detector = HotspotDetector()
result = detector.detect(df, uhi_column='UHI_d')

# Get recommendations
recommender = MitigationRecommender()
strategies = recommender.recommend(df)

# Train prediction model
model = UHIPredictionModel()
metrics = model.train(df, target_column='UHI_d')
predictions = model.predict(new_data)
```

### CLI Usage

```bash
# Run full analysis
python main.py

# With custom data paths
python -c "
from main import *
loader = UHIDataLoader()
df = loader.load_csv('custom_data.csv')
# ... continue analysis
"
```

---

## 8. API Reference

### UHIDataLoader

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `load_csv` | filepath, uhi_column | DataFrame | Load CSV file |
| `load_multiple` | day_path, night_path | Tuple | Load both datasets |
| `add_synthetic_coordinates` | df, grid_size | DataFrame | Add lat/lon |
| `preprocess` | df, fill_missing, scale | DataFrame | Clean data |

### HotspotDetector

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `detect` | df, uhi_col, lat_col, lon_col | HotspotResult | Run DBSCAN |

### MitigationRecommender

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `analyze_conditions` | df | Dict | Analyze urban metrics |
| `recommend` | df, max_rec, budget | List[Strategy] | Get strategies |
| `to_dataframe` | recommendations | DataFrame | Export to table |

### UHIPredictionModel

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `train` | df, features, target | ModelMetrics | Train model |
| `predict` | df | np.ndarray | Make predictions |
| `save_model` | filepath | bool | Persist model |
| `load_model` | filepath | bool | Load model |

---

## 9. Contributing

### Development Setup

```bash
git clone https://github.com/username/uhi-analysis.git
cd uhi-analysis
pip install -e ".[dev]"
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings (Google style)
- Maintain 80% test coverage

### Pull Request Process

1. Fork the repository
2. Create feature branch
3. Write tests
4. Submit PR with description

---

## License

MIT License - see [LICENSE](LICENSE) file.

## Authors

- UHI Analysis Team
- Final Year Project - 2024

## Acknowledgments

- Dataset: UHI simulation data (NZFL)
- Three.js community
- scikit-learn maintainers

---

*Last Updated: February 2026*
