"""
Mitigation Strategies Module
============================

Rule-based recommender system for UHI mitigation strategies.
Matches urban characteristics to appropriate interventions with
estimated costs, timelines, and cooling impacts.
"""

import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class StrategyCategory(Enum):
    """Categories of mitigation strategies."""
    VEGETATION = "vegetation"
    BUILDING = "building"
    PAVEMENT = "pavement"
    WATER = "water"
    PLANNING = "urban_planning"


class Priority(Enum):
    """Priority levels for recommendations."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


@dataclass
class Strategy:
    """
    Represents a UHI mitigation strategy.
    
    Attributes:
        name: Strategy name
        category: Type of intervention
        description: Detailed description
        cost_per_sqm: Implementation cost in USD per square meter
        timeline_months: Expected implementation time
        cooling_impact_celsius: Expected temperature reduction
        maintenance_annual: Annual maintenance cost per sqm
        co_benefits: Additional benefits (air quality, stormwater, etc.)
        priority: Recommendation priority
        applicability_score: How well it matches the urban context (0-1)
    """
    name: str
    category: StrategyCategory
    description: str
    cost_per_sqm: float
    timeline_months: int
    cooling_impact_celsius: float
    maintenance_annual: float = 0.0
    co_benefits: List[str] = field(default_factory=list)
    priority: Priority = Priority.MEDIUM
    applicability_score: float = 0.0
    
    def total_5year_cost(self, area_sqm: float = 1000) -> float:
        """Calculate total 5-year cost including maintenance."""
        initial = self.cost_per_sqm * area_sqm
        maintenance = self.maintenance_annual * area_sqm * 5
        return initial + maintenance
    
    def cost_effectiveness(self) -> float:
        """Calculate cost-effectiveness ratio (cooling per dollar spent)."""
        if self.cost_per_sqm == 0:
            return float('inf')
        return self.cooling_impact_celsius / self.cost_per_sqm
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'name': self.name,
            'category': self.category.value,
            'description': self.description,
            'cost_per_sqm_usd': self.cost_per_sqm,
            'timeline_months': self.timeline_months,
            'cooling_impact_celsius': self.cooling_impact_celsius,
            'maintenance_annual_usd': self.maintenance_annual,
            'co_benefits': self.co_benefits,
            'priority': self.priority.name,
            'applicability_score': round(self.applicability_score, 2),
            'cost_effectiveness': round(self.cost_effectiveness() * 1000, 4)
        }


class StrategyDatabase:
    """Database of all available mitigation strategies."""
    
    @staticmethod
    def get_all_strategies() -> List[Strategy]:
        """Return all available mitigation strategies."""
        return [
            # Vegetation Strategies
            Strategy(
                name="Street Tree Planting",
                category=StrategyCategory.VEGETATION,
                description="Plant shade trees along streets and parking areas to provide cooling through evapotranspiration and shading.",
                cost_per_sqm=45.0,
                timeline_months=6,
                cooling_impact_celsius=2.5,
                maintenance_annual=3.5,
                co_benefits=["Air quality improvement", "Stormwater management", "Carbon sequestration", "Aesthetic value"]
            ),
            Strategy(
                name="Urban Parks & Green Spaces",
                category=StrategyCategory.VEGETATION,
                description="Create or expand urban parks with trees, grass, and water features.",
                cost_per_sqm=85.0,
                timeline_months=18,
                cooling_impact_celsius=3.5,
                maintenance_annual=8.0,
                co_benefits=["Recreation", "Biodiversity", "Mental health", "Property value increase"]
            ),
            Strategy(
                name="Green Roofs (Extensive)",
                category=StrategyCategory.VEGETATION,
                description="Install lightweight green roofs with sedums and drought-resistant plants.",
                cost_per_sqm=120.0,
                timeline_months=4,
                cooling_impact_celsius=1.8,
                maintenance_annual=5.0,
                co_benefits=["Building insulation", "Stormwater retention", "Extended roof life", "Habitat creation"]
            ),
            Strategy(
                name="Green Roofs (Intensive)",
                category=StrategyCategory.VEGETATION,
                description="Install rooftop gardens with deeper soil and diverse vegetation including shrubs.",
                cost_per_sqm=250.0,
                timeline_months=6,
                cooling_impact_celsius=2.8,
                maintenance_annual=15.0,
                co_benefits=["Urban agriculture", "Recreation space", "High insulation", "Biodiversity"]
            ),
            Strategy(
                name="Vertical Gardens / Green Walls",
                category=StrategyCategory.VEGETATION,
                description="Install living walls on building facades to provide shading and evaporative cooling.",
                cost_per_sqm=350.0,
                timeline_months=3,
                cooling_impact_celsius=2.2,
                maintenance_annual=25.0,
                co_benefits=["Building insulation", "Air purification", "Noise reduction", "Aesthetic appeal"]
            ),
            Strategy(
                name="Pocket Parks",
                category=StrategyCategory.VEGETATION,
                description="Create small neighborhood parks in underutilized spaces.",
                cost_per_sqm=65.0,
                timeline_months=8,
                cooling_impact_celsius=1.5,
                maintenance_annual=6.0,
                co_benefits=["Community gathering", "Local cooling", "Walkability improvement"]
            ),
            
            # Building Strategies
            Strategy(
                name="Cool Roofs (Reflective Coating)",
                category=StrategyCategory.BUILDING,
                description="Apply high-albedo reflective coating to existing roofs.",
                cost_per_sqm=25.0,
                timeline_months=1,
                cooling_impact_celsius=1.5,
                maintenance_annual=1.5,
                co_benefits=["Energy savings", "Extended roof life", "Quick implementation"]
            ),
            Strategy(
                name="Cool Roofs (White Membrane)",
                category=StrategyCategory.BUILDING,
                description="Install white thermoplastic or EPDM roofing membrane.",
                cost_per_sqm=55.0,
                timeline_months=2,
                cooling_impact_celsius=2.0,
                maintenance_annual=2.0,
                co_benefits=["High durability", "Energy savings", "Waterproofing"]
            ),
            Strategy(
                name="Building Shading Devices",
                category=StrategyCategory.BUILDING,
                description="Install external shading like louvers, awnings, and pergolas.",
                cost_per_sqm=80.0,
                timeline_months=3,
                cooling_impact_celsius=1.2,
                maintenance_annual=4.0,
                co_benefits=["Energy savings", "Glare reduction", "Architectural interest"]
            ),
            Strategy(
                name="High-Performance Glazing",
                category=StrategyCategory.BUILDING,
                description="Replace windows with low-e, reflective, or tinted glass.",
                cost_per_sqm=200.0,
                timeline_months=4,
                cooling_impact_celsius=0.8,
                maintenance_annual=1.0,
                co_benefits=["Energy efficiency", "Comfort improvement", "Noise reduction"]
            ),
            
            # Pavement Strategies
            Strategy(
                name="Cool Pavements (Reflective)",
                category=StrategyCategory.PAVEMENT,
                description="Apply reflective coating or use light-colored materials for roads and parking.",
                cost_per_sqm=35.0,
                timeline_months=2,
                cooling_impact_celsius=1.8,
                maintenance_annual=3.0,
                co_benefits=["Reduced lighting needs", "Longer pavement life", "Safety improvement"]
            ),
            Strategy(
                name="Permeable Pavements",
                category=StrategyCategory.PAVEMENT,
                description="Install porous concrete, asphalt, or interlocking pavers.",
                cost_per_sqm=75.0,
                timeline_months=4,
                cooling_impact_celsius=2.0,
                maintenance_annual=5.0,
                co_benefits=["Stormwater infiltration", "Groundwater recharge", "Flood reduction"]
            ),
            Strategy(
                name="Grass Pavers",
                category=StrategyCategory.PAVEMENT,
                description="Use grid pavers with grass for parking areas and low-traffic zones.",
                cost_per_sqm=55.0,
                timeline_months=3,
                cooling_impact_celsius=2.5,
                maintenance_annual=8.0,
                co_benefits=["Natural appearance", "Stormwater management", "Habitat support"]
            ),
            
            # Water Features
            Strategy(
                name="Urban Water Features",
                category=StrategyCategory.WATER,
                description="Install fountains, misting systems, or reflecting pools.",
                cost_per_sqm=150.0,
                timeline_months=6,
                cooling_impact_celsius=2.0,
                maintenance_annual=20.0,
                co_benefits=["Aesthetic value", "Recreation", "Humidity increase", "Sound masking"]
            ),
            Strategy(
                name="Blue-Green Infrastructure",
                category=StrategyCategory.WATER,
                description="Integrate water features with vegetation like rain gardens and bioswales.",
                cost_per_sqm=95.0,
                timeline_months=8,
                cooling_impact_celsius=2.8,
                maintenance_annual=10.0,
                co_benefits=["Stormwater management", "Biodiversity", "Water quality", "Flood control"]
            ),
            
            # Urban Planning
            Strategy(
                name="Urban Ventilation Corridors",
                category=StrategyCategory.PLANNING,
                description="Preserve or create wind corridors aligned with prevailing winds.",
                cost_per_sqm=5.0,
                timeline_months=36,
                cooling_impact_celsius=1.5,
                maintenance_annual=0.5,
                co_benefits=["Air quality", "Natural ventilation", "Long-term impact"]
            ),
            Strategy(
                name="Building Orientation Optimization",
                category=StrategyCategory.PLANNING,
                description="Guide new development to optimize solar orientation and wind flow.",
                cost_per_sqm=2.0,
                timeline_months=24,
                cooling_impact_celsius=0.8,
                maintenance_annual=0.0,
                co_benefits=["Energy efficiency", "Daylighting", "Passive cooling"]
            ),
        ]


class MitigationRecommender:
    """
    Rule-based system for recommending UHI mitigation strategies.
    
    Analyzes urban characteristics and matches them to appropriate
    interventions based on predefined rules and thresholds.
    """
    
    # Threshold definitions
    THRESHOLDS = {
        'low_green_coverage': 0.3,      # GnPR < 30%
        'high_road_density': 0.5,       # roadDensity > 50%
        'high_building_density': 4,      # bldDensity > 4
        'low_tree_density': 0.1,        # treeDensity < 10%
        'high_asphalt_ratio': 0.15,     # asphalt_ratio > 15%
        'low_greenroof_ratio': 0.05,    # greenroof_ratio < 5%
        'high_uhi': 0.15,               # UHI > 0.15°C
    }
    
    def __init__(self, custom_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize the recommender.
        
        Args:
            custom_thresholds: Optional custom threshold values
        """
        self.thresholds = self.THRESHOLDS.copy()
        if custom_thresholds:
            self.thresholds.update(custom_thresholds)
            
        self.strategy_db = StrategyDatabase.get_all_strategies()
        self._last_recommendations: List[Strategy] = []
        
    def analyze_conditions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze urban conditions from data.
        
        Args:
            data: DataFrame with urban characteristics
            
        Returns:
            Dictionary of analyzed conditions
        """
        if data is None or data.empty:
            return self._get_default_conditions()
            
        conditions = {}
        
        # Calculate mean values for available columns
        column_mapping = {
            'green_coverage': 'GnPR',
            'road_density': 'roadDensity',
            'building_density': 'bldDensity',
            'tree_density': 'treeDensity',
            'asphalt_ratio': 'asphalt_ratio',
            'greenroof_ratio': 'greenroof_ratio',
            'park_grass_ratio': 'park_grass_ratio',
            'avg_building_height': 'avg_BH',
        }
        
        for cond_name, col_name in column_mapping.items():
            if col_name in data.columns:
                conditions[cond_name] = data[col_name].mean()
            else:
                conditions[cond_name] = 0.0
                
        # Check UHI values
        for uhi_col in ['UHI_d', 'UHI_n']:
            if uhi_col in data.columns:
                conditions[f'mean_{uhi_col}'] = data[uhi_col].mean()
                conditions[f'max_{uhi_col}'] = data[uhi_col].max()
                
        return conditions
    
    def _get_default_conditions(self) -> Dict[str, Any]:
        """Return default conditions when analysis fails."""
        return {
            'green_coverage': 0.5,
            'road_density': 0.3,
            'building_density': 3,
            'tree_density': 0.05,
            'asphalt_ratio': 0.1,
            'greenroof_ratio': 0.02,
            'mean_UHI_d': 0.15,
            'max_UHI_d': 0.18
        }
    
    def recommend(self, data: pd.DataFrame = None, 
                  conditions: Dict[str, float] = None,
                  max_recommendations: int = 10,
                  budget_limit: float = None) -> List[Strategy]:
        """
        Generate mitigation recommendations based on urban conditions.
        
        Args:
            data: DataFrame with urban characteristics
            conditions: Pre-analyzed conditions (alternative to data)
            max_recommendations: Maximum number of strategies to return
            budget_limit: Maximum cost per sqm to consider
            
        Returns:
            List of recommended Strategy objects sorted by priority
        """
        try:
            # Get conditions
            if conditions is None:
                conditions = self.analyze_conditions(data)
                
            if not conditions:
                logger.warning("No conditions available, using defaults")
                conditions = self._get_default_conditions()
                
            # Score each strategy based on conditions
            scored_strategies = []
            
            for strategy in self.strategy_db:
                score = self._calculate_applicability(strategy, conditions)
                strategy_copy = Strategy(
                    name=strategy.name,
                    category=strategy.category,
                    description=strategy.description,
                    cost_per_sqm=strategy.cost_per_sqm,
                    timeline_months=strategy.timeline_months,
                    cooling_impact_celsius=strategy.cooling_impact_celsius,
                    maintenance_annual=strategy.maintenance_annual,
                    co_benefits=strategy.co_benefits.copy(),
                    priority=self._determine_priority(score, conditions),
                    applicability_score=score
                )
                
                # Apply budget filter
                if budget_limit is None or strategy_copy.cost_per_sqm <= budget_limit:
                    scored_strategies.append(strategy_copy)
            
            # Sort by priority and score
            scored_strategies.sort(
                key=lambda s: (s.priority.value, -s.applicability_score)
            )
            
            # Get top recommendations
            recommendations = scored_strategies[:max_recommendations]
            
            # Ensure we always have default recommendations
            if not recommendations:
                recommendations = self._get_default_recommendations()
                
            self._last_recommendations = recommendations
            logger.info(f"Generated {len(recommendations)} recommendations")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return self._get_default_recommendations()
    
    def _calculate_applicability(self, strategy: Strategy, 
                                  conditions: Dict[str, float]) -> float:
        """
        Calculate how applicable a strategy is to current conditions.
        
        Returns a score from 0 to 1.
        """
        score = 0.0
        factors = 0
        
        green_coverage = conditions.get('green_coverage', 0.5)
        road_density = conditions.get('road_density', 0.3)
        building_density = conditions.get('building_density', 3)
        tree_density = conditions.get('tree_density', 0.05)
        asphalt_ratio = conditions.get('asphalt_ratio', 0.1)
        greenroof_ratio = conditions.get('greenroof_ratio', 0.02)
        
        # Vegetation strategies
        if strategy.category == StrategyCategory.VEGETATION:
            # More applicable when green coverage is low
            if green_coverage < self.thresholds['low_green_coverage']:
                score += 0.9
            else:
                score += 0.3
            factors += 1
            
            # Tree planting more applicable when tree density is low
            if 'Tree' in strategy.name and tree_density < self.thresholds['low_tree_density']:
                score += 0.8
                factors += 1
                
            # Green roofs more applicable in dense areas
            if 'Roof' in strategy.name and building_density > self.thresholds['high_building_density']:
                score += 0.7
                factors += 1
                
            # Vertical gardens for high-density areas
            if 'Vertical' in strategy.name and building_density > self.thresholds['high_building_density']:
                score += 0.85
                factors += 1
                
        # Building strategies
        elif strategy.category == StrategyCategory.BUILDING:
            # Cool roofs applicable when greenroof coverage is low
            if greenroof_ratio < self.thresholds['low_greenroof_ratio']:
                score += 0.7
                factors += 1
                
            # More applicable in dense areas
            if building_density > self.thresholds['high_building_density']:
                score += 0.6
                factors += 1
                
        # Pavement strategies
        elif strategy.category == StrategyCategory.PAVEMENT:
            # Highly applicable when road/asphalt is high
            if road_density > self.thresholds['high_road_density']:
                score += 0.9
                factors += 1
                
            if asphalt_ratio > self.thresholds['high_asphalt_ratio']:
                score += 0.85
                factors += 1
                
        # Water features
        elif strategy.category == StrategyCategory.WATER:
            # Applicable in dense areas with limited green space
            if building_density > 3 and green_coverage < 0.4:
                score += 0.6
                factors += 1
            else:
                score += 0.3
                factors += 1
                
        # Planning strategies
        elif strategy.category == StrategyCategory.PLANNING:
            # Applicable in all conditions but lower score
            score += 0.4
            factors += 1
            
        # Normalize score
        final_score = score / max(factors, 1)
        
        # Boost score based on UHI severity
        uhi = conditions.get('mean_UHI_d', 0) or conditions.get('max_UHI_d', 0)
        if uhi > self.thresholds['high_uhi']:
            final_score *= 1.2
            
        return min(final_score, 1.0)
    
    def _determine_priority(self, score: float, 
                           conditions: Dict[str, float]) -> Priority:
        """Determine priority based on score and conditions."""
        uhi = conditions.get('mean_UHI_d', 0) or conditions.get('max_UHI_d', 0)
        
        if score > 0.8 or (uhi > 0.18 and score > 0.5):
            return Priority.CRITICAL
        elif score > 0.6:
            return Priority.HIGH
        elif score > 0.4:
            return Priority.MEDIUM
        else:
            return Priority.LOW
    
    def _get_default_recommendations(self) -> List[Strategy]:
        """Return default recommendations when analysis fails."""
        logger.info("Returning default recommendations")
        defaults = [
            Strategy(
                name="Street Tree Planting",
                category=StrategyCategory.VEGETATION,
                description="Plant shade trees for immediate cooling benefits.",
                cost_per_sqm=45.0,
                timeline_months=6,
                cooling_impact_celsius=2.5,
                maintenance_annual=3.5,
                priority=Priority.HIGH,
                applicability_score=0.7
            ),
            Strategy(
                name="Cool Roofs (Reflective Coating)",
                category=StrategyCategory.BUILDING,
                description="Quick and cost-effective cooling solution.",
                cost_per_sqm=25.0,
                timeline_months=1,
                cooling_impact_celsius=1.5,
                maintenance_annual=1.5,
                priority=Priority.HIGH,
                applicability_score=0.65
            ),
            Strategy(
                name="Cool Pavements (Reflective)",
                category=StrategyCategory.PAVEMENT,
                description="Reduce heat absorption from roads and parking.",
                cost_per_sqm=35.0,
                timeline_months=2,
                cooling_impact_celsius=1.8,
                maintenance_annual=3.0,
                priority=Priority.MEDIUM,
                applicability_score=0.6
            ),
        ]
        return defaults
    
    def get_recommendations_report(self, recommendations: List[Strategy] = None) -> str:
        """Generate a formatted text report of recommendations."""
        if recommendations is None:
            recommendations = self._last_recommendations
            
        if not recommendations:
            return "No recommendations available. Please run recommend() first."
            
        lines = [
            "=" * 70,
            "UHI MITIGATION RECOMMENDATIONS REPORT",
            "=" * 70,
            ""
        ]
        
        for i, strategy in enumerate(recommendations, 1):
            lines.extend([
                f"{i}. {strategy.name}",
                f"   Category: {strategy.category.value.upper()}",
                f"   Priority: {strategy.priority.name}",
                f"   Applicability Score: {strategy.applicability_score:.2f}",
                "",
                f"   Description: {strategy.description}",
                "",
                f"   Cost: ${strategy.cost_per_sqm:.2f}/m²",
                f"   Timeline: {strategy.timeline_months} months",
                f"   Cooling Impact: {strategy.cooling_impact_celsius}°C reduction",
                f"   Annual Maintenance: ${strategy.maintenance_annual:.2f}/m²",
                "",
                f"   Co-benefits: {', '.join(strategy.co_benefits) if strategy.co_benefits else 'N/A'}",
                "-" * 70,
                ""
            ])
            
        return "\n".join(lines)
    
    def to_dataframe(self, recommendations: List[Strategy] = None) -> pd.DataFrame:
        """Convert recommendations to a DataFrame."""
        if recommendations is None:
            recommendations = self._last_recommendations
            
        if not recommendations:
            return pd.DataFrame()
            
        return pd.DataFrame([s.to_dict() for s in recommendations])
    
    @property
    def last_recommendations(self) -> List[Strategy]:
        return self._last_recommendations
