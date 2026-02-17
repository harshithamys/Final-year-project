# Machine Learning Models for UHI Prediction - Explanation & Results

## Overview

This document explains the machine learning models used in the Urban Heat Island (UHI) Analysis System and provides a detailed analysis of their performance results.

---

## 1. Models Used

### 1.1 Random Forest Regressor
**Type:** Ensemble Learning (Bagging)

**How it works:**
- Creates multiple decision trees using random subsets of data and features
- Each tree makes independent predictions
- Final prediction is the average of all tree predictions
- Reduces overfitting through randomization and averaging

**Parameters Used:**
- `n_estimators`: 100 (number of trees)
- `max_depth`: 12 (maximum tree depth)
- `random_state`: 42 (for reproducibility)

**Best For:** Complex non-linear relationships, feature importance analysis

---

### 1.2 Extra Trees Regressor (Extremely Randomized Trees)
**Type:** Ensemble Learning (Bagging with extra randomization)

**How it works:**
- Similar to Random Forest but with more randomization
- Splits are chosen randomly instead of finding optimal splits
- Faster training due to random split selection
- Often provides similar accuracy with less overfitting

**Parameters Used:**
- `n_estimators`: 100
- `max_depth`: 12

**Best For:** Large datasets, faster training, reducing variance

---

### 1.3 Gradient Boosting Regressor
**Type:** Ensemble Learning (Boosting)

**How it works:**
- Builds trees sequentially, each correcting errors of previous trees
- Uses gradient descent to minimize loss function
- Later trees focus on hard-to-predict samples
- Combines weak learners into a strong predictor

**Parameters Used:**
- `n_estimators`: 100
- `max_depth`: 6
- `learning_rate`: 0.1

**Best For:** High accuracy, handles complex patterns

---

### 1.4 Support Vector Regression (SVR)
**Type:** Kernel-based Learning

**How it works:**
- Maps data to higher-dimensional space using kernel functions
- Finds optimal hyperplane that fits data within margin (epsilon)
- RBF kernel captures non-linear relationships
- Effective with proper parameter tuning

**Parameters Used:**
- `kernel`: RBF (Radial Basis Function)
- `C`: 10 (regularization parameter)
- `epsilon`: 0.01 (margin of tolerance)

**Best For:** Small to medium datasets, non-linear patterns

---

### 1.5 K-Nearest Neighbors (KNN)
**Type:** Instance-based Learning

**How it works:**
- Stores all training data
- For prediction, finds K nearest neighbors to new point
- Averages their values for regression
- Simple but effective for local patterns

**Parameters Used:**
- `n_neighbors`: 5

**Best For:** Simple patterns, local relationships, interpretability

---

### 1.6 Ridge Regression
**Type:** Linear Regression with L2 Regularization

**How it works:**
- Linear model with penalty on large coefficients
- Prevents overfitting by shrinking coefficients
- Handles multicollinearity well
- Fast training and prediction

**Parameters Used:**
- `alpha`: 1.0 (regularization strength)

**Best For:** Linear relationships, baseline model, feature selection

---

### 1.7 Bayesian Ridge Regression
**Type:** Probabilistic Linear Model

**How it works:**
- Bayesian approach to linear regression
- Automatically tunes regularization parameters
- Provides uncertainty estimates
- Robust to outliers and noise

**Best For:** Uncertainty quantification, automatic regularization

---

### 1.8 AdaBoost Regressor
**Type:** Ensemble Learning (Adaptive Boosting)

**How it works:**
- Trains weak learners sequentially
- Each learner focuses on previously misclassified samples
- Weights samples based on prediction errors
- Combines learners with weighted voting

**Parameters Used:**
- `n_estimators`: 100
- `learning_rate`: 0.1

**Best For:** Improving weak learners, handling imbalanced importance

---

### 1.9 Neural Network (MLP Regressor)
**Type:** Deep Learning

**How it works:**
- Multiple layers of interconnected neurons
- Learns complex non-linear mappings
- Uses backpropagation for training
- Requires more data and tuning

**Parameters Used:**
- `hidden_layer_sizes`: (64, 32) - two hidden layers
- `max_iter`: 500
- `activation`: ReLU (default)

**Best For:** Very complex patterns, large datasets

---

## 2. Performance Results

### 2.1 Results Summary Table

| Rank | Model             | R² Score | RMSE (°C) | MAE (°C) |
|------|-------------------|----------|-----------|----------|
| 1    | Random Forest     | 0.9985   | 0.00192   | 0.00098  |
| 2    | Extra Trees       | 0.9983   | 0.00202   | 0.00133  |
| 3    | Gradient Boosting | 0.9980   | 0.00218   | 0.00137  |
| 4    | SVR               | 0.9889   | 0.00515   | 0.00419  |
| 5    | KNN               | 0.9789   | 0.00709   | 0.00484  |
| 6    | Bayesian Ridge    | 0.9596   | 0.00981   | 0.00749  |
| 7    | Ridge             | 0.9595   | 0.00981   | 0.00749  |
| 8    | AdaBoost          | 0.9204   | 0.01377   | 0.01073  |
| 9    | Neural Network    | 0.8180   | 0.02082   | 0.01613  |

### 2.2 Metrics Explanation

**R² Score (Coefficient of Determination)**
- Range: 0 to 1 (higher is better)
- Measures how well the model explains variance in data
- R² = 1 means perfect prediction
- R² = 0.9985 means the model explains 99.85% of variance

**RMSE (Root Mean Square Error)**
- Unit: °C (same as UHI measurements)
- Penalizes large errors more heavily
- Lower is better
- RMSE = 0.00192°C is extremely accurate

**MAE (Mean Absolute Error)**
- Unit: °C
- Average absolute difference between predicted and actual
- Lower is better
- More interpretable than RMSE

---

## 3. Results Analysis

### 3.1 Top Performers: Tree-Based Ensembles

The top 3 models are all tree-based ensemble methods:
1. **Random Forest (R² = 0.9985)**
2. **Extra Trees (R² = 0.9983)**
3. **Gradient Boosting (R² = 0.9980)**

**Why they perform best:**
- UHI has complex non-linear relationships with urban features
- Tree ensembles capture feature interactions automatically
- Bagging/boosting reduces overfitting
- Handle mixed feature types well (ratios, densities, heights)

### 3.2 Middle Performers: Kernel & Instance-Based

**SVR (R² = 0.9889)**
- RBF kernel captures non-linear patterns
- Good performance but computationally expensive
- Sensitive to parameter tuning

**KNN (R² = 0.9789)**
- Captures local patterns in urban areas
- Simple but effective
- Performance depends on neighborhood size

### 3.3 Linear Models: Ridge & Bayesian Ridge

**Ridge & Bayesian Ridge (R² ≈ 0.96)**
- Both achieve similar performance
- UHI has significant linear components
- Regularization prevents overfitting
- Fast and interpretable

### 3.4 Lower Performers

**AdaBoost (R² = 0.9204)**
- Boosting on weak learners less effective here
- May need different base estimators
- Still achieves >92% accuracy

**Neural Network (R² = 0.8180)**
- Requires more data and hyperparameter tuning
- Architecture may be too simple
- Potential for improvement with:
  - More layers/neurons
  - Better learning rate scheduling
  - More training iterations
  - Data normalization tuning

---

## 4. Key Findings

### 4.1 Model Selection Recommendations

| Use Case | Recommended Model | Reason |
|----------|-------------------|--------|
| **Production/Accuracy** | Random Forest | Best accuracy, robust |
| **Speed Priority** | Extra Trees | Fast training, similar accuracy |
| **Interpretability** | Ridge Regression | Linear, clear coefficients |
| **Uncertainty Needed** | Bayesian Ridge | Provides confidence intervals |
| **Real-time Prediction** | KNN or Ridge | Fast inference |

### 4.2 Feature Importance Insights

Based on Random Forest feature importance:
1. **GnPR (Green Plot Ratio)** - Most important
2. **asphalt_ratio** - Strong positive correlation with UHI
3. **bldDensity** - Building density affects heat retention
4. **roadDensity** - Paved surfaces increase UHI
5. **treeDensity** - Trees help reduce UHI (negative correlation)

### 4.3 Prediction Accuracy

With RMSE of 0.00192°C, the Random Forest model can:
- Predict UHI intensity with sub-0.01°C accuracy
- Reliably identify hotspot areas
- Support urban planning decisions
- Enable what-if scenario analysis

---

## 5. Generated Visualizations

### 5.1 Bar Chart (`ml_model_comparison_bar.png`)
- Compares R², RMSE, and MAE across all models
- Horizontal bars for easy comparison
- Red dashed line shows mean performance

### 5.2 Line Chart (`ml_model_comparison_line.png`)
- Top panel: Actual vs Predicted values (sorted)
- Bottom panel: Prediction errors by model
- Shows how well models track actual UHI values

### 5.3 Combined Metrics (`ml_model_comparison_combined.png`)
- Normalized metrics (0-1 scale) for comparison
- All metrics inverted so higher = better
- Easy to compare overall model quality

### 5.4 Scatter Plots (`ml_model_scatter_comparison.png`)
- Actual vs Predicted for top 6 models
- Perfect prediction shown as red dashed line
- Tighter clustering = better model

---

## 6. Conclusions

1. **Random Forest is the best model** for UHI prediction with R² = 0.9985
2. **Tree-based ensembles dominate** due to non-linear UHI relationships
3. **Linear models provide good baselines** with ~96% accuracy
4. **Neural networks need more tuning** to reach full potential
5. **All models achieve >80% accuracy**, validating the dataset quality

### Recommendations for Future Work

1. **Hyperparameter Optimization**: Use AutoML (Optuna) for better tuning
2. **Deep Learning**: Try deeper architectures with more data
3. **Ensemble Stacking**: Combine top models for marginal gains
4. **Spatial Features**: Add geographic coordinates for spatial patterns
5. **Temporal Models**: Include time-series data for diurnal patterns

---

## 7. How to Run

```bash
# Generate comparison plots
python3 generate_ml_comparison_plots.py

# Output files will be in:
# - output/plots/ml_model_comparison_bar.png
# - output/plots/ml_model_comparison_line.png
# - output/plots/ml_model_comparison_combined.png
# - output/plots/ml_model_scatter_comparison.png
# - output/plots/model_comparison_results.csv
```

---

*Generated: February 2026*
*UHI Analysis System - Final Year Project*
