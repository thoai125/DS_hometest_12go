# CTR Prediction Project — test_case_12ga


[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Linting - Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style - Black](https://img.shields.io/badge/Code%20Style-Black-000000.svg)](https://github.com/psf/black)

## Personally, I found the take-home test a bit unusual. It asked candidates to build a model using a fairly heavy dataset, which I don’t think is the best way to showcase skills and experience. The overall recruitment process also wasn’t very clear. That said, the HR team was genuinely lovely. Please keep these points in mind and consider them carefully before applying.

A comprehensive machine learning pipeline for **Booking-Through Rate (CTR) prediction** featuring advanced feature engineering, multiple modeling approaches, hyperparameter optimization, model interpretability, and **business impact analysis** with ROI quantification.

## 🎯 Project Overview

This project implements an end-to-end ML pipeline for predicting booking-through rates in digital advertising, addressing real-world challenges including high-cardinality categorical features, severe class imbalance, and delivering measurable business value through optimized ad targeting.

### Key Features
- **Advanced Feature Engineering**: WOE encoding, Cramér's V correlation analysis, recursive feature elimination
- **Multiple Model Approaches**: Logistic Regression baseline, CatBoost with extensive tuning
- **Business Impact Analysis**: ROI quantification, threshold optimization, cost-benefit analysis
- **Model Interpretability**: SHAP analysis, feature importance, business insights
- **Production Pipeline**: End-to-end deployment with monitoring capabilities

### Business Value Delivered
- **Profit Optimization**: Threshold-based targeting for maximum ROI
- **Cost Reduction**: Up to 85% reduction in wasted ad spend
- **Targeting Efficiency**: 3-4x improvement in booking-through rates vs baseline
- **Scalable Framework**: Production-ready infrastructure for real-time decisions

## 📁 Project Structure

```
test_case_12ga/
├── README.md                                 # Project documentation
├── pyproject.toml                            # Poetry dependencies
├── notebooks/                               # Complete ML pipeline
│   ├── 1_data_collection_optimization.ipynb  # Data loading & optimization
│   ├── 2_eda.ipynb                           # Exploratory Data Analysis
│   ├── 3_train_test_splitting.ipynb          # Stratified data splitting
│   ├── 4_lr.ipynb                            # Logistic Regression + WOE
│   ├── 5_1_catboost.ipynb                    # CatBoost baseline
│   ├── 5_2_catboost_fine_tunning.ipynb       # Hyperparameter optimization
│   ├── 5_3_optimized_catboost.ipynb          # Final tuned CatBoost model
│   ├── 6_model_comparison.ipynb              # Performance benchmarking
│   ├── 7_model_explaination.ipynb            # SHAP interpretability
│   ├── 8_model_deployment.ipynb              # Production deployment
│   ├── 9_business_impact.ipynb               # 💰 ROI & business analysis
│  
├── test_case_12ga/
│   ├── scripts/
│   │   └── data_process.py                   # Automated data processing
│   └── utils/                                # Core utility modules
│       ├── constants.py                      # Paths & configuration
│       ├── cramers_v1.py                     # Categorical correlation
│       ├── data_processing.py                # Data preprocessing
│       ├── eda.py                            # EDA visualizations
│       ├── feature_evaluation.py             # Feature selection
│       ├── lr_recursive_elimination.py       # Feature elimination
│       ├── model_comparison.py               # Model benchmarking
│       ├── model_training.py                 # Training pipelines
│       ├── model_validation.py               # Evaluation metrics
│       ├── performance.py                    # Performance monitoring
│       ├── settings.py                       # Configuration
│       └── woe_encoding.py                   # Weight of Evidence
└── data/                                    # Data storage (not tracked)
    ├── raw/                                 # Original datasets
    ├── compressed/                          # Optimized storage
    ├── processed/                           # Train/val/test splits
    └── models/                              # Saved model artifacts
```

## 🛠 Installation

### Prerequisites
- Python 3.8+
- Poetry (recommended) or pip
- 16GB+ RAM recommended for full dataset processing

### Using Poetry (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd test_case_12ga

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

### Using pip
```bash
# Clone and setup virtual environment
git clone <repository-url>
cd test_case_12ga
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn catboost matplotlib seaborn
pip install optbinning shap jupyter fastparquet
```

## 🚀 Usage Guide

### Complete Pipeline Execution

Execute notebooks in the recommended sequence for full ML pipeline:

```bash
# Launch Jupyter environment
poetry run jupyter lab  # or: jupyter lab

# Recommended execution order:
# 1. notebooks/1_data_collection_optimization.ipynb  → Data loading & optimization
# 2. notebooks/2_eda.ipynb                           → Feature analysis & correlation
# 3. notebooks/3_train_test_splitting.ipynb          → Stratified data splitting
# 4. notebooks/4_lr.ipynb                            → Logistic regression baseline
# 5. notebooks/5_1_catboost.ipynb                    → CatBoost initial model
# 6. notebooks/5_2_catboost_fine_tunning.ipynb       → Hyperparameter tuning
# 7. notebooks/5_3_optimized_catboost.ipynb          → Final optimized model
# 8. notebooks/6_model_comparison.ipynb              → Model performance comparison
# 9. notebooks/9_business_impact.ipynb               → 💰 Business ROI analysis
```

### Key Analysis Highlights

#### 1. Feature Engineering Excellence
```python
# notebooks/2_eda.ipynb
# - Cramér's V correlation analysis eliminates 35% redundant features
# - WOE encoding for optimal categorical feature transformation
# - Time-based feature validation and optimization
```

#### 2. Advanced Model Development
```python
# notebooks/4_lr.ipynb & 5_x_catboost.ipynb
# - Recursive feature elimination for optimal feature sets
# - Hyperparameter tuning with cross-validation
# - Class imbalance handling strategies
```

#### 3. Business Impact Analysis
```python
# notebooks/9_business_impact.ipynb
# - ROI calculation with cost/revenue optimization
# - Threshold optimization for maximum profit
# - A/B testing framework for model deployment
```

## 📊 Dataset & Business Context

### Dataset Characteristics
- **Domain**: Digital Advertising Booking Prediction
- **Type**: Binary Classification (Booking: 1, No-Booking: 0)
- **Scale**: Large dataset with millions of impressions
- **Challenge**: Severe class imbalance

### Business Model
- **Cost per ad impression**: $2.00
- **Revenue per booking**: $12.00  
- **Net profit per successful booking**: $10.00
- **Optimization goal**: Maximize total campaign profit

### Feature Portfolio
| Category | Count | Description | Examples |
|----------|-------|-------------|-----------|
| **Categorical** | 13* | Site, app, device attributes | `site_id`, `app_category`, `device_type` |
| **High-Cardinality** | 2 | Unique identifiers | `device_id` (2.7M), `device_ip` (6.7M) |
| **Numerical** | 4* | Time-based cyclical patterns | `hour_day_sin/cos`, `hour_hour_sin/cos` |
| **Boolean** | 1 | Binary behavioral indicators | `is_weekend` |

*After feature engineering optimization (reduced from 20 categorical, 8 numerical)

### Data Quality Assessment
- ✅ **Perfect Completeness**: Zero missing values across all features
- ⚠️ **Class Imbalance**: Severe imbalance requiring specialized handling
- ✅ **Rich Information**: Comprehensive user, content, and temporal features

## 🔬 Methodology & Technical Approach

### 1. Advanced Feature Engineering

#### Correlation Analysis & Feature Selection
- **Cramér's V Analysis**: Systematic detection of redundant categorical features
- **Information Value Filtering**: Remove low-predictive features (IV < 0.01)
- **Recursive Elimination**: Optimize feature sets through systematic removal
- **Result**: 75% reduction in feature space while maintaining predictive power

#### Categorical Feature Transformation
```python
# WOE Encoding with Optimal Binning
from test_case_12ga.utils.woe_encoding import perform_woe_encoding_categorical

# - Optimal binning for categorical features
# - Information value calculation for feature ranking
# - Cross-validation to prevent overfitting
```

#### Time-Based Feature Engineering
- **Cyclical Encoding**: Sin/cos transformation for temporal patterns
- **Feature Validation**: Remove constant features (month/minute components)
- **Pattern Recognition**: Capture day-of-week and hour-of-day behaviors

### 2. Model Development Pipeline

#### Logistic Regression (Interpretable Baseline)
- **Regularization**: L2 penalty (C=0.1) for stability
- **Feature Selection**: Recursive elimination to 9 optimal features
- **Performance**: AUC 0.733, highly interpretable coefficients

#### CatBoost (Advanced Ensemble)
- **Native Categorical Handling**: No manual encoding required
- **Hyperparameter Optimization**: Extensive grid search with CV
- **Custom Loss Functions**: Optimized for imbalanced classification
- **Performance**: AUC 0.7605, superior handling of categorical features

#### Model Selection Criteria
- **Predictive Performance**: ROC-AUC, PR-AUC metrics
- **Business Impact**: Profit optimization through threshold tuning
- **Operational Efficiency**: Training time, inference speed, interpretability

### 3. Business Impact Framework

#### Profit Optimization
```python
# Threshold optimization for maximum ROI
# Profit = (True Positives × $10) - (False Positives × $2)
# Find threshold that maximizes total campaign profit
```

#### Cost-Benefit Analysis
- **Baseline Strategy**: Show ads to all users (no targeting)
- **Model Strategy**: Targeted ads based on predicted probabilities
- **Optimization Goal**: Maximize net profit while minimizing wasted spend

## 📈 Results & Business Impact

### Model Performance Comparison

| Model | AUC-ROC | PR-AUC | Features | Training Time | Business Use |
|-------|---------|--------|----------|---------------|--------------|
| **Logistic Regression** | 0.7330 | 0.3437 | 9 | Fast | Interpretable baseline |
| **CatBoost (Baseline)** | 0.766 |0.3443 | 18 | Medium | Production candidate |
| **CatBoost (Optimized)** | 0.7605 | 0.3933 | 7 | Medium | Best performance |

### Business Value Delivered

#### Financial Impact Analysis (Test Data)
```
💰 BUSINESS IMPACT SUMMARY (Based on Test Set)
================================================
Baseline Strategy (No Model):
  • Total cost: $2.00 × all users
  • Revenue: $12.00 × actual booking only
  • Net profit: Typically negative due to low CTR

Optimized Model Strategy:
  • Targeted advertising: Show ads only to high-probability users
  • Cost reduction: Up to 85% fewer ad impressions
  • Efficiency gain: 3-4x higher booking-through rate
  • Net profit: Significant positive ROI
```

#### Key Business Metrics
- **Cost Efficiency**: 75-85% reduction in wasted ad spend
- **Targeting Precision**: 3-4x improvement in effective CTR
- **ROI Enhancement**: Positive profit vs negative baseline
- **Scalability**: Framework supports millions of real-time decisions

### Technical Achievements
- **Feature Engineering**: 75% feature reduction with maintained performance
- **Model Stability**: Consistent performance across train/validation/test splits
- **Production Readiness**: Optimized inference pipeline for real-time decisions
- **Interpretability**: SHAP analysis for business stakeholder communication

## 🔍 Advanced Capabilities

### Model Interpretability & Explainability
```python
# notebooks/7_model_explaination.ipynb
import shap

# - Global feature importance analysis
# - Individual prediction explanations
# - Business stakeholder communication tools
# - Model behavior validation and debugging
```

### Hyperparameter Optimization
```python
# notebooks/5_2_catboost_fine_tunning.ipynb
# - Comprehensive parameter grid search
# - Cross-validation with stratified splits
# - Early stopping for optimal model complexity
# - Custom evaluation metrics for business objectives
```

### Business Impact Analysis
```python
# notebooks/9_business_impact.ipynb
# - Threshold optimization for profit maximization
# - Cost-benefit analysis with realistic business assumptions
# - A/B testing framework design
# - ROI quantification and sensitivity analysis
```

## 🛡️ Production & Deployment

### Model Deployment Pipeline
- **Model Serialization**: Optimized storage with joblib/pickle
- **Inference API**: Real-time prediction capabilities
- **Batch Processing**: Support for large-scale scoring
- **Performance Monitoring**: Drift detection and alerting

### Scalability Features
- **Memory Optimization**: Efficient data type usage (50%+ memory reduction)
- **Feature Pipeline**: Automated preprocessing for new data
- **Monitoring Framework**: Performance tracking and model health metrics
- **A/B Testing**: Framework for gradual model rollout and validation

### Quality Assurance
- **Code Quality**: Black formatting, Ruff linting
- **Version Control**: Git-based model versioning
- **Documentation**: Comprehensive notebooks and code comments
- **Testing**: Model validation and business logic verification

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
poetry install --with dev

# Setup pre-commit hooks for quality gates
pre-commit install

# Run quality checks
poetry run ruff check .      # Linting
poetry run black .           # Code formatting  
poetry run pytest           # Run tests
```

### Code Standards
- **Formatting**: Black for consistent code style
- **Linting**: Ruff for fast Python quality checks
- **Type Hints**: Encouraged for better documentation
- **Documentation**: Docstrings for all functions and classes

## 📚 Dependencies & Technical Stack

### Core ML & Data Science
```toml
# Core dependencies
pandas = "^2.0"           # Data manipulation
numpy = "^1.24"           # Numerical computing
scikit-learn = "^1.3"     # ML algorithms & metrics
catboost = "^1.2"         # Gradient boosting
optbinning = "^0.17"      # WOE encoding & binning
```

## 🔮 Future Enhancements

### Advanced Modeling
- [ ] **Ensemble Methods**: Stacking and blending multiple models
- [ ] **Deep Learning**: TabNet or neural network approaches
- [ ] **AutoML Integration**: Automated model selection and tuning
- [ ] **Online Learning**: Continuous model updates with new data

### Business Intelligence
- [ ] **Real-time Dashboards**: Business impact monitoring
- [ ] **Segment Analysis**: Performance across user demographics
- [ ] **Competitive Analysis**: Market benchmark comparisons
- [ ] **Attribution Modeling**: Multi-touch attribution analysis

### Production Scaling
- [ ] **MLOps Pipeline**: Automated training and deployment
- [ ] **Feature Store**: Centralized feature management
- [ ] **Distributed Processing**: Spark/Dask for larger datasets
- [ ] **Edge Deployment**: Mobile/edge device optimization

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **CatBoost Team**: Outstanding gradient boosting with categorical feature support
- **optbinning Library**: Robust Weight of Evidence encoding implementation
- **SHAP Project**: Comprehensive model interpretability framework
- **scikit-learn Community**: Foundational machine learning toolkit
- **Jupyter Project**: Interactive development environment excellence
