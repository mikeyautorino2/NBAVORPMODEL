# 🏀 NBA Player Value Predictor

**Advanced Machine Learning System for Predicting NBA Player Value Over Replacement Player (VORP)**

A comprehensive data science project that uses advanced feature engineering and ensemble machine learning to predict NBA player performance with **89.5% accuracy**.

## 🎯 Project Overview

This project predicts NBA player value (VORP) using advanced statistics and machine learning. The model achieved **R² = 0.895** on test data, meaning it explains nearly 90% of the variance in player value.

### 🏆 Key Results
- **Best Model**: Gradient Boosting Regressor
- **Test Accuracy**: 89.5% (R² = 0.8949)
- **Prediction Error**: ±0.23 VORP points (RMSE = 0.2315)
- **Dataset**: 547 NBA players with 39 engineered features

### 🔍 Key Insights
1. **Player Efficiency × Minutes** is the strongest predictor (54% importance)
2. **Advanced feature engineering** improved performance by ~25%
3. **Tree-based models** significantly outperform linear regression
4. **Interaction features** capture complex player value relationships

## 🚀 Quick Start

### Option 1: Terminal Analysis (Recommended)
```bash
git clone <this-repo>
cd NBA-Player-Value-Predictor
./run_analysis.sh
```

### Option 2: Interactive Jupyter Notebook
```bash
./start_jupyter.sh
# Navigate to analysis/playeranalylsis.ipynb
```

### Option 3: Python Script
```bash
source vorp-env/bin/activate
python run_vorp_model.py
```

## 📊 Model Performance Comparison

| Model | Cross-Validation R² | Validation R² | Test R² | Status |
|-------|-------------------|---------------|---------|--------|
| **Gradient Boosting** | 0.847 | **0.923** | **0.895** | 🏆 Best |
| Random Forest | 0.837 | 0.915 | - | 🥈 Strong |
| Ridge Regression | -0.448 | 0.635 | - | 📈 Decent |
| Lasso Regression | 0.685 | 0.572 | - | 📉 Limited |
| Linear Regression | -21.914 | 0.842 | - | ⚠️ Overfits |

## 🛠 Technical Architecture

### Data Pipeline
1. **Web Scraping**: Basketball-Reference.com advanced stats
2. **Advanced Preprocessing**: Position-specific imputation, KNN for missing values
3. **Feature Engineering**: 20+ engineered features including interaction terms
4. **Model Training**: 5 algorithms with hyperparameter optimization
5. **Validation**: Nested cross-validation with train/val/test splits

### Key Features Created
- **Per-36 Statistics**: AST/36, STL/36, BLK/36, TOV/36
- **Efficiency Ratios**: AST/TOV, WS per minute, Total rebound rate
- **Interaction Terms**: MP×USG%, Age×MP, PER×MP, TS%×USG%
- **Composite Metrics**: Offensive impact, Defensive impact
- **Position Encoding**: One-hot encoded positions (vs ordinal)

### Data Quality Improvements
- **Smart Missing Value Handling**: Position-specific medians + KNN imputation
- **Outlier Treatment**: Robust scaling to handle superstar/bust players
- **Data Leakage Prevention**: Removed BPM-related features
- **Multicollinearity Analysis**: VIF analysis and correlation matrices

## 📈 Feature Importance

| Rank | Feature | Importance | Description |
|------|---------|------------|-------------|
| 1 | PER_x_MP | 53.9% | Player Efficiency × Minutes Played |
| 2 | PER | 14.0% | Player Efficiency Rating |
| 3 | WS | 11.6% | Win Shares |
| 4 | WS/48 | 3.3% | Win Shares per 48 minutes |
| 5 | TS_x_USG | 2.0% | True Shooting × Usage Rate |

## 🎯 Use Cases

### For NBA Teams
- **Player Evaluation**: Predict rookie/free agent value
- **Trade Analysis**: Quantify player trade value
- **Roster Construction**: Optimize player combinations
- **Draft Preparation**: Identify undervalued prospects

### For Fantasy Basketball
- **Player Rankings**: Data-driven player valuations
- **Waiver Wire**: Find undervalued players
- **Trade Evaluation**: Fair trade analysis
- **Season Projections**: Predict player performance

### For Analytics Research
- **Feature Engineering**: Advanced stat creation techniques
- **Model Comparison**: ML algorithm benchmarking
- **Basketball Analytics**: Understanding player value drivers

## 📁 Project Structure

```
NBA-Player-Value-Predictor/
├── 🏀 Core Analysis
│   ├── analysis/playeranalylsis.ipynb    # Main Jupyter notebook
│   ├── run_vorp_model.py                 # Terminal Python script
│   └── CLAUDE.md                         # Development guidance
├── 📊 Data
│   ├── data/output.csv                   # NBA player statistics
│   └── scripts/gettingdata.py           # Web scraping script
├── ⚙️ Environment
│   ├── vorp-env/                        # Virtual environment
│   ├── requirements.txt                 # Package dependencies
│   └── sklearn-env/                     # Legacy environment
└── 🚀 Runners
    ├── run_analysis.sh                  # Terminal analysis
    ├── start_jupyter.sh                 # Jupyter launcher
    └── README.md                        # This file
```

## 🔧 Technical Requirements

### Required Packages
- **Data Science**: pandas, numpy, scipy, scikit-learn
- **Visualization**: matplotlib, seaborn  
- **Statistical Analysis**: statsmodels (VIF analysis)
- **Web Scraping**: beautifulsoup4, requests
- **Environment**: jupyter, notebook

### Installation
```bash
# Automatic setup
./run_analysis.sh  # Includes environment activation

# Manual setup
source vorp-env/bin/activate
pip install -r requirements.txt
```

## 🎓 Methodology Highlights

### Advanced Data Science Techniques
- **Sophisticated Imputation**: Position-specific + KNN imputation
- **Feature Engineering**: Domain-knowledge driven feature creation
- **Model Validation**: Nested cross-validation prevents overfitting
- **Hyperparameter Tuning**: GridSearchCV + RandomizedSearchCV
- **Ensemble Methods**: Gradient boosting with optimized parameters
- **Residual Analysis**: Comprehensive diagnostic testing

### Statistical Rigor
- **Multicollinearity Detection**: VIF analysis
- **Data Leakage Prevention**: Careful feature selection
- **Normality Testing**: Shapiro-Wilk tests on residuals
- **Homoscedasticity Analysis**: Residual variance testing
- **Cross-Validation**: 5-fold CV with proper data splitting

## 📊 Results Interpretation

### Model Reliability
- **R² = 0.895**: Explains 89.5% of VORP variance
- **RMSE = 0.23**: Typical error of ±0.23 VORP points
- **Cross-Validation**: Consistent performance across folds
- **Feature Importance**: Intuitive basketball metrics dominate

### Business Impact
- **Player Evaluation**: Predict player value within 0.23 VORP
- **Cost Efficiency**: Identify undervalued players
- **Risk Assessment**: Quantify player performance uncertainty
- **Strategic Planning**: Data-driven roster decisions

## 🔮 Future Enhancements

### Model Improvements
- **Deep Learning**: Neural networks for complex patterns
- **Ensemble Stacking**: Combine multiple model predictions
- **Time Series**: Multi-season performance trends
- **Advanced Stats**: Incorporate tracking data (SportVU)

### Feature Engineering
- **Situational Stats**: Clutch time, playoff performance  
- **Team Context**: Teammate quality, coaching effects
- **Health Metrics**: Injury history, load management
- **Draft Analytics**: College performance integration

### Deployment Options
- **Web Application**: Interactive player evaluation tool
- **API Service**: Real-time predictions via REST API
- **Mobile App**: On-the-go player analysis
- **Dashboard**: Executive summary visualizations

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Additional feature engineering ideas
- Alternative machine learning algorithms
- Data visualization improvements
- Performance optimization
- Documentation enhancements

## 📄 License

MIT License - Feel free to use this project for research, commercial applications, or learning purposes.

## 🙏 Acknowledgments

- **Basketball-Reference.com** for comprehensive NBA statistics
- **scikit-learn** team for excellent ML library
- **NBA Analytics Community** for inspiration and domain knowledge

---

**⭐ Star this repo if you found it useful for your NBA analytics projects!**

*Built with ❤️ for basketball analytics and data science*