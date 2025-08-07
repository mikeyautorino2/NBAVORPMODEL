#!/usr/bin/env python3
"""
NBA VORP Prediction Model - Terminal Runner
This script runs the full NBA VORP analysis pipeline from the command line.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the NBA data"""
    print("🏀 Loading NBA VORP Data...")
    
    try:
        # Try relative path first
        pdf = pd.read_csv("data/output.csv", sep='\s+')
        print("✅ Data loaded from data/output.csv")
    except FileNotFoundError:
        print("❌ Could not find data/output.csv")
        print("Please ensure the data file exists in the data/ directory")
        return None
    
    print(f"📊 Dataset shape: {pdf.shape}")
    return pdf

def preprocess_data(pdf):
    """Advanced data preprocessing"""
    print("\n🔧 Preprocessing Data...")
    
    # Drop team column
    pdf = pdf.drop(['Team'], axis=1, errors='ignore')
    
    # One-hot encode positions
    position_dummies = pd.get_dummies(pdf['Pos'], prefix='Pos')
    pdf = pd.concat([pdf, position_dummies], axis=1)
    
    # Advanced missing value treatment
    numeric_cols = pdf.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in ['Player']]
    
    # Position-specific imputation for low-minute players
    low_min_players = pdf['MP'] < 200
    high_min_players = pdf['MP'] >= 200
    pdf_imputed = pdf.copy()
    
    for pos in pdf['Pos'].unique():
        pos_mask = pdf['Pos'] == pos
        low_min_pos = pos_mask & low_min_players
        
        if low_min_pos.sum() > 0:
            for col in feature_cols:
                if pdf[col].isnull().any():
                    pos_median = pdf[pos_mask & ~pdf[col].isnull()][col].median()
                    if not np.isnan(pos_median):
                        pdf_imputed.loc[low_min_pos, col] = pdf_imputed.loc[low_min_pos, col].fillna(pos_median)
    
    # KNN imputation for regular players
    high_min_data = pdf_imputed[high_min_players].copy()
    if len(high_min_data) > 5:
        knn_features = [col for col in feature_cols if col not in ['Awards', 'VORP'] and not col.startswith('Pos_')]
        if len(knn_features) > 0:
            imputer = KNNImputer(n_neighbors=5, weights='distance')
            high_min_data[knn_features] = imputer.fit_transform(high_min_data[knn_features])
            pdf_imputed.update(high_min_data)
    
    # Final cleanup
    remaining_numeric = pdf_imputed.select_dtypes(include=[np.number])
    for col in remaining_numeric.columns:
        if pdf_imputed[col].isnull().any():
            median_val = pdf_imputed[col].median()
            pdf_imputed[col].fillna(median_val, inplace=True)
    
    pdf_imputed['Awards'].fillna('None', inplace=True)
    
    print(f"✅ Missing values handled: {pdf_imputed.isnull().sum().sum()} remaining")
    return pdf_imputed

def engineer_features(pdf):
    """Advanced feature engineering"""
    print("\n⚙️ Engineering Features...")
    
    pdf_engineered = pdf.copy()
    
    # Per-36 minute statistics
    pdf_engineered['AST_per36'] = (pdf_engineered['AST%'] * pdf_engineered['MP']) / 36
    pdf_engineered['STL_per36'] = (pdf_engineered['STL%'] * pdf_engineered['MP']) / 36
    pdf_engineered['BLK_per36'] = (pdf_engineered['BLK%'] * pdf_engineered['MP']) / 36
    pdf_engineered['TOV_per36'] = (pdf_engineered['TOV%'] * pdf_engineered['MP']) / 36
    
    # Efficiency ratios
    pdf_engineered['AST_TOV_ratio'] = pdf_engineered['AST%'] / (pdf_engineered['TOV%'] + 0.01)
    pdf_engineered['WS_per_minute'] = pdf_engineered['WS'] / (pdf_engineered['MP'] + 1)
    pdf_engineered['Total_rebound_rate'] = pdf_engineered['ORB%'] + pdf_engineered['DRB%']
    
    # Interaction terms
    pdf_engineered['MP_x_USG'] = pdf_engineered['MP'] * pdf_engineered['USG%']
    pdf_engineered['Age_x_MP'] = pdf_engineered['Age'] * pdf_engineered['MP']
    pdf_engineered['PER_x_MP'] = pdf_engineered['PER'] * pdf_engineered['MP']
    pdf_engineered['TS_x_USG'] = pdf_engineered['TS%'] * pdf_engineered['USG%']
    
    # Advanced composites
    pdf_engineered['Offensive_impact'] = (pdf_engineered['USG%'] * pdf_engineered['TS%'] * pdf_engineered['AST%']) / 100
    pdf_engineered['Defensive_impact'] = (pdf_engineered['STL%'] + pdf_engineered['BLK%']) * pdf_engineered['DRB%']
    
    # Remove potential data leakage
    leakage_features = ['BPM', 'OBPM', 'DBPM']
    pdf_engineered = pdf_engineered.drop(columns=leakage_features, errors='ignore')
    
    # Clean up
    pdf_engineered = pdf_engineered.drop(columns=['Pos', 'Awards'], errors='ignore')
    
    print(f"✅ Feature engineering complete: {pdf_engineered.shape[1]} total features")
    return pdf_engineered

def train_models(X_train, X_val, y_train, y_val):
    """Train and compare multiple models"""
    print("\n🤖 Training Models...")
    
    models = {
        'LinearRegression': LinearRegression(),
        'Lasso': Lasso(random_state=42),
        'Ridge': Ridge(random_state=42), 
        'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(random_state=42)
    }
    
    param_grids = {
        'Lasso': {'alpha': np.logspace(-4, 1, 10)},
        'Ridge': {'alpha': np.logspace(-4, 2, 10)},
        'RandomForest': {'n_estimators': [100, 200], 'max_depth': [10, None]},
        'GradientBoosting': {'n_estimators': [100, 200], 'max_depth': [3, 5]}
    }
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    results = {}
    cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        if name in param_grids:
            if name in ['RandomForest', 'GradientBoosting']:
                search = RandomizedSearchCV(model, param_grids[name], n_iter=10, cv=cv_strategy, scoring='r2', random_state=42)
            else:
                search = GridSearchCV(model, param_grids[name], cv=cv_strategy, scoring='r2')
            
            search.fit(X_train_scaled, y_train)
            best_model = search.best_estimator_
            cv_score = search.best_score_
        else:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_strategy, scoring='r2')
            cv_score = cv_scores.mean()
            best_model = model
            best_model.fit(X_train_scaled, y_train)
        
        # Validate
        val_pred = best_model.predict(X_val_scaled)
        val_r2 = r2_score(y_val, val_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        
        results[name] = {
            'model': best_model,
            'cv_r2': cv_score,
            'val_r2': val_r2,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'scaler': scaler
        }
        
        print(f"  CV R²: {cv_score:.4f}, Val R²: {val_r2:.4f}")
    
    return results

def main():
    """Main execution function"""
    print("🏀 NBA VORP Prediction Model")
    print("=" * 50)
    
    # Load data
    pdf = load_and_prepare_data()
    if pdf is None:
        return
    
    # Preprocess
    pdf = preprocess_data(pdf)
    
    # Engineer features
    pdf = engineer_features(pdf)
    
    # Prepare for modeling
    feature_columns = [col for col in pdf.columns if col not in ['Player', 'VORP']]
    X = pdf[feature_columns].copy()
    y = pdf['VORP'].copy()
    
    # Fill any remaining NaNs
    X = X.fillna(X.median())
    
    print(f"\n📈 Final dataset: {X.shape[0]} players, {X.shape[1]} features")
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    
    print(f"Training: {X_train.shape[0]}, Validation: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Train models
    results = train_models(X_train, X_val, y_train, y_val)
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['val_r2'])
    best_result = results[best_model_name]
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"Validation R²: {best_result['val_r2']:.4f}")
    print(f"Validation RMSE: {best_result['val_rmse']:.4f}")
    
    # Final test
    X_test_scaled = best_result['scaler'].transform(X_test)
    test_pred = best_result['model'].predict(X_test_scaled)
    test_r2 = r2_score(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    print(f"\n🎯 Final Test Results:")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    
    # Feature importance (if available)
    if hasattr(best_result['model'], 'feature_importances_'):
        importances = best_result['model'].feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\n📊 Top 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
    
    print(f"\n✅ Analysis Complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()