#!/usr/bin/env python3
"""
NBA Player Value Predictor - Live Prediction Demo
Input a player's stats and get a VORP prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import pickle
import os

class VORPPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.is_trained = False
    
    def train_model(self):
        """Train the VORP prediction model"""
        print("🏀 Training VORP Prediction Model...")
        
        # Load data
        try:
            pdf = pd.read_csv("data/output.csv", sep=r'\s+')
        except FileNotFoundError:
            print("❌ Could not find data/output.csv")
            return False
        
        # Preprocessing (simplified version)
        pdf = pdf.drop(['Team'], axis=1, errors='ignore')
        
        # One-hot encode positions
        position_dummies = pd.get_dummies(pdf['Pos'], prefix='Pos')
        pdf = pd.concat([pdf, position_dummies], axis=1)
        
        # Feature engineering (key features only)
        pdf['AST_TOV_ratio'] = pdf['AST%'] / (pdf['TOV%'] + 0.01)
        pdf['WS_per_minute'] = pdf['WS'] / (pdf['MP'] + 1)
        pdf['MP_x_USG'] = pdf['MP'] * pdf['USG%']
        pdf['PER_x_MP'] = pdf['PER'] * pdf['MP']
        pdf['TS_x_USG'] = pdf['TS%'] * pdf['USG%']
        pdf['Offensive_impact'] = (pdf['USG%'] * pdf['TS%'] * pdf['AST%']) / 100
        pdf['Defensive_impact'] = (pdf['STL%'] + pdf['BLK%']) * pdf['DRB%']
        
        # Remove potential leakage
        pdf = pdf.drop(columns=['BPM', 'OBPM', 'DBPM', 'Pos', 'Awards'], errors='ignore')
        
        # Handle missing values
        numeric_cols = pdf.select_dtypes(include=[np.number]).columns
        pdf[numeric_cols] = pdf[numeric_cols].fillna(pdf[numeric_cols].median())
        
        # Prepare features and target
        self.feature_columns = [col for col in pdf.columns if col not in ['Player', 'VORP']]
        X = pdf[self.feature_columns].copy()
        y = pdf['VORP'].copy()
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train Gradient Boosting
        self.model = GradientBoostingRegressor(
            n_estimators=200, 
            max_depth=5, 
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Test performance
        X_test_scaled = self.scaler.transform(X_test)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"✅ Model trained! Test R² = {test_score:.3f}")
        self.is_trained = True
        return True
    
    def predict_player_vorp(self, player_stats):
        """Predict VORP for a player given their stats"""
        if not self.is_trained:
            print("❌ Model not trained yet!")
            return None
        
        # Create DataFrame from player stats
        player_df = pd.DataFrame([player_stats])
        
        # Add missing position columns
        for col in self.feature_columns:
            if col.startswith('Pos_') and col not in player_df.columns:
                player_df[col] = 0
        
        # Engineer features (same as training)
        player_df['AST_TOV_ratio'] = player_df['AST%'] / (player_df['TOV%'] + 0.01)
        player_df['WS_per_minute'] = player_df['WS'] / (player_df['MP'] + 1)
        player_df['MP_x_USG'] = player_df['MP'] * player_df['USG%']
        player_df['PER_x_MP'] = player_df['PER'] * player_df['MP']
        player_df['TS_x_USG'] = player_df['TS%'] * player_df['USG%']
        player_df['Offensive_impact'] = (player_df['USG%'] * player_df['TS%'] * player_df['AST%']) / 100
        player_df['Defensive_impact'] = (player_df['STL%'] + player_df['BLK%']) * player_df['DRB%']
        
        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in player_df.columns:
                player_df[col] = 0  # Default value for missing features
        
        # Select and order features
        X_player = player_df[self.feature_columns]
        
        # Handle any missing values
        X_player = X_player.fillna(0)
        
        # Scale and predict
        X_player_scaled = self.scaler.transform(X_player)
        vorp_prediction = self.model.predict(X_player_scaled)[0]
        
        return vorp_prediction

def main():
    """Demo the VORP predictor"""
    print("🏀 NBA Player Value Predictor - Live Demo")
    print("=" * 50)
    
    # Initialize and train predictor
    predictor = VORPPredictor()
    if not predictor.train_model():
        return
    
    print("\n" + "=" * 50)
    print("🎯 LIVE PREDICTION DEMO")
    print("=" * 50)
    
    # Example player stats for prediction
    print("\n📊 Example Predictions:")
    
    # Superstar example (approximating Jayson Tatum)
    superstar_stats = {
        'Age': 26, 'G': 41, 'GS': 41, 'MP': 1495,
        'PER': 22.8, 'TS%': 0.593, '3PAr': 0.510, 'FTr': 0.328,
        'ORB%': 1.8, 'DRB%': 25.7, 'TRB%': 13.7, 'AST%': 24.9,
        'STL%': 1.8, 'BLK%': 1.3, 'TOV%': 10.9, 'USG%': 30.9,
        'OWS': 3.5, 'DWS': 2.5, 'WS': 6.0, 'WS/48': 0.193,
        'Pos_PF': 1, 'Pos_C': 0, 'Pos_PG': 0, 'Pos_SF': 0, 'Pos_SG': 0
    }
    
    superstar_vorp = predictor.predict_player_vorp(superstar_stats)
    print(f"🌟 Elite Player (Tatum-like): {superstar_vorp:.2f} VORP")
    
    # Average starter example
    starter_stats = {
        'Age': 27, 'G': 70, 'GS': 70, 'MP': 2100,
        'PER': 15.0, 'TS%': 0.550, '3PAr': 0.350, 'FTr': 0.200,
        'ORB%': 4.0, 'DRB%': 15.0, 'TRB%': 9.5, 'AST%': 15.0,
        'STL%': 1.5, 'BLK%': 1.0, 'TOV%': 12.0, 'USG%': 20.0,
        'OWS': 2.0, 'DWS': 2.0, 'WS': 4.0, 'WS/48': 0.095,
        'Pos_SF': 1, 'Pos_C': 0, 'Pos_PG': 0, 'Pos_PF': 0, 'Pos_SG': 0
    }
    
    starter_vorp = predictor.predict_player_vorp(starter_stats)
    print(f"📈 Average Starter: {starter_vorp:.2f} VORP")
    
    # Bench player example
    bench_stats = {
        'Age': 24, 'G': 60, 'GS': 5, 'MP': 900,
        'PER': 12.0, 'TS%': 0.520, '3PAr': 0.400, 'FTr': 0.150,
        'ORB%': 3.0, 'DRB%': 10.0, 'TRB%': 6.5, 'AST%': 10.0,
        'STL%': 1.0, 'BLK%': 0.5, 'TOV%': 15.0, 'USG%': 18.0,
        'OWS': 0.8, 'DWS': 1.0, 'WS': 1.8, 'WS/48': 0.096,
        'Pos_SG': 1, 'Pos_C': 0, 'Pos_PG': 0, 'Pos_PF': 0, 'Pos_SF': 0
    }
    
    bench_vorp = predictor.predict_player_vorp(bench_stats)
    print(f"🪑 Bench Player: {bench_vorp:.2f} VORP")
    
    # What drives these predictions?
    print(f"\n🔍 WHAT DRIVES THESE PREDICTIONS?")
    print(f"Elite Player: High PER×MP ({22.8*1495:.0f}), great efficiency (59.3% TS)")
    print(f"Avg Starter: Decent PER×MP ({15.0*2100:.0f}), solid efficiency (55.0% TS)")  
    print(f"Bench Player: Low PER×MP ({12.0*900:.0f}), poor efficiency (52.0% TS)")
    
    print(f"\n🎯 KEY INSIGHT: VORP = f(Efficiency × Opportunity)")
    print(f"The model learned that great players need both:")
    print(f"  📊 High efficiency (PER, TS%, etc.)")
    print(f"  ⏰ Significant opportunity (Minutes Played)")
    
    print(f"\n🚀 This is PREDICTION, not just analysis!")
    print(f"Give it any player's stats → Get specific VORP forecast")

if __name__ == "__main__":
    main()