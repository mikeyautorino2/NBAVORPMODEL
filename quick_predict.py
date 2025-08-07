#!/usr/bin/env python3
"""
NBA Player VORP Predictor - Quick Command Line Tool
Usage: python quick_predict.py "Player Name"
"""

import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def quick_predict(player_name):
    """Quick prediction for a player"""
    
    # Load data
    try:
        pdf = pd.read_csv("data/output.csv", sep=r'\s+')
    except FileNotFoundError:
        print("❌ Error: Could not find data/output.csv")
        print("Make sure you're in the NBA-Player-Value-Predictor directory")
        return None
    
    # Find player
    matches = pdf[pdf['Player'].str.contains(player_name, case=False, na=False)]
    if len(matches) == 0:
        # Try partial match
        words = player_name.split()
        for word in words:
            if len(word) > 2:
                matches = pdf[pdf['Player'].str.contains(word, case=False, na=False)]
                if len(matches) > 0:
                    break
    
    if len(matches) == 0:
        print(f"❌ Player '{player_name}' not found")
        print("💡 Try: 'Tatum', 'LeBron', 'Curry', 'Jokic', etc.")
        return None
    
    player = matches.iloc[0]  # Use first match
    
    # Quick model training (simplified)
    pdf_model = pdf.copy()
    pdf_model = pdf_model.drop(['Team'], axis=1, errors='ignore')
    
    # Basic feature engineering
    position_dummies = pd.get_dummies(pdf_model['Pos'], prefix='Pos')
    pdf_model = pd.concat([pdf_model, position_dummies], axis=1)
    
    pdf_model['PER_x_MP'] = pdf_model['PER'] * pdf_model['MP']
    pdf_model['TS_x_USG'] = pdf_model['TS%'] * pdf_model['USG%']
    pdf_model = pdf_model.drop(columns=['BPM', 'OBPM', 'DBPM', 'Pos', 'Awards'], errors='ignore')
    
    # Handle missing values
    numeric_cols = pdf_model.select_dtypes(include=[np.number]).columns
    pdf_model[numeric_cols] = pdf_model[numeric_cols].fillna(pdf_model[numeric_cols].median())
    
    # Prepare for training
    feature_columns = [col for col in pdf_model.columns if col not in ['Player', 'VORP']]
    X = pdf_model[feature_columns]
    y = pdf_model['VORP']
    
    # Quick train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Prepare player for prediction
    player_data = player.copy()
    for position in ['C', 'PF', 'SF', 'SG', 'PG']:
        player_data[f'Pos_{position}'] = 1 if player_data.get('Pos') == position else 0
    
    player_data['PER_x_MP'] = player_data['PER'] * player_data['MP']
    player_data['TS_x_USG'] = player_data['TS%'] * player_data['USG%']
    
    # Create prediction input
    player_df = pd.DataFrame([player_data])
    for col in feature_columns:
        if col not in player_df.columns:
            player_df[col] = 0
    
    X_player = player_df[feature_columns].fillna(0)
    X_player_scaled = scaler.transform(X_player)
    prediction = model.predict(X_player_scaled)[0]
    
    # Display result
    actual_vorp = player.get('VORP', 0)
    error = abs(prediction - actual_vorp) if actual_vorp else 0
    
    print(f"🏀 {player['Player']} VORP Prediction")
    print(f"{'='*50}")
    print(f"📊 Stats: {player['Age']:.0f}y, {player['Pos']}, {player['G']:.0f}G, {player['MP']:.0f}MP")
    print(f"⭐ PER: {player['PER']:.1f} | TS%: {player['TS%']:.1%} | USG: {player['USG%']:.1f}%")
    print(f"")
    print(f"🎯 Predicted VORP: {prediction:.2f}")
    if actual_vorp:
        print(f"📈 Actual VORP: {actual_vorp:.2f}")
        print(f"📏 Error: ±{error:.2f}")
    
    # Quick interpretation
    if prediction > 2.0:
        print(f"⭐ Elite All-Star level player!")
    elif prediction > 1.0:
        print(f"📈 Solid starter, good value contributor")
    elif prediction > 0:
        print(f"📊 Decent role player")
    else:
        print(f"📉 Below replacement level")
    
    return prediction

def main():
    if len(sys.argv) != 2:
        print("🏀 NBA Player VORP Predictor - Quick Tool")
        print("")
        print("Usage:")
        print("  python quick_predict.py \"Player Name\"")
        print("")
        print("Examples:")
        print("  python quick_predict.py \"Tatum\"")
        print("  python quick_predict.py \"LeBron James\"")
        print("  python quick_predict.py \"Curry\"")
        print("")
        return
    
    player_name = sys.argv[1]
    quick_predict(player_name)

if __name__ == "__main__":
    main()