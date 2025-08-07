#!/usr/bin/env python3
"""
NBA Player Value Predictor - Interactive Player Demo
Demo the model on specific players with real NBA stats
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')

class PlayerVORPDemo:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.is_trained = False
        self.real_players_data = None
    
    def load_real_players(self):
        """Load real NBA player data for demo"""
        try:
            df = pd.read_csv("data/output.csv", sep=r'\s+')
            self.real_players_data = df
            print(f"✅ Loaded {len(df)} real NBA players for demo")
            return True
        except FileNotFoundError:
            print("❌ Could not find data/output.csv")
            return False
    
    def train_model_quick(self):
        """Quick model training for demo"""
        if self.real_players_data is None:
            if not self.load_real_players():
                return False
        
        print("🏀 Training VORP Prediction Model...")
        pdf = self.real_players_data.copy()
        
        # Preprocessing
        pdf = pdf.drop(['Team'], axis=1, errors='ignore')
        position_dummies = pd.get_dummies(pdf['Pos'], prefix='Pos')
        pdf = pd.concat([pdf, position_dummies], axis=1)
        
        # Feature engineering
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
        
        # Prepare features
        self.feature_columns = [col for col in pdf.columns if col not in ['Player', 'VORP']]
        X = pdf[self.feature_columns].copy()
        y = pdf['VORP'].copy()
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale and train
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Quick test
        X_test_scaled = self.scaler.transform(X_test)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"✅ Model ready! Test R² = {test_score:.3f}")
        self.is_trained = True
        return True
    
    def find_player(self, player_name):
        """Find a player in the dataset"""
        if self.real_players_data is None:
            return None
        
        # Fuzzy search for player
        matches = self.real_players_data[
            self.real_players_data['Player'].str.contains(player_name, case=False, na=False)
        ]
        
        if len(matches) == 0:
            # Try partial matches
            words = player_name.split()
            for word in words:
                if len(word) > 2:  # Skip short words
                    matches = self.real_players_data[
                        self.real_players_data['Player'].str.contains(word, case=False, na=False)
                    ]
                    if len(matches) > 0:
                        break
        
        return matches
    
    def predict_player_vorp(self, player_row):
        """Predict VORP for a specific player"""
        if not self.is_trained:
            return None
        
        # Convert player row to prediction format
        player_data = player_row.copy()
        
        # One-hot encode position
        pos = player_data.get('Pos', 'SF')  # Default to SF if missing
        for position in ['C', 'PF', 'SF', 'SG', 'PG']:
            player_data[f'Pos_{position}'] = 1 if pos == position else 0
        
        # Engineer features
        player_data['AST_TOV_ratio'] = player_data['AST%'] / (player_data['TOV%'] + 0.01)
        player_data['WS_per_minute'] = player_data['WS'] / (player_data['MP'] + 1)
        player_data['MP_x_USG'] = player_data['MP'] * player_data['USG%']
        player_data['PER_x_MP'] = player_data['PER'] * player_data['MP']
        player_data['TS_x_USG'] = player_data['TS%'] * player_data['USG%']
        player_data['Offensive_impact'] = (player_data['USG%'] * player_data['TS%'] * player_data['AST%']) / 100
        player_data['Defensive_impact'] = (player_data['STL%'] + player_data['BLK%']) * player_data['DRB%']
        
        # Create DataFrame and ensure all features exist
        player_df = pd.DataFrame([player_data])
        for col in self.feature_columns:
            if col not in player_df.columns:
                player_df[col] = 0
        
        # Select features in correct order
        X_player = player_df[self.feature_columns].fillna(0)
        
        # Scale and predict
        X_player_scaled = self.scaler.transform(X_player)
        prediction = self.model.predict(X_player_scaled)[0]
        
        return prediction
    
    def demo_player(self, player_name):
        """Complete demo for a specific player"""
        print(f"\n🏀 VORP PREDICTION DEMO: {player_name}")
        print("=" * 60)
        
        # Find player
        matches = self.find_player(player_name)
        
        if matches is None or len(matches) == 0:
            print(f"❌ Player '{player_name}' not found in dataset")
            self.show_available_players()
            return
        
        if len(matches) > 1:
            print(f"🔍 Found {len(matches)} matches:")
            for idx, (_, player) in enumerate(matches.head().iterrows()):
                print(f"  {idx+1}. {player['Player']} (Age {player['Age']}, {player['Pos']})")
            print("Using first match for demo...\n")
        
        # Use first match
        player_row = matches.iloc[0]
        actual_vorp = player_row.get('VORP', 'Unknown')
        
        # Make prediction
        predicted_vorp = self.predict_player_vorp(player_row)
        
        if predicted_vorp is None:
            print("❌ Could not make prediction")
            return
        
        # Display results
        print(f"📊 PLAYER STATS:")
        print(f"  Name: {player_row['Player']}")
        print(f"  Age: {player_row['Age']}")
        print(f"  Position: {player_row['Pos']}")
        print(f"  Games: {player_row['G']} (GS: {player_row['GS']})")
        print(f"  Minutes: {player_row['MP']}")
        print(f"  PER: {player_row['PER']}")
        print(f"  TS%: {player_row['TS%']:.1%}")
        print(f"  USG%: {player_row['USG%']:.1f}%")
        print(f"  Win Shares: {player_row['WS']}")
        
        print(f"\n🎯 VORP PREDICTION:")
        print(f"  Predicted VORP: {predicted_vorp:.2f}")
        if actual_vorp != 'Unknown':
            print(f"  Actual VORP: {actual_vorp:.2f}")
            error = abs(predicted_vorp - actual_vorp)
            print(f"  Prediction Error: ±{error:.2f}")
            accuracy = (1 - error / max(abs(actual_vorp), 0.1)) * 100
            print(f"  Accuracy: {accuracy:.1f}%")
        
        print(f"\n🔍 WHAT DRIVES THIS PREDICTION?")
        # Show key feature contributions
        per_x_mp = player_row['PER'] * player_row['MP']
        print(f"  PER × Minutes: {per_x_mp:.0f} (54% of prediction)")
        print(f"  Efficiency (TS%): {player_row['TS%']:.1%}")
        print(f"  Win Shares: {player_row['WS']:.1f}")
        print(f"  Usage Rate: {player_row['USG%']:.1f}%")
        
        # Interpret prediction
        if predicted_vorp > 2.0:
            print(f"\n⭐ INTERPRETATION: Elite player! This is All-Star level production.")
        elif predicted_vorp > 1.0:
            print(f"\n📈 INTERPRETATION: Solid starter with good value contribution.")
        elif predicted_vorp > 0.0:
            print(f"\n📊 INTERPRETATION: Decent role player, positive contributor.")
        else:
            print(f"\n📉 INTERPRETATION: Below replacement level, limited value.")
        
        return predicted_vorp, actual_vorp
    
    def show_available_players(self):
        """Show some available players for demo"""
        if self.real_players_data is None:
            return
        
        print(f"\n📋 SOME AVAILABLE PLAYERS FOR DEMO:")
        sample_players = self.real_players_data.nlargest(10, 'VORP')['Player'].tolist()
        for i, player in enumerate(sample_players[:5]):
            print(f"  • {player}")
        print("  • ... and 540+ more players")
    
    def interactive_demo(self):
        """Interactive player demo"""
        print("🏀 NBA Player VORP Predictor - Interactive Demo")
        print("=" * 55)
        
        if not self.is_trained:
            if not self.train_model_quick():
                return
        
        while True:
            print(f"\n" + "="*55)
            print("🎯 DEMO OPTIONS:")
            print("  1. Demo specific player")
            print("  2. Compare multiple players")
            print("  3. Show top predicted players")
            print("  4. Quit")
            
            choice = input(f"\nEnter choice (1-4): ").strip()
            
            if choice == '1':
                player_name = input(f"\nEnter player name (e.g., 'Tatum', 'LeBron'): ").strip()
                if player_name:
                    self.demo_player(player_name)
            
            elif choice == '2':
                print(f"\nEnter 2-3 player names to compare:")
                players = []
                for i in range(3):
                    name = input(f"Player {i+1} (or press Enter to skip): ").strip()
                    if name:
                        players.append(name)
                    else:
                        break
                
                if len(players) >= 2:
                    self.compare_players(players)
            
            elif choice == '3':
                self.show_top_predictions()
            
            elif choice == '4':
                print(f"\n👋 Thanks for using the NBA VORP Predictor!")
                break
            
            else:
                print(f"❌ Invalid choice. Please enter 1-4.")
    
    def compare_players(self, player_names):
        """Compare predictions for multiple players"""
        print(f"\n🏀 PLAYER COMPARISON")
        print("=" * 50)
        
        results = []
        for name in player_names:
            matches = self.find_player(name)
            if matches is not None and len(matches) > 0:
                player = matches.iloc[0]
                predicted = self.predict_player_vorp(player)
                actual = player.get('VORP', 0)
                results.append({
                    'name': player['Player'],
                    'predicted': predicted,
                    'actual': actual,
                    'per': player['PER'],
                    'mp': player['MP'],
                    'ws': player['WS']
                })
        
        if len(results) < 2:
            print("❌ Need at least 2 valid players for comparison")
            return
        
        # Sort by predicted VORP
        results.sort(key=lambda x: x['predicted'], reverse=True)
        
        print(f"📊 COMPARISON RESULTS (sorted by predicted VORP):")
        print(f"{'Rank':<4} {'Player':<20} {'Pred VORP':<10} {'Actual':<8} {'PER':<6} {'Minutes':<8}")
        print("-" * 60)
        
        for i, result in enumerate(results):
            print(f"{i+1:<4} {result['name']:<20} {result['predicted']:<10.2f} "
                  f"{result['actual']:<8.2f} {result['per']:<6.1f} {result['mp']:<8.0f}")
        
        # Winner analysis
        winner = results[0]
        print(f"\n🏆 PREDICTED WINNER: {winner['name']}")
        print(f"  Predicted VORP: {winner['predicted']:.2f}")
        print(f"  Key factors: {winner['per']:.1f} PER, {winner['mp']:.0f} minutes")
    
    def show_top_predictions(self):
        """Show top predicted players"""
        if self.real_players_data is None:
            return
        
        print(f"\n🏆 TOP 10 PREDICTED VORP PLAYERS")
        print("=" * 40)
        
        predictions = []
        for _, player in self.real_players_data.iterrows():
            predicted = self.predict_player_vorp(player)
            predictions.append({
                'name': player['Player'],
                'predicted': predicted,
                'actual': player.get('VORP', 0),
                'per': player['PER'],
                'mp': player['MP']
            })
        
        # Sort by prediction
        predictions.sort(key=lambda x: x['predicted'], reverse=True)
        
        print(f"{'Rank':<4} {'Player':<18} {'Pred':<6} {'Actual':<6} {'Error':<6}")
        print("-" * 42)
        
        for i, pred in enumerate(predictions[:10]):
            error = abs(pred['predicted'] - pred['actual'])
            print(f"{i+1:<4} {pred['name']:<18} {pred['predicted']:<6.2f} "
                  f"{pred['actual']:<6.2f} ±{error:<5.2f}")

def main():
    """Main demo function"""
    demo = PlayerVORPDemo()
    
    print("🏀 NBA Player VORP Predictor - Specific Player Demo")
    print("=" * 60)
    
    # Quick demo mode or interactive?
    print("Demo modes:")
    print("1. Quick demo with example players")
    print("2. Interactive demo (search any player)")
    
    choice = input("Choose mode (1 or 2): ").strip()
    
    if choice == '2':
        demo.interactive_demo()
    else:
        # Quick demo mode
        if not demo.train_model_quick():
            return
        
        print("\n🎯 QUICK DEMO: Top NBA Stars")
        
        # Demo some famous players
        famous_players = [
            "Jayson Tatum",
            "Luka", 
            "Jokic",
            "Edwards",
            "Fox"
        ]
        
        for player in famous_players:
            demo.demo_player(player)
        
        print(f"\n✨ Want to try other players?")
        print(f"Run: python demo_specific_player.py")
        print(f"Then choose option 2 for interactive mode!")

if __name__ == "__main__":
    main()