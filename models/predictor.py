import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class NFLPredictor:
    """
    NFL game prediction model using team statistics and machine learning.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.home_field_advantage = 0.1  # 10% advantage for home team
    
    def _extract_features(self, home_team, away_team, team_stats):
        """Extract features for prediction from team statistics"""
        home_stats = team_stats[team_stats['team'] == home_team].iloc[0]
        away_stats = team_stats[team_stats['team'] == away_team].iloc[0]
        
        features = [
            # Home team stats
            home_stats['win_percentage'],
            home_stats['offense_rating'],
            home_stats['defense_rating'],
            home_stats['points_per_game'],
            home_stats['points_allowed_per_game'],
            home_stats['turnover_diff'],
            
            # Away team stats
            away_stats['win_percentage'],
            away_stats['offense_rating'],
            away_stats['defense_rating'],
            away_stats['points_per_game'],
            away_stats['points_allowed_per_game'],
            away_stats['turnover_diff'],
            
            # Comparative stats
            home_stats['win_percentage'] - away_stats['win_percentage'],
            home_stats['offense_rating'] - away_stats['defense_rating'],
            away_stats['offense_rating'] - home_stats['defense_rating'],
            home_stats['points_per_game'] - away_stats['points_allowed_per_game'],
            away_stats['points_per_game'] - home_stats['points_allowed_per_game']
        ]
        
        return np.array(features).reshape(1, -1)
    
    def predict_game(self, home_team, away_team, team_stats):
        """
        Predict the outcome of a game between two teams.
        Returns prediction with probabilities and confidence.
        """
        # Get team statistics
        home_stats = team_stats[team_stats['team'] == home_team].iloc[0]
        away_stats = team_stats[team_stats['team'] == away_team].iloc[0]
        
        # Simple statistical prediction
        # Base prediction on win percentage with home field advantage
        home_win_prob = self._calculate_win_probability(home_stats, away_stats, is_home=True)
        away_win_prob = 1 - home_win_prob
        
        # Calculate confidence based on difference in team quality
        team_quality_diff = abs(home_stats['win_percentage'] - away_stats['win_percentage'])
        offense_defense_factor = abs(
            (home_stats['offense_rating'] - away_stats['defense_rating']) - 
            (away_stats['offense_rating'] - home_stats['defense_rating'])
        ) / 100
        
        confidence = min(0.95, max(0.5, (team_quality_diff + offense_defense_factor) * 1.5))
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_win_probability': home_win_prob,
            'away_win_probability': away_win_prob,
            'predicted_winner': home_team if home_win_prob > away_win_prob else away_team,
            'confidence': confidence,
            'predicted_spread': self._calculate_spread(home_stats, away_stats)
        }
    
    def _calculate_win_probability(self, home_stats, away_stats, is_home=True):
        """Calculate win probability using multiple factors"""
        # Base probability from win percentage
        home_base = home_stats['win_percentage']
        away_base = away_stats['win_percentage']
        
        # Adjust for offensive vs defensive matchup
        home_offense_vs_away_defense = self._matchup_factor(
            home_stats['offense_rating'], away_stats['defense_rating']
        )
        away_offense_vs_home_defense = self._matchup_factor(
            away_stats['offense_rating'], home_stats['defense_rating']
        )
        
        # Combine factors
        home_advantage = self.home_field_advantage if is_home else 0
        
        # Weighted combination of factors
        home_score = (
            home_base * 0.4 +
            home_offense_vs_away_defense * 0.3 +
            (1 - away_offense_vs_home_defense) * 0.2 +
            home_advantage * 0.1
        )
        
        # Normalize to probability
        away_score = (
            away_base * 0.4 +
            away_offense_vs_home_defense * 0.3 +
            (1 - home_offense_vs_away_defense) * 0.2
        )
        
        total_score = home_score + away_score
        if total_score > 0:
            home_win_prob = home_score / total_score
        else:
            home_win_prob = 0.5
        
        # Ensure probability is between 0.05 and 0.95
        return max(0.05, min(0.95, home_win_prob))
    
    def _matchup_factor(self, offense_rating, defense_rating):
        """Calculate matchup factor between offense and defense"""
        # Higher offense rating vs lower defense rating = advantage
        return offense_rating / (offense_rating + defense_rating)
    
    def _calculate_spread(self, home_stats, away_stats):
        """Calculate predicted point spread"""
        home_scoring = home_stats['points_per_game'] - away_stats['points_allowed_per_game']
        away_scoring = away_stats['points_per_game'] - home_stats['points_allowed_per_game']
        
        # Add home field advantage (typically 3 points)
        spread = home_scoring - away_scoring + 3
        
        return round(spread, 1)
    
    def batch_predict(self, games, team_stats):
        """Predict multiple games at once"""
        predictions = []
        
        for idx, game in games.iterrows():
            prediction = self.predict_game(
                game['home_team'], 
                game['away_team'], 
                team_stats
            )
            predictions.append(prediction)
        
        return predictions
