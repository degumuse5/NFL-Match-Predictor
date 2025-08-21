import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class NFLDataGenerator:
    """
    Mock NFL data generator for demonstration purposes.
    Generates realistic team statistics, schedules, and historical data.
    """
    
    def __init__(self):
        self.teams = [
            "Buffalo Bills", "Miami Dolphins", "New England Patriots", "New York Jets",
            "Baltimore Ravens", "Cincinnati Bengals", "Cleveland Browns", "Pittsburgh Steelers",
            "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars", "Tennessee Titans",
            "Denver Broncos", "Kansas City Chiefs", "Las Vegas Raiders", "Los Angeles Chargers",
            "Dallas Cowboys", "New York Giants", "Philadelphia Eagles", "Washington Commanders",
            "Chicago Bears", "Detroit Lions", "Green Bay Packers", "Minnesota Vikings",
            "Atlanta Falcons", "Carolina Panthers", "New Orleans Saints", "Tampa Bay Buccaneers",
            "Arizona Cardinals", "Los Angeles Rams", "San Francisco 49ers", "Seattle Seahawks"
        ]
        
        # Set random seed for consistent mock data
        np.random.seed(42)
        random.seed(42)
        
        self._generate_team_stats()
        self._generate_schedule()
        self._generate_historical_accuracy()
    
    def _generate_team_stats(self):
        """Generate mock team statistics"""
        stats_data = []
        
        for team in self.teams:
            # Generate realistic win-loss records
            wins = np.random.randint(2, 15)
            losses = 16 - wins
            win_percentage = wins / 16
            
            # Generate performance metrics with some correlation to wins
            base_performance = 0.3 + (win_percentage * 0.7)
            
            offense_rating = np.random.normal(base_performance * 100, 15)
            defense_rating = np.random.normal((1 - base_performance) * 100, 15)
            
            points_per_game = np.random.normal(20 + (offense_rating * 0.3), 3)
            points_allowed_per_game = np.random.normal(20 + (defense_rating * 0.2), 3)
            
            yards_per_game = np.random.normal(300 + (offense_rating * 2), 50)
            yards_allowed_per_game = np.random.normal(300 + (defense_rating * 1.5), 50)
            
            turnover_diff = np.random.normal((win_percentage - 0.5) * 20, 5)
            
            stats_data.append({
                'team': team,
                'wins': wins,
                'losses': losses,
                'win_percentage': win_percentage,
                'offense_rating': max(0, min(100, offense_rating)),
                'defense_rating': max(0, min(100, defense_rating)),
                'points_per_game': max(10, points_per_game),
                'points_allowed_per_game': max(10, points_allowed_per_game),
                'yards_per_game': max(200, yards_per_game),
                'yards_allowed_per_game': max(200, yards_allowed_per_game),
                'turnover_diff': turnover_diff
            })
        
        self.team_stats = pd.DataFrame(stats_data)
    
    def _generate_schedule(self):
        """Generate mock game schedule"""
        schedule_data = []
        
        # Generate games for each week
        for week in range(1, 19):  # 18 weeks in NFL season
            week_teams = self.teams.copy()
            random.shuffle(week_teams)
            
            # Create matchups (16 games per week)
            for i in range(0, len(week_teams), 2):
                if i + 1 < len(week_teams):
                    home_team = week_teams[i]
                    away_team = week_teams[i + 1]
                    
                    # Generate game time
                    base_date = datetime(2024, 9, 8) + timedelta(weeks=week-1)
                    game_day = base_date + timedelta(days=random.randint(0, 6))
                    game_time = game_day.strftime("%m/%d %I:%M %p")
                    
                    schedule_data.append({
                        'week': week,
                        'home_team': home_team,
                        'away_team': away_team,
                        'game_time': game_time,
                        'played': week < 10  # Assume first 9 weeks are played
                    })
        
        self.schedule = pd.DataFrame(schedule_data)
    
    def _generate_historical_accuracy(self):
        """Generate mock historical prediction accuracy data"""
        accuracy_data = []
        
        played_games = self.schedule[self.schedule['played'] == True]
        
        for idx, game in played_games.iterrows():
            # Simulate prediction
            home_stats = self.team_stats[self.team_stats['team'] == game['home_team']].iloc[0]
            away_stats = self.team_stats[self.team_stats['team'] == game['away_team']].iloc[0]
            
            # Simple prediction based on win percentage
            home_advantage = 0.1  # 10% home field advantage
            home_prob = (home_stats['win_percentage'] + home_advantage) / \
                       (home_stats['win_percentage'] + away_stats['win_percentage'] + home_advantage)
            
            confidence = abs(home_prob - 0.5) * 2  # Confidence based on how far from 50-50
            
            predicted_winner = game['home_team'] if home_prob > 0.5 else game['away_team']
            win_probability = max(home_prob, 1 - home_prob)
            
            # Simulate actual winner (with some randomness but favoring better teams)
            actual_home_prob = home_stats['win_percentage'] * 0.7 + 0.15  # Add some randomness
            actual_winner = game['home_team'] if np.random.random() < actual_home_prob else game['away_team']
            
            correct = predicted_winner == actual_winner
            
            accuracy_data.append({
                'week': game['week'],
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'predicted_winner': predicted_winner,
                'actual_winner': actual_winner,
                'win_probability': win_probability,
                'confidence': confidence,
                'correct': correct
            })
        
        self.historical_accuracy = pd.DataFrame(accuracy_data)
    
    def get_team_stats(self):
        """Get team statistics"""
        return self.team_stats.copy()
    
    def get_games_by_week(self, week):
        """Get games for a specific week"""
        return self.schedule[self.schedule['week'] == week].copy()
    
    def get_historical_accuracy(self):
        """Get historical prediction accuracy data"""
        return self.historical_accuracy.copy()
    
    def get_all_games(self):
        """Get all games in schedule"""
        return self.schedule.copy()
