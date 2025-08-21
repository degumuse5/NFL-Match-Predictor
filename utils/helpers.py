import streamlit as st

def format_probability(prob):
    """Format probability as percentage string"""
    return f"{prob:.1%}"

def format_decimal(value, decimals=1):
    """Format decimal value to specified decimal places"""
    return f"{value:.{decimals}f}"

def get_team_logo_placeholder():
    """Return placeholder for team logos (since we're not using actual images)"""
    return "🏈"

def get_win_loss_color(win_percentage):
    """Return color based on win percentage"""
    if win_percentage >= 0.75:
        return "green"
    elif win_percentage >= 0.5:
        return "orange"
    else:
        return "red"

def display_team_record(wins, losses):
    """Format team record display"""
    return f"{wins}-{losses}"

def calculate_strength_of_schedule(team, team_stats, schedule):
    """Calculate a team's strength of schedule"""
    # This is a simplified calculation
    # In reality, it would consider opponent win percentages
    return 0.5  # Placeholder

def get_prediction_confidence_color(confidence):
    """Return color based on prediction confidence"""
    if confidence >= 0.8:
        return "green"
    elif confidence >= 0.6:
        return "orange"
    else:
        return "red"

def format_spread(spread):
    """Format point spread for display"""
    if spread > 0:
        return f"+{spread}"
    else:
        return str(spread)

@st.cache_data
def load_team_colors():
    """Load team color schemes (simplified)"""
    # In a real app, this would contain actual team colors
    colors = {}
    teams = [
        "Buffalo Bills", "Miami Dolphins", "New England Patriots", "New York Jets",
        "Baltimore Ravens", "Cincinnati Bengals", "Cleveland Browns", "Pittsburgh Steelers",
        "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars", "Tennessee Titans",
        "Denver Broncos", "Kansas City Chiefs", "Las Vegas Raiders", "Los Angeles Chargers",
        "Dallas Cowboys", "New York Giants", "Philadelphia Eagles", "Washington Commanders",
        "Chicago Bears", "Detroit Lions", "Green Bay Packers", "Minnesota Vikings",
        "Atlanta Falcons", "Carolina Panthers", "New Orleans Saints", "Tampa Bay Buccaneers",
        "Arizona Cardinals", "Los Angeles Rams", "San Francisco 49ers", "Seattle Seahawks"
    ]
    
    # Assign default colors
    for team in teams:
        colors[team] = {"primary": "#1f77b4", "secondary": "#ff7f0e"}
    
    return colors

def validate_team_name(team_name, valid_teams):
    """Validate that team name exists in the dataset"""
    return team_name in valid_teams

def calculate_prediction_accuracy(predictions, actual_results):
    """Calculate accuracy of predictions against actual results"""
    if len(predictions) != len(actual_results):
        return 0.0
    
    correct = sum(1 for pred, actual in zip(predictions, actual_results) if pred == actual)
    return correct / len(predictions)

def get_game_status_icon(is_played, is_correct_prediction=None):
    """Get icon for game status"""
    if not is_played:
        return "⏳"  # Upcoming
    elif is_correct_prediction is True:
        return "✅"  # Correct prediction
    elif is_correct_prediction is False:
        return "❌"  # Incorrect prediction
    else:
        return "🏈"  # Played (no prediction info)

def format_team_name(team_name, max_length=20):
    """Format team name for display, truncating if necessary"""
    if len(team_name) <= max_length:
        return team_name
    else:
        return team_name[:max_length-3] + "..."
