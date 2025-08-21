import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

from data.nfl_data import NFLDataGenerator
from models.predictor import NFLPredictor
from utils.helpers import format_probability, get_team_logo_placeholder

# Configure page
st.set_page_config(
    page_title="NFL Match Predictor",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize data and predictor
@st.cache_data
def load_data():
    """Load and cache NFL data"""
    data_gen = NFLDataGenerator()
    return data_gen

@st.cache_resource
def load_predictor():
    """Load and cache predictor model"""
    return NFLPredictor()

# Load data
data_generator = load_data()
predictor = load_predictor()

# Sidebar navigation
st.sidebar.title("🏈 NFL Predictor")
page = st.sidebar.selectbox(
    "Navigate to:",
    ["Game Predictions", "Team Statistics", "Prediction Accuracy", "About"]
)

if page == "Game Predictions":
    st.title("NFL Game Predictions")
    
    # Week selector
    current_week = st.sidebar.slider("Select Week", 1, 18, 1)
    
    # Get games for selected week
    games = data_generator.get_games_by_week(current_week)
    team_stats = data_generator.get_team_stats()
    
    if games.empty:
        st.warning(f"No games scheduled for Week {current_week}")
    else:
        st.subheader(f"Week {current_week} Predictions")
        
        # Display each game
        for idx, game in games.iterrows():
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col1:
                st.markdown(f"### {game['home_team']}")
                home_stats = team_stats[team_stats['team'] == game['home_team']].iloc[0]
                st.metric("Wins", home_stats['wins'])
                st.metric("Points Per Game", f"{home_stats['points_per_game']:.1f}")
                st.metric("Defense Rating", f"{home_stats['defense_rating']:.1f}")
            
            with col2:
                st.markdown("### VS")
                st.markdown(f"**{game['game_time']}**")
                
                # Get prediction
                prediction = predictor.predict_game(
                    game['home_team'], 
                    game['away_team'], 
                    team_stats
                )
                
                home_prob = prediction['home_win_probability']
                away_prob = prediction['away_win_probability']
                confidence = prediction['confidence']
                
                # Display prediction
                if home_prob > away_prob:
                    winner = game['home_team']
                    win_prob = home_prob
                else:
                    winner = game['away_team']
                    win_prob = away_prob
                
                st.success(f"**Predicted Winner: {winner}**")
                st.info(f"Win Probability: {format_probability(win_prob)}")
                st.info(f"Confidence: {format_probability(confidence)}")
            
            with col3:
                st.markdown(f"### {game['away_team']}")
                away_stats = team_stats[team_stats['team'] == game['away_team']].iloc[0]
                st.metric("Wins", away_stats['wins'])
                st.metric("Points Per Game", f"{away_stats['points_per_game']:.1f}")
                st.metric("Defense Rating", f"{away_stats['defense_rating']:.1f}")
            
            # Probability visualization
            st.markdown("---")
            prob_data = pd.DataFrame({
                'Team': [game['home_team'], game['away_team']],
                'Win Probability': [home_prob, away_prob]
            })
            
            fig = px.bar(
                prob_data, 
                x='Team', 
                y='Win Probability',
                title=f"Win Probability - {game['home_team']} vs {game['away_team']}",
                color='Win Probability',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")

elif page == "Team Statistics":
    st.title("Team Statistics")
    
    team_stats = data_generator.get_team_stats()
    
    # Team selector
    selected_team = st.sidebar.selectbox("Select Team", team_stats['team'].tolist())
    
    if selected_team:
        team_data = team_stats[team_stats['team'] == selected_team].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Wins", team_data['wins'])
        with col2:
            st.metric("Losses", team_data['losses'])
        with col3:
            st.metric("Win Percentage", f"{team_data['win_percentage']:.1%}")
        with col4:
            st.metric("Points Per Game", f"{team_data['points_per_game']:.1f}")
        
        # Additional stats
        st.subheader("Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Offense Rating", f"{team_data['offense_rating']:.1f}")
            st.metric("Defense Rating", f"{team_data['defense_rating']:.1f}")
            st.metric("Turnover Differential", f"{team_data['turnover_diff']:+.1f}")
        
        with col2:
            st.metric("Points Allowed Per Game", f"{team_data['points_allowed_per_game']:.1f}")
            st.metric("Yards Per Game", f"{team_data['yards_per_game']:.0f}")
            st.metric("Yards Allowed Per Game", f"{team_data['yards_allowed_per_game']:.0f}")
    
    # League standings
    st.subheader("League Standings")
    standings = team_stats.sort_values('win_percentage', ascending=False)
    standings['Rank'] = range(1, len(standings) + 1)
    
    display_standings = standings[['Rank', 'team', 'wins', 'losses', 'win_percentage', 'points_per_game']].copy()
    display_standings.columns = ['Rank', 'Team', 'Wins', 'Losses', 'Win %', 'PPG']
    display_standings['Win %'] = display_standings['Win %'].apply(lambda x: f"{x:.1%}")
    display_standings['PPG'] = display_standings['PPG'].apply(lambda x: f"{x:.1f}")
    
    st.dataframe(display_standings, use_container_width=True, hide_index=True)

elif page == "Prediction Accuracy":
    st.title("Prediction Accuracy Tracking")
    
    # Generate historical accuracy data
    historical_accuracy = data_generator.get_historical_accuracy()
    
    # Overall accuracy metrics
    col1, col2, col3 = st.columns(3)
    
    overall_accuracy = historical_accuracy['correct'].mean()
    high_confidence_accuracy = historical_accuracy[historical_accuracy['confidence'] > 0.7]['correct'].mean()
    low_confidence_accuracy = historical_accuracy[historical_accuracy['confidence'] <= 0.7]['correct'].mean()
    
    with col1:
        st.metric("Overall Accuracy", f"{overall_accuracy:.1%}")
    with col2:
        st.metric("High Confidence Accuracy", f"{high_confidence_accuracy:.1%}")
    with col3:
        st.metric("Low Confidence Accuracy", f"{low_confidence_accuracy:.1%}")
    
    # Accuracy by week
    st.subheader("Accuracy by Week")
    weekly_accuracy = historical_accuracy.groupby('week')['correct'].mean().reset_index()
    
    fig = px.line(
        weekly_accuracy, 
        x='week', 
        y='correct',
        title="Prediction Accuracy by Week",
        labels={'correct': 'Accuracy', 'week': 'Week'}
    )
    fig.update_traces(mode='lines+markers')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Confidence distribution
    st.subheader("Prediction Confidence Distribution")
    
    fig = px.histogram(
        historical_accuracy, 
        x='confidence',
        nbins=20,
        title="Distribution of Prediction Confidence Levels",
        labels={'confidence': 'Confidence Level', 'count': 'Number of Predictions'}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent predictions
    st.subheader("Recent Predictions")
    recent_predictions = historical_accuracy.tail(10)
    
    for idx, pred in recent_predictions.iterrows():
        status = "✅ Correct" if pred['correct'] else "❌ Incorrect"
        confidence_color = "green" if pred['confidence'] > 0.7 else "orange" if pred['confidence'] > 0.5 else "red"
        
        st.markdown(f"""
        **{pred['home_team']} vs {pred['away_team']}** (Week {pred['week']})
        - Predicted: {pred['predicted_winner']} ({format_probability(pred['win_probability'])})
        - Actual: {pred['actual_winner']}
        - Status: {status}
        - Confidence: <span style="color: {confidence_color}">{format_probability(pred['confidence'])}</span>
        """, unsafe_allow_html=True)
        st.markdown("---")

elif page == "About":
    st.title("About NFL Match Predictor")
    
    st.markdown("""
    ## Overview
    This NFL Match Predictor uses historical team performance data and statistical analysis to predict game outcomes.
    
    ## Prediction Algorithm
    Our prediction model considers multiple factors:
    - **Team Statistics**: Win-loss record, points scored/allowed
    - **Performance Ratings**: Offensive and defensive efficiency
    - **Historical Matchups**: Past performance between teams
    - **Home Field Advantage**: Statistical advantage for home teams
    
    ## Features
    - 📊 **Game Predictions**: Week-by-week game predictions with confidence levels
    - 📈 **Team Statistics**: Comprehensive team performance metrics
    - 🎯 **Accuracy Tracking**: Historical prediction accuracy analysis
    - 📱 **Responsive Design**: Works on desktop and mobile devices
    
    ## Data Sources
    **Note**: This application uses mock NFL data for demonstration purposes. 
    Team statistics and game schedules are generated to simulate realistic NFL scenarios.
    
    ## Accuracy Disclaimer
    Predictions are based on statistical analysis and should be used for entertainment purposes only.
    Actual game outcomes may vary significantly from predictions.
    
    ## Technology Stack
    - **Frontend**: Streamlit
    - **Data Processing**: Pandas, NumPy
    - **Visualization**: Plotly
    - **Machine Learning**: Scikit-learn
    
    ---
    *Last Updated: August 2025*
    """)
