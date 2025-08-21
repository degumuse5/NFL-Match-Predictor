# NFL-Match-Predictor
NFL Match Predictor is a Streamlit web application that provides NFL game predictions using machine learning. The application generates mock NFL data including team statistics, and historical performance metrics, then uses a Random Forest classifier to predict game outcomes. Users can view predictions for different weeks, analyze team statistics. 

NFL Match Predictor

## Overview

NFL Match Predictor is a Streamlit web application that provides NFL game predictions using machine learning. The application generates mock NFL data including team statistics, schedules, and historical performance metrics, then uses a Random Forest classifier to predict game outcomes. Users can view predictions for different weeks, analyze team statistics, and review prediction accuracy through an interactive dashboard.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit Framework**: Single-page application with sidebar navigation for different views (Game Predictions, Team Statistics, Prediction Accuracy, About)
- **Interactive Visualizations**: Uses Plotly Express and Plotly Graph Objects for charts and data visualization
- **Caching Strategy**: Implements Streamlit's `@st.cache_data` and `@st.cache_resource` decorators for performance optimization of data loading and model initialization

### Backend Architecture
- **Data Layer**: Mock data generation system (`NFLDataGenerator`) that creates realistic NFL team statistics, schedules, and historical data with consistent seeding for reproducibility
- **Prediction Engine**: Machine learning model (`NFLPredictor`) using scikit-learn's Random Forest classifier with feature engineering based on team performance metrics
- **Utility Layer**: Helper functions for data formatting, display utilities, and calculation methods

### Data Processing
- **Feature Engineering**: Extracts comparative statistics between home and away teams including win percentages, offensive/defensive ratings, and point differentials
- **Mock Data Generation**: Creates 32 NFL teams with realistic statistics including wins/losses, performance ratings, and game schedules
- **Statistical Analysis**: Implements strength of schedule calculations, prediction confidence metrics, and historical accuracy tracking

### Model Architecture
- **Random Forest Classifier**: Primary prediction model using team statistical features
- **Home Field Advantage**: Built-in 10% advantage factor for home teams
- **Feature Scaling**: StandardScaler preprocessing for numerical features
- **Prediction Output**: Provides win probability, point spreads, and confidence intervals

## External Dependencies

### Python Libraries
- **streamlit**: Web application framework for the user interface
- **pandas**: Data manipulation and analysis
- **plotly**: Interactive data visualization (plotly.express and plotly.graph_objects)
- **numpy**: Numerical computing and array operations
- **scikit-learn**: Machine learning library providing RandomForestClassifier, train_test_split, and StandardScaler

### Data Sources
- **Mock Data Generation**: Currently uses internally generated mock NFL data with realistic statistical distributions
