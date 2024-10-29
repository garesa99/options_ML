import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
import matplotlib.pyplot as plt
import shap

# Set page configuration
st.set_page_config(
    page_title="Advanced Options Analysis Tool",
    page_icon="📈",
    layout="wide"
)

# Global variables
API_KEY = None  # Will be set by user input
BASE_URL = "https://api.polygon.io"

###############################################################################
#                            DATA FETCHING FUNCTIONS                          #
###############################################################################

def get_option_contracts(ticker, expiration_date=None):
    """
    Fetches a list of option contracts for a given ticker from the Polygon.io API.

    Parameters:
    - ticker (str): The stock ticker symbol.
    - expiration_date (str): Optional expiration date to filter contracts.

    Returns:
    - DataFrame containing option contracts data.
    """
    url = f"{BASE_URL}/v3/reference/options/contracts"
    params = {
        'underlying_ticker': ticker,
        'limit': 1000,
        'apiKey': API_KEY
    }
    if expiration_date:
        params['expiration_date'] = expiration_date

    with st.spinner(f"Fetching option contracts for {ticker}..."):
        try:
            response = requests.get(url, params=params)
            data = response.json()
            if 'results' in data and data['results']:
                df = pd.DataFrame(data['results'])
                return df
            else:
                st.error(f"No options contract data found for {ticker}")
                return None
        except Exception as e:
            st.error(f"Error fetching option contracts: {e}")
            return None

def get_option_aggregates(option_ticker, from_date, to_date):
    """
    Fetches historical price data for a specific option contract.

    Parameters:
    - option_ticker (str): The option contract ticker symbol.
    - from_date (str): Start date in 'YYYY-MM-DD' format.
    - to_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
    - DataFrame containing historical price data.
    """
    url = f"{BASE_URL}/v2/aggs/ticker/{option_ticker}/range/1/day/{from_date}/{to_date}"
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000,
        'apiKey': API_KEY
    }
    with st.spinner(f"Fetching aggregates for {option_ticker}..."):
        try:
            response = requests.get(url, params=params)
            data = response.json()
            if 'results' in data and data['results']:
                df = pd.DataFrame(data['results'])
                df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df
            else:
                st.error(f"No aggregates data found for {option_ticker}")
                return None
        except Exception as e:
            st.error(f"Error fetching aggregates for {option_ticker}: {e}")
            return None

def get_stock_price(ticker, from_date, to_date):
    """
    Fetches historical stock price data for a given ticker.

    Parameters:
    - ticker (str): The stock ticker symbol.
    - from_date (str): Start date in 'YYYY-MM-DD' format.
    - to_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
    - DataFrame containing historical stock price data.
    """
    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}"
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000,
        'apiKey': API_KEY
    }
    with st.spinner(f"Fetching stock price data for {ticker}..."):
        try:
            response = requests.get(url, params=params)
            data = response.json()
            if 'results' in data and data['results']:
                df = pd.DataFrame(data['results'])
                df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df
            else:
                st.error(f"No stock price data found for {ticker}")
                return None
        except Exception as e:
            st.error(f"Error fetching stock price for {ticker}: {e}")
            return None

###############################################################################
#                           FEATURE ENGINEERING                               #
###############################################################################

def calculate_features(contracts_df, stock_df):
    """
    Calculates additional features for options analysis.

    Parameters:
    - contracts_df (DataFrame): DataFrame containing option contracts.
    - stock_df (DataFrame): DataFrame containing stock price data.

    Returns:
    - DataFrame with additional features.
    """
    df = contracts_df.copy()
    try:
        # Calculate time to expiration
        df['expiration_date'] = pd.to_datetime(df['expiration_date'])
        df['days_to_expiration'] = (df['expiration_date'] - pd.Timestamp.now()).dt.days

        # Calculate moneyness
        current_price = stock_df['c'].iloc[-1]
        df['moneyness'] = (current_price - df['strike_price']) / df['strike_price']

        # Calculate ratios
        df['strike_to_stock_ratio'] = df['strike_price'] / current_price

        # Calculate volatility and momentum
        stock_returns = stock_df['c'].pct_change().dropna()
        df['stock_volatility'] = stock_returns.std() * np.sqrt(252)  # Annualized volatility

        if len(stock_returns) >= 30:
            df['volatility_10d'] = stock_returns.rolling(10).std().iloc[-1] * np.sqrt(252)
            df['volatility_30d'] = stock_returns.rolling(30).std().iloc[-1] * np.sqrt(252)
            df['price_momentum_5d'] = stock_df['c'].pct_change(5).iloc[-1]
            df['price_momentum_20d'] = stock_df['c'].pct_change(20).iloc[-1]
        else:
            df['volatility_10d'] = df['stock_volatility']
            df['volatility_30d'] = df['stock_volatility']
            df['price_momentum_5d'] = 0
            df['price_momentum_20d'] = 0

        # Handle missing values with SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        df[[
            'volatility_10d', 'volatility_30d',
            'price_momentum_5d', 'price_momentum_20d'
        ]] = imputer.fit_transform(df[[
            'volatility_10d', 'volatility_30d',
            'price_momentum_5d', 'price_momentum_20d'
        ]])

        return df

    except Exception as e:
        st.error(f"Error in feature calculation: {str(e)}")
        return None

###############################################################################
#                           DATA PREPARATION                                  #
###############################################################################

def prepare_training_data(contracts_df, stock_df, return_threshold):
    """
    Prepares data for machine learning model training.

    Parameters:
    - contracts_df (DataFrame): DataFrame containing option contracts with features.
    - stock_df (DataFrame): DataFrame containing stock price data.
    - return_threshold (float): Return threshold to define a positive outcome.

    Returns:
    - X_original (DataFrame): Original feature set.
    - y_original (Series): Original target variable.
    - X_resampled (DataFrame): Resampled feature set for training.
    - y_resampled (Series): Resampled target variable for training.
    - simulated_returns (Series): Simulated returns for options.
    """
    required_features = [
        'strike_price', 'days_to_expiration', 'moneyness',
        'strike_to_stock_ratio', 'stock_volatility',
        'volatility_10d', 'volatility_30d',
        'price_momentum_5d', 'price_momentum_20d'
    ]

    try:
        # Ensure all required features are present
        missing_features = [f for f in required_features if f not in contracts_df.columns]
        if missing_features:
            st.error(f"Missing required features: {missing_features}")
            return None, None, None, None, None

        # Create feature matrix
        X_original = contracts_df[required_features].copy()

        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X_original), columns=X_original.columns)

        # Generate target variable based on hypothetical performance
        # For demonstration purposes, we'll simulate option returns
        # In practice, you should use actual historical option price data

        # Simulate option returns
        np.random.seed(42)
        simulated_returns = np.random.normal(0, 0.1, size=len(X))
        # Do not modify contracts_df here; return simulated_returns instead

        # Define target variable
        y = (simulated_returns > return_threshold).astype(int)

        # Address class imbalance by resampling
        X_resampled, y_resampled = resample_data(X, y)

        return X, y, X_resampled, y_resampled, simulated_returns

    except Exception as e:
        st.error(f"Error in data preparation: {str(e)}")
        return None, None, None, None, None

def resample_data(X, y):
    """
    Addresses class imbalance by upsampling the minority class.

    Parameters:
    - X (DataFrame): Feature matrix.
    - y (Series): Target variable.

    Returns:
    - X_resampled (DataFrame): Resampled feature matrix.
    - y_resampled (Series): Resampled target variable.
    """
    # Combine X and y into a single DataFrame
    data = X.copy()
    data['target'] = y

    # Separate majority and minority classes
    majority_class = data[data['target'] == 0]
    minority_class = data[data['target'] == 1]

    # Upsample minority class
    if len(minority_class) > 0:
        minority_upsampled = resample(
            minority_class,
            replace=True,
            n_samples=len(majority_class),
            random_state=42
        )
        # Combine majority class with upsampled minority class
        upsampled_data = pd.concat([majority_class, minority_upsampled])
    else:
        # If no minority class, return the original data
        upsampled_data = majority_class

    # Shuffle the dataset
    upsampled_data = upsampled_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Separate features and target
    X_resampled = upsampled_data.drop('target', axis=1)
    y_resampled = upsampled_data['target']

    return X_resampled, y_resampled

###############################################################################
#                           MODEL TRAINING                                    #
###############################################################################

def train_model(X_resampled, y_resampled):
    """
    Trains a Random Forest model with hyperparameter tuning and cross-validation.

    Parameters:
    - X_resampled (DataFrame): Resampled feature matrix.
    - y_resampled (Series): Resampled target variable.

    Returns:
    - best_model: Trained model with the best hyperparameters.
    - cv_results (DataFrame): Cross-validation results.
    """
    try:
        # Define pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=42))
        ])

        # Define hyperparameters grid
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2],
            'classifier__class_weight': ['balanced']
        }

        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )

        # Fit the model
        grid_search.fit(X_resampled, y_resampled)

        # Best estimator
        best_model = grid_search.best_estimator_

        # Cross-validated predictions
        cv_results = pd.DataFrame(grid_search.cv_results_)

        st.write("### Best Hyperparameters:")
        st.json(grid_search.best_params_)

        return best_model, cv_results

    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
        return None, None

###############################################################################
#                           MODEL EVALUATION                                  #
###############################################################################

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on test data using various metrics.

    Parameters:
    - model: Trained machine learning model.
    - X_test (DataFrame): Test feature set.
    - y_test (Series): True labels for the test set.
    """
    try:
        # Predict probabilities
        y_proba = model.predict_proba(X_test)[:, 1]

        # Predict classes
        y_pred = model.predict(X_test)

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        st.write("### Classification Report:")
        st.dataframe(pd.DataFrame(report).transpose())

        # Confusion matrix
        st.write("### Confusion Matrix:")
        conf_matrix = confusion_matrix(y_test, y_pred)
        conf_matrix_fig = px.imshow(
            conf_matrix,
            labels=dict(x="Predicted", y="Actual"),
            x=['Negative', 'Positive'],
            y=['Negative', 'Positive'],
            text_auto=True,
            color_continuous_scale='Blues',
            title="Confusion Matrix"
        )
        st.plotly_chart(conf_matrix_fig)

        st.write("""
        **How to Interpret the Confusion Matrix:**

        - **True Positives (TP):** Correctly predicted positive cases (bottom-right cell).
        - **True Negatives (TN):** Correctly predicted negative cases (top-left cell).
        - **False Positives (FP):** Incorrectly predicted positive cases (top-right cell).
        - **False Negatives (FN):** Incorrectly predicted negative cases (bottom-left cell).

        A higher number of TPs and TNs indicates better model performance.
        """)

        # ROC AUC Score
        roc_auc = roc_auc_score(y_test, y_proba)
        st.write(f"**ROC AUC Score:** {roc_auc:.2f}")

        # Plot ROC Curve using Plotly
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_fig = px.area(
            x=fpr, y=tpr,
            title='ROC Curve',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
            width=700, height=500
        )
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='black', dash='dash')))
        st.plotly_chart(roc_fig)

        st.write("""
        **How to Interpret the ROC Curve:**

        - The ROC curve plots the True Positive Rate (Recall) against the False Positive Rate at various threshold settings.
        - An ROC curve closer to the top-left corner indicates a better performing model.
        - The diagonal line represents a model with no discriminative power (AUC = 0.5).
        - **Area Under the Curve (AUC):** Measures the entire two-dimensional area underneath the entire ROC curve. An AUC of 1.0 represents a perfect model.
        """)

        # Plot Precision-Recall Curve using Plotly
        precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba)
        pr_fig = px.area(
            x=recall, y=precision,
            title='Precision-Recall Curve',
            labels=dict(x='Recall', y='Precision'),
            width=700, height=500
        )
        st.plotly_chart(pr_fig)

        st.write("""
        **How to Interpret the Precision-Recall Curve:**

        - The Precision-Recall curve shows the trade-off between precision and recall for different threshold settings.
        - A larger area under the curve indicates a better model.
        - Useful when dealing with imbalanced datasets.
        """)

    except Exception as e:
        st.error(f"Error in model evaluation: {str(e)}")

###############################################################################
#                           MODEL INTERPRETATION                              #
###############################################################################

def plot_feature_importance(model, X):
    """
    Creates a feature importance plot using SHAP values.

    Parameters:
    - model: Trained machine learning model.
    - X (DataFrame): Feature set used in the model.
    """
    try:
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model.named_steps['classifier'])
        shap_values = explainer.shap_values(model.named_steps['scaler'].transform(X))

        # For binary classification, shap_values is a list with two elements
        # We take the SHAP values for class 1
        shap_values_class1 = shap_values[1]

        # Plot feature importance
        st.write("### Feature Importance (SHAP Values):")
        plt.figure()
        shap.summary_plot(shap_values_class1, X, plot_type="bar", show=False)
        st.pyplot(plt.gcf())
        plt.clf()  # Clear the current figure

        st.write("""
        **How to Interpret the SHAP Bar Plot:**

        - Features are ranked by their average impact on the model's output.
        - Longer bars indicate features that contribute more to the predictions.
        - Positive values increase the prediction probability, while negative values decrease it.
        """)

        # Detailed SHAP summary plot
        st.write("### SHAP Summary Plot:")
        plt.figure()
        shap.summary_plot(shap_values_class1, X, show=False)
        st.pyplot(plt.gcf())
        plt.clf()

        st.write("""
        **How to Interpret the SHAP Summary Plot:**

        - Each dot represents an individual option contract.
        - The color represents the feature value (red high, blue low).
        - The position on the X-axis shows whether the effect of that value is associated with a higher or lower prediction.
        """)

    except Exception as e:
        st.error(f"Error plotting feature importance: {str(e)}")

###############################################################################
#                           OPPORTUNITY ANALYSIS                              #
###############################################################################

def display_opportunity_analysis(enhanced_df, probabilities, threshold):
    """
    Displays a comprehensive opportunity analysis.

    Parameters:
    - enhanced_df (DataFrame): DataFrame containing options with features.
    - probabilities (array): Predicted probabilities of being an opportunity.
    - threshold (float): Probability threshold to define a positive prediction.
    """
    st.header("Opportunity Analysis")

    # Add opportunity scores
    enhanced_df['opportunity_score'] = probabilities

    # Apply threshold to get predictions
    enhanced_df['prediction'] = (enhanced_df['opportunity_score'] >= threshold).astype(int)

    # Heatmap visualization
    st.subheader("Options Opportunity Heatmap")
    st.write("""
    This heatmap shows the predicted opportunity scores across different strike prices and days to expiration.
    Hover over the cells to see the exact opportunity scores.
    """)
    fig = plot_opportunity_heatmap(enhanced_df)
    st.plotly_chart(fig, use_container_width=True)

    # Top opportunities
    st.subheader("Top Trading Opportunities")
    st.write("""
    These are the top options contracts with the highest predicted opportunity scores.
    The scores represent the probability that the option will exceed the return threshold.
    """)
    top_opportunities = enhanced_df.sort_values(by='opportunity_score', ascending=False).head(10)
    display_cols = [
        'ticker', 'contract_type', 'strike_price',
        'expiration_date', 'days_to_expiration',
        'moneyness', 'opportunity_score'
    ]
    formatted_df = top_opportunities[display_cols].copy()
    formatted_df['strike_price'] = formatted_df['strike_price'].map('${:,.2f}'.format)
    formatted_df['moneyness'] = formatted_df['moneyness'].map('{:,.2%}'.format)
    formatted_df['opportunity_score'] = formatted_df['opportunity_score'].map('{:,.2%}'.format)
    st.dataframe(formatted_df.style.highlight_max(axis=0), use_container_width=True)

def plot_opportunity_heatmap(contracts_df):
    """
    Creates a heatmap of predicted opportunities.

    Parameters:
    - contracts_df (DataFrame): DataFrame containing options with opportunity scores.

    Returns:
    - Plotly Figure object for the heatmap.
    """
    try:
        heatmap_data = contracts_df.copy()

        # Pivot the data to create a matrix for the heatmap
        heatmap_matrix = heatmap_data.pivot_table(
            index='strike_price',
            columns='days_to_expiration',
            values='opportunity_score',
            aggfunc='mean'
        )

        # Sort indices for better visualization
        heatmap_matrix = heatmap_matrix.sort_index(ascending=False)

        # Create the heatmap using Plotly
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_matrix.values,
            x=heatmap_matrix.columns,
            y=heatmap_matrix.index,
            colorscale='Viridis',
            colorbar=dict(title="Opportunity Score"),
            hoverongaps=False
        ))

        fig.update_layout(
            xaxis_title='Days to Expiration',
            yaxis_title='Strike Price',
            autosize=True,
            height=600
        )

        return fig

    except Exception as e:
        st.error(f"Error creating heatmap: {str(e)}")
        return go.Figure()

###############################################################################
#                           BACKTESTING                                       #
###############################################################################

def backtest_strategy(enhanced_df, initial_capital, trade_size):
    """
    Simulates a trading strategy based on the model's predictions.

    Parameters:
    - enhanced_df (DataFrame): DataFrame containing options with predictions.
    - initial_capital (float): Starting capital for the backtest.
    - trade_size (float): Amount to invest in each trade.

    Returns:
    - results_df (DataFrame): DataFrame containing backtest results over time.
    - total_profit (float): Total profit or loss from the backtest.
    - total_return_pct (float): Total return percentage.
    """
    # For the backtest, we'll simulate buying options with high opportunity scores
    # and calculate returns based on the simulated returns

    # Sort options by opportunity score
    sorted_options = enhanced_df.sort_values(by='opportunity_score', ascending=False)

    # Keep only options with positive predictions
    positive_options = sorted_options[sorted_options['prediction'] == 1]

    # Calculate the number of trades we can make
    total_trades = int(initial_capital / trade_size)
    selected_trades = positive_options.head(total_trades)

    # Calculate profit/loss for each trade
    selected_trades['profit_loss'] = trade_size * selected_trades['simulated_return']
    selected_trades['return_%'] = selected_trades['simulated_return'] * 100

    # Cumulative returns
    selected_trades['cumulative_profit'] = selected_trades['profit_loss'].cumsum()
    selected_trades['capital'] = initial_capital + selected_trades['cumulative_profit']
    selected_trades['cumulative_return_%'] = (selected_trades['capital'] - initial_capital) / initial_capital * 100

    # Prepare results DataFrame
    results_df = selected_trades[['expiration_date', 'ticker', 'profit_loss', 'return_%', 'capital', 'cumulative_return_%']]
    results_df.reset_index(drop=True, inplace=True)

    # Calculate total profit and return percentage
    total_profit = results_df['profit_loss'].sum()
    total_return_pct = (results_df['capital'].iloc[-1] - initial_capital) / initial_capital * 100

    return results_df, total_profit, total_return_pct

def display_backtest_results(results_df, total_profit, total_return_pct):
    """
    Displays the backtest results.

    Parameters:
    - results_df (DataFrame): DataFrame containing backtest results over time.
    - total_profit (float): Total profit or loss from the backtest.
    - total_return_pct (float): Total return percentage.
    """
    st.header("Backtesting Results")
    st.write("""
    The backtest simulates a trading strategy where we invest a fixed amount in each option predicted to be an opportunity.
    The profit or loss is calculated based on the simulated returns.
    """)

    # Display summary metrics
    st.subheader("Backtest Summary")
    st.write(f"**Total Profit/Loss:** ${total_profit:,.2f}")
    st.write(f"**Total Return:** {total_return_pct:.2f}%")
    st.write(f"**Number of Trades Executed:** {len(results_df)}")

    # Display cumulative capital over time
    st.subheader("Capital Over Time")
    fig = px.line(results_df, x=results_df.index + 1, y='capital', markers=True,
                  title='Capital Over Time',
                  labels={'index': 'Trade Number', 'capital': 'Capital ($)'})
    st.plotly_chart(fig, use_container_width=True)

    st.write("""
    **How to Interpret the Capital Over Time Chart:**

    - The chart shows how your capital changes with each trade executed.
    - An upward trend indicates profitable trades, while a downward trend indicates losses.
    - It helps assess the overall performance and risk of the trading strategy.
    """)

    # Display individual trade results
    st.subheader("Trade Details")
    st.write("Details of each trade executed during the backtest.")
    results_display = results_df.copy()
    results_display['profit_loss'] = results_display['profit_loss'].map('${:,.2f}'.format)
    results_display['capital'] = results_display['capital'].map('${:,.2f}'.format)
    results_display['return_%'] = results_display['return_%'].map('{:,.2f}%'.format)
    results_display['cumulative_return_%'] = results_display['cumulative_return_%'].map('{:,.2f}%'.format)
    st.dataframe(results_display, use_container_width=True)

###############################################################################
#                           MAIN APPLICATION                                  #
###############################################################################

def main():
    st.title("📈 Advanced Options Analysis Tool")
    st.markdown("""
    Welcome to the Advanced Options Analysis Tool! This application helps you analyze option contracts and identify potential trading opportunities using machine learning.

    Created by [Gabriel Reyes - gabriel.reyes@gsom.polimi.it]

    **Key Features:**
    - Fetch and display historical stock and options data.
    - Perform feature engineering for options analysis.
    - Train a machine learning model to predict profitable options.
    - Visualize model performance and feature importance.
    - Simulate trading strategies with a backtesting module.
    """)
    # Sidebar configuration
    st.sidebar.title("Settings")
    st.sidebar.write("Configure your analysis parameters")

    # API Key input
    api_key_input = st.sidebar.text_input("Enter your Polygon.io API Key", type="password")
    if api_key_input:
        global API_KEY
        API_KEY = api_key_input
    else:
        st.sidebar.warning("Please enter your API key to proceed")
        st.stop()

    # Ticker input
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL").upper()

    # Date range input
    start_date = st.sidebar.date_input(
        "Start Date",
        value=datetime.now().date() - timedelta(days=365)
    )
    end_date = st.sidebar.date_input(
        "End Date",
        value=datetime.now().date()
    )

    st.write(f"Analyzing options for **{ticker}** from **{start_date}** to **{end_date}**")

    # Additional parameters
    st.sidebar.subheader("Model Parameters")
    return_threshold = st.sidebar.slider(
        "Return Threshold (%) for Positive Outcome",
        min_value=-20,
        max_value=20,
        value=5,
        step=1,
        format="%d%%"
    ) / 100  # Convert to decimal

    probability_threshold = st.sidebar.slider(
        "Probability Threshold for Opportunity",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )

    st.sidebar.subheader("Backtesting Parameters")
    initial_capital = st.sidebar.number_input(
        "Initial Capital ($)",
        min_value=1000,
        value=10000,
        step=1000
    )

    trade_size = st.sidebar.number_input(
        "Trade Size ($)",
        min_value=100,
        value=1000,
        step=100
    )

    if st.sidebar.button("Analyze Options"):
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')

        # 1. Get stock price data
        stock_df = get_stock_price(ticker, from_date, to_date)
        if stock_df is not None:
            st.success(f"Successfully fetched stock data for {ticker}")

            # Display stock price chart
            st.subheader(f"{ticker} Stock Price")
            st.write("""
            The following chart shows the historical stock prices for the selected period.
            This information is essential for calculating features like volatility and momentum.
            """)
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=stock_df.index,
                open=stock_df['o'],
                high=stock_df['h'],
                low=stock_df['l'],
                close=stock_df['c'],
                name=ticker
            ))
            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Price',
                autosize=True,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

            # 2. Get option contracts
            contracts_df = get_option_contracts(ticker)
            if contracts_df is not None:
                st.success("Successfully fetched options contracts")

                # Calculate features
                enhanced_df = calculate_features(contracts_df, stock_df)
                if enhanced_df is not None:
                    # Prepare training data
                    X_original, y_original, X_resampled, y_resampled, simulated_returns = prepare_training_data(
                        enhanced_df, stock_df, return_threshold
                    )
                    if X_resampled is not None and y_resampled is not None:
                        st.write(f"Training data shape: {X_resampled.shape}")
                        st.write("""
                        ### Machine Learning Model Training

                        **Objective:**
                        We aim to build a machine learning model that can predict whether an option contract will yield a return above a specified threshold.

                        **Strategy Thesis:**
                        By analyzing historical data and engineered features, we can identify patterns and indicators that suggest an option is likely to be profitable. The Random Forest classifier is chosen for its ability to handle complex interactions between features and its robustness to overfitting.

                        **What We're Looking For:**
                        - High accuracy in predicting profitable options.
                        - Key features that significantly influence profitability.
                        - A model that can generalize well to unseen data.
                        """)
                        # Split data into train and test sets
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
                        )

                        # Train model
                        model, cv_results = train_model(X_train, y_train)
                        if model is not None:
                            st.success("Model training completed")

                            # Evaluate model
                            evaluate_model(model, X_test, y_test)

                            # Make predictions on the original dataset
                            probabilities = model.predict_proba(X_original)[:, 1]

                            # Assign simulated returns to enhanced_df
                            enhanced_df['simulated_return'] = simulated_returns

                            # Display opportunities analysis
                            display_opportunity_analysis(
                                enhanced_df, probabilities, probability_threshold
                            )

                            # Interpret model
                            st.header("Model Interpretation")
                            st.write("""
                            Understanding which features influence the model's predictions the most can provide insights into the key drivers of option profitability.
                            The SHAP (SHapley Additive exPlanations) values show the impact of each feature on the model's output.
                            """)
                            plot_feature_importance(model, X_original)

                            # Backtesting
                            st.header("Backtesting the Strategy")
                            st.write("""
                            We will now simulate a trading strategy based on the model's predictions.
                            The backtest assumes that we invest a fixed amount in each option predicted to be an opportunity.

                            **Strategy Thesis:**
                            By investing consistently in options identified as high-probability opportunities by our model, we aim to achieve positive returns over time. The backtest helps evaluate the practical performance of this strategy.

                            **What We're Looking For:**
                            - Positive total profit and return percentage.
                            - Consistent capital growth over time.
                            - Insights into the risk and reward profile of the strategy.
                            """)
                            # Run backtest
                            backtest_results, total_profit, total_return_pct = backtest_strategy(
                                enhanced_df, initial_capital, trade_size
                            )

                            # Display backtest results
                            display_backtest_results(backtest_results, total_profit, total_return_pct)

                        else:
                            st.error("Model training failed")
                    else:
                        st.error("Failed to prepare training data")
                else:
                    st.error("Could not calculate features")
            else:
                st.error("Could not fetch options contracts data")
        else:
            st.error("Could not fetch stock price data")

if __name__ == "__main__":
    main()
