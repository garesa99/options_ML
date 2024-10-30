import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

        # Categorize moneyness
        df['moneyness_category'] = pd.cut(
            df['moneyness'],
            bins=[-np.inf, -0.05, 0.05, np.inf],
            labels=['OTM', 'ATM', 'ITM']
        )

        # Encode moneyness_category
        le = LabelEncoder()
        df['moneyness_encoded'] = le.fit_transform(df['moneyness_category'])

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
    - X (DataFrame): Original feature set.
    - y (Series): Original target variable.
    - X_resampled (DataFrame): Resampled feature set for training.
    - y_resampled (Series): Resampled target variable for training.
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
            return None, None, None, None

        # Create feature matrix
        X = contracts_df[required_features].copy()

        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        # Define target variable based on moneyness_category
        # For demonstration, we'll classify options as ITM (1) vs not ITM (0)
        y = (contracts_df['moneyness_category'] == 'ITM').astype(int)

        # Address class imbalance by resampling
        X_resampled, y_resampled = resample_data(X, y)

        return X, y, X_resampled, y_resampled

    except Exception as e:
        st.error(f"Error in data preparation: {str(e)}")
        return None, None, None, None

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
            x=['Not ITM', 'ITM'],
            y=['Not ITM', 'ITM'],
            text_auto=True,
            color_continuous_scale='Blues',
            title="Confusion Matrix"
        )
        st.plotly_chart(conf_matrix_fig)

        st.write("""
        **How to Interpret the Confusion Matrix:**

        - **True Positives (TP):** Correctly predicted ITM options (bottom-right cell).
        - **True Negatives (TN):** Correctly predicted Not ITM options (top-left cell).
        - **False Positives (FP):** Incorrectly predicted ITM options (top-right cell).
        - **False Negatives (FN):** Incorrectly predicted Not ITM options (bottom-left cell).

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
    - probabilities (array): Predicted probabilities of being an ITM option.
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
    The scores represent the probability that the option is In-The-Money (ITM).
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
#                           MAIN APPLICATION                                  #
###############################################################################

def main():
    st.title("📈 Advanced Options Analysis Tool")
    st.markdown("""
    Welcome to the Advanced Options Analysis Tool! This application helps you analyze option contracts and identify potential trading opportunities using machine learning.

    Created by [Gabriel Reyes - gabriel.reyes@gsom.polimi.it]

    **Key Features:**
    - Fetch and display current stock and options data.
    - Perform feature engineering for options analysis.
    - Train a machine learning model to predict ITM options.
    - Visualize model performance and feature importance.
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
        min_value=-50,
        max_value=50,
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
                    X, y, X_resampled, y_resampled = prepare_training_data(
                        enhanced_df, stock_df, return_threshold
                    )
                    if X_resampled is not None and y_resampled is not None:
                        st.write(f"Training data shape: {X_resampled.shape}")
                        st.write("""
                        ### Machine Learning Model Training

                        **Objective:**
                        We aim to build a machine learning model that can predict whether an option contract is In-The-Money (ITM).

                        **Strategy Thesis:**
                        By analyzing current data and engineered features, we can identify patterns and indicators that suggest an option is ITM. The Random Forest classifier is chosen for its ability to handle complex interactions between features and its robustness to overfitting.

                        **What We're Looking For:**
                        - High accuracy in predicting ITM options.
                        - Key features that significantly influence ITM classification.
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
                            probabilities = model.predict_proba(X)[:, 1]

                            # Assign predicted probabilities to enhanced_df
                            enhanced_df['predicted_probability'] = probabilities

                            # Display opportunities analysis
                            display_opportunity_analysis(
                                enhanced_df, probabilities, probability_threshold
                            )

                            # Interpret model
                            st.header("Model Interpretation")
                            st.write("""
                            Understanding which features influence the model's predictions the most can provide insights into the key drivers of an option being In-The-Money (ITM).
                            The SHAP (SHapley Additive exPlanations) values show the impact of each feature on the model's output.
                            """)
                            plot_feature_importance(model, X)

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
