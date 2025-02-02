import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class SupplyChainOptimizer:
    def __init__(self):
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        
    def load_and_clean_data(self, data_path):
        """
        Load and clean supply chain data
        """
        # Load data
        self.data = pd.read_csv("C:\\Users\\Dell\\Downloads\\supply_chain_sample_data.csv")
        
        # Basic cleaning
        self.data = self.data.dropna()  # Remove missing values
        
        # Convert date columns to datetime
        date_columns = self.data.select_dtypes(include=['object']).columns
        for col in date_columns:
            try:
                self.data[col] = pd.to_datetime(self.data[col])
            except:
                continue
                
        # Calculate delay in days if shipping_date and delivery_date exist
        if 'shipping_date' in self.data.columns and 'delivery_date' in self.data.columns:
            self.data['delay_days'] = (self.data['delivery_date'] - 
                                     self.data['shipping_date']).dt.days
        
        return self.data
    
    def perform_eda(self):
        """
        Perform Exploratory Data Analysis
        """
        eda_results = {}
        
        # Basic statistics
        eda_results['basic_stats'] = self.data.describe()
        
        # Correlation matrix
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        eda_results['correlation'] = self.data[numerical_cols].corr()
        corr_matrix=eda_results['correlation']
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr_matrix,annot=True)
            st.pyplot(fig)
        except:
            print("try error")    
        
        # Distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        for idx, col in enumerate(numerical_cols[:4]):  # Plot first 4 numerical columns
            row = idx // 2
            col_idx = idx % 2
            sns.histplot(self.data[col], ax=axes[row, col_idx])
            axes[row, col_idx].set_title(f'Distribution of {col}')
        eda_results['distribution_plot'] = fig
        
        return eda_results
    
    def prepare_features(self):
        """
        Prepare features for modeling
        """
        # Select features (customize based on your data)
        feature_cols = self.data.select_dtypes(include=[np.number]).columns
        feature_cols = feature_cols.drop('delay_days') if 'delay_days' in feature_cols else feature_cols
        
        X = self.data[feature_cols]
        y = self.data['delay_days'] if 'delay_days' in self.data.columns else None
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, feature_cols
    
    def build_model(self, X, y):
        """
        Build and train the prediction model
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        return metrics, (X_test, y_test, y_pred)
    
    def create_gui():
        """
        Create Streamlit GUI
        """
        st.title('Supply Chain Delay Prediction System')
        
        # File upload
        uploaded_file = st.file_uploader("Upload your supply chain data (CSV)", type="csv")
        
        if uploaded_file is not None:
            optimizer = SupplyChainOptimizer()
            
            # Load and clean data
            data = optimizer.load_and_clean_data(uploaded_file)
            st.write("Data Preview:", data.head())
            
            # EDA
            if st.button('Perform EDA'):
                eda_results = optimizer.perform_eda()
                st.write("Basic Statistics:", eda_results['basic_stats'])
                st.write("Correlation Matrix:", eda_results['correlation'])
                st.pyplot(eda_results['distribution_plot'])
            
            # Model Building
            if st.button('Build Model'):
                X, y, feature_cols = optimizer.prepare_features()
                metrics, (X_test, y_test, y_pred) = optimizer.build_model(X, y)
                
                st.write("Model Performance Metrics:")
                st.write(f"Mean Squared Error: {metrics['mse']:.2f}")
                st.write(f"Root Mean Squared Error: {metrics['rmse']:.2f}")
                st.write(f"R² Score: {metrics['r2']:.2f}")
                
                # Plot actual vs predicted
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                ax.set_xlabel('Actual Delay')
                ax.set_ylabel('Predicted Delay')
                ax.set_title('Actual vs Predicted Delays')
                st.pyplot(fig)
                
                # Feature importance
                importance_df = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': optimizer.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                st.write("Feature Importance:")
                st.bar_chart(importance_df.set_index('feature'))

if __name__ == "__main__":
    SupplyChainOptimizer.create_gui()