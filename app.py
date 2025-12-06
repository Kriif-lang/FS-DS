import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set page configuration
st.set_page_config(page_title="GDP Forecasting Analysis", layout="wide")

st.title("GDP Forecasting Analysis & Macroeconomic Indicators")
st.markdown("""
This application allows you to explore and analyze macroeconomic data, focusing on GDP Growth forecasting.
""")

# File Uploader
uploaded_file = st.file_uploader("Upload CSV Dataset (Optional if file exists locally)", type=["csv"])
local_file_path = "finance_economics_dataset.csv"

df = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
elif os.path.exists(local_file_path):
    try:
        df = pd.read_csv(local_file_path)
        st.info(f"Loaded local dataset: `{local_file_path}`")
    except Exception as e:
        st.error(f"Error loading local file: {e}")

if df is not None:
    try:
        # --- DAY 1: Load Data ---

        st.subheader("1. Raw Data Preview")
        st.write(f"**Shape:** {df.shape}")
        st.dataframe(df.head())

        # --- DAY 3: Data Cleaning ---
        st.subheader("2. Data Cleaning & Processing")
        
        # Convert Date to datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            st.info("Converted 'Date' column to datetime and sorted by date.")
        else:
            st.error("Column 'Date' not found in the dataset.")
            st.stop()

        # Handle Missing Values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        df = df.dropna() # Drop remaining non-numeric missing values
        
        st.write(f"**Shape after cleaning:** {df.shape}")
        st.write("**Missing Values after cleaning:**")
        st.write(df.isnull().sum()[df.isnull().sum() > 0]) # Show only columns with missing values if any

        # --- DAY 4: Data Manipulation ---
        # Set Date as index for plotting
        df_indexed = df.set_index('Date')

        # Select relevant columns
        macro_cols = [
            "GDP Growth (%)",
            "Inflation Rate (%)",
            "Unemployment Rate (%)",
            "Interest Rate (%)"
        ]
        
        # Verify columns exist
        available_macro_cols = [col for col in macro_cols if col in df.columns]
        
        if not available_macro_cols:
            st.warning("Expected macroeconomic columns (GDP Growth, Inflation, Unemployment, Interest) not found.")
        else:
            st.subheader("3. Macroeconomic Indicators Over Time")
            st.dataframe(df_indexed[available_macro_cols].tail())

            # --- DAY 5-6: EDA ---
            st.subheader("4. Exploratory Data Analysis (EDA)")

            # Descriptive Statistics
            st.markdown("### Descriptive Statistics")
            st.write(df[available_macro_cols].describe().T)

            # Time Series Plots
            st.markdown("### Time Series Trends")
            
            # Allow user to select columns to plot
            selected_cols = st.multiselect("Select indicators to plot:", available_macro_cols, default=available_macro_cols)
            
            if selected_cols:
                st.line_chart(df_indexed[selected_cols])
            
            # Correlation Heatmap
            st.markdown("### Correlation Matrix")
            if len(available_macro_cols) > 1:
                corr = df[available_macro_cols].corr()
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
                st.pyplot(fig)
            else:
                st.info("Not enough columns for correlation heatmap.")

            # Stock Index Analysis (if available)
            if 'Stock Index' in df.columns and 'Close Price' in df.columns:
                st.markdown("### Stock Index Trends")
                stock_indices = df['Stock Index'].unique()
                selected_index = st.selectbox("Select Stock Index:", stock_indices)
                
                df_stock = df[df['Stock Index'] == selected_index]
                st.line_chart(df_stock.set_index('Date')['Close Price'])

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

else:
    st.info("Awaiting CSV file upload or local file...")
