import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import math

# LSTM imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Konfigurasi halaman
st.set_page_config(
    page_title="GDP Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fungsi untuk memuat data
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif os.path.exists("finance_economics_dataset.csv"):
        df = pd.read_csv("finance_economics_dataset.csv")
    else:
        return None
    
    # Membersihkan nama kolom
    df.columns = df.columns.str.strip()
    
    # Konversi kolom tanggal
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
    
    # Handle Missing Values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df = df.dropna()
    
    # Membuat kolom Year-Month untuk filtering
    df['Year-Month'] = df['Date'].dt.to_period('M').astype(str)
    
    return df

# Memuat data
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])
df = load_data(uploaded_file)

if df is not None:
    # Sidebar untuk filter
    st.sidebar.header("Filter Data")
    
    # Filter Stock Index jika ada
    if 'Stock Index' in df.columns:
        stock_options = ['All'] + list(df['Stock Index'].unique())
        selected_stock = st.sidebar.selectbox("Pilih Stock Index", stock_options)
    else:
        selected_stock = 'All'
    
    # Filter Tanggal
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    selected_date_range = st.sidebar.date_input("Pilih Rentang Tanggal", [min_date, max_date])
    
    # Filter untuk indikator ekonomi
    macro_cols = ["GDP Growth (%)", "Inflation Rate (%)", "Unemployment Rate (%)", "Interest Rate (%)"]
    available_macro_cols = [col for col in macro_cols if col in df.columns]
    
    if available_macro_cols:
        selected_indicators = st.sidebar.multiselect("Pilih Indikator untuk Analisis", 
                                                    available_macro_cols, 
                                                    default=available_macro_cols)
    else:
        selected_indicators = []
    
    # Tombol Apply Filter
    apply_filter = st.sidebar.button("Terapkan Filter")
    
    # Menerapkan filter
    filtered_df = df.copy()
    
    if apply_filter or True:  # Auto-apply filter saat pertama kali load
        if selected_stock != 'All' and 'Stock Index' in df.columns:
            filtered_df = filtered_df[filtered_df['Stock Index'] == selected_stock]
        
        # Filter tanggal
        start_date = pd.to_datetime(selected_date_range[0])
        end_date = pd.to_datetime(selected_date_range[1])
        filtered_df = filtered_df[(filtered_df['Date'] >= start_date) & (filtered_df['Date'] <= end_date)]
    
    # Header Dashboard
    st.title("📊 GDP Forecasting Dashboard")
    st.markdown("### Analisis Indikator Ekonomi Makro dan Peramalan GDP")
    st.markdown("---")
    
    # Metrics Cards
    if selected_indicators:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if "GDP Growth (%)" in selected_indicators:
                avg_gdp = filtered_df["GDP Growth (%)"].mean()
                st.metric("Rata-rata GDP Growth", f"{avg_gdp:.2f}%")
        
        with col2:
            if "Inflation Rate (%)" in selected_indicators:
                avg_inflation = filtered_df["Inflation Rate (%)"].mean()
                st.metric("Rata-rata Inflasi", f"{avg_inflation:.2f}%")
        
        with col3:
            if "Unemployment Rate (%)" in selected_indicators:
                avg_unemployment = filtered_df["Unemployment Rate (%)"].mean()
                st.metric("Rata-rata Pengangguran", f"{avg_unemployment:.2f}%")
        
        with col4:
            if "Interest Rate (%)" in selected_indicators:
                avg_interest = filtered_df["Interest Rate (%)"].mean()
                st.metric("Rata-rata Suku Bunga", f"{avg_interest:.2f}%")
    
    st.markdown("---")
    
    # Tab untuk visualisasi
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Overview", "📈 Trend Analysis", "🔮 Forecasting", "📋 Model Performance", "📄 Detail Data"])
    
    with tab1:
        # Data Overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Statistik Deskriptif")
            if selected_indicators:
                st.dataframe(filtered_df[selected_indicators].describe().T)
        
        with col2:
            st.subheader("Korelasi Indikator")
            if len(selected_indicators) > 1:
                corr_matrix = filtered_df[selected_indicators].corr()
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    title="Matriks Korelasi"
                )
                st.plotly_chart(fig_corr, use_container_width=True)
        
        # Time series plot
        if selected_indicators:
            st.subheader("Trend Indikator Ekonomi")
            fig_trend = go.Figure()
            
            for indicator in selected_indicators:
                fig_trend.add_trace(go.Scatter(
                    x=filtered_df['Date'],
                    y=filtered_df[indicator],
                    mode='lines+markers',
                    name=indicator,
                    line=dict(width=2)
                ))
            
            fig_trend.update_layout(
                title="Trend Indikator Ekonomi Makro",
                xaxis_title="Tanggal",
                yaxis_title="Nilai (%)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_trend, use_container_width=True)
    
    with tab2:
        # Trend Analysis
        if selected_indicators:
            # Monthly aggregation
            monthly_data = filtered_df.groupby('Year-Month')[selected_indicators].mean().reset_index()
            monthly_data['Date'] = pd.to_datetime(monthly_data['Year-Month'])
            monthly_data = monthly_data.sort_values('Date')
            
            # Create subplots for each indicator
            for indicator in selected_indicators:
                st.subheader(f"Trend Bulanan - {indicator}")
                
                fig_monthly = go.Figure()
                fig_monthly.add_trace(go.Scatter(
                    x=monthly_data['Date'],
                    y=monthly_data[indicator],
                    mode='lines+markers',
                    name=indicator,
                    line=dict(width=3),
                    marker=dict(size=8)
                ))
                
                # Add moving average
                monthly_data[f'{indicator}_MA3'] = monthly_data[indicator].rolling(window=3).mean()
                fig_monthly.add_trace(go.Scatter(
                    x=monthly_data['Date'],
                    y=monthly_data[f'{indicator}_MA3'],
                    mode='lines',
                    name=f'{indicator} (MA 3)',
                    line=dict(width=2, dash='dash')
                ))
                
                fig_monthly.update_layout(
                    title=f"Trend Bulanan {indicator} dengan Moving Average",
                    xaxis_title="Tanggal",
                    yaxis_title="Nilai (%)",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_monthly, use_container_width=True)
    
    with tab3:
        # Forecasting Section
        st.subheader("Model Forecasting")
        
        if "GDP Growth (%)" in selected_indicators:
            # Feature Engineering
            fe_df = filtered_df.set_index('Date')[selected_indicators].copy()
            
            # Create lag features
            for col in selected_indicators:
                fe_df[f"{col}_lag1"] = fe_df[col].shift(1)
                fe_df[f"{col}_lag2"] = fe_df[col].shift(2)
            
            # Rolling features
            fe_df["GDP_Growth_rolling3"] = fe_df["GDP Growth (%)"].rolling(window=3).mean()
            fe_df = fe_df.dropna()
            
            # Prepare data for modeling
            target_col = "GDP Growth (%)"
            feature_cols = [c for c in fe_df.columns if c != target_col]
            
            X = fe_df[feature_cols].values
            y = fe_df[target_col].values
            
            # Time series split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train Random Forest
            rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
            rf.fit(X_train, y_train)
            
            y_pred_rf = rf.predict(X_test)
            
            # Create prediction visualization
            test_dates = fe_df.index[split_idx:]
            
            fig_forecast = go.Figure()
            
            # Actual values
            fig_forecast.add_trace(go.Scatter(
                x=test_dates,
                y=y_test,
                mode='lines+markers',
                name='Actual GDP Growth',
                line=dict(color='blue', width=2)
            ))
            
            # Predicted values
            fig_forecast.add_trace(go.Scatter(
                x=test_dates,
                y=y_pred_rf,
                mode='lines+markers',
                name='Predicted GDP Growth (RF)',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig_forecast.update_layout(
                title="GDP Growth Forecasting - Actual vs Predicted",
                xaxis_title="Tanggal",
                yaxis_title="GDP Growth (%)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': rf.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            st.subheader("Feature Importance")
            fig_importance = px.bar(
                feature_importance,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 10 Feature Importance"
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        else:
            st.warning("Pilih 'GDP Growth (%)' untuk melakukan forecasting")
    
    with tab4:
        # Model Performance
        st.subheader("Evaluasi Model")
        
        if "GDP Growth (%)" in selected_indicators:
            # Metrics calculation
            mae = mean_absolute_error(y_test, y_pred_rf)
            rmse = math.sqrt(mean_squared_error(y_test, y_pred_rf))
            r2 = r2_score(y_test, y_pred_rf)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("MAE", f"{mae:.4f}")
            
            with col2:
                st.metric("RMSE", f"{rmse:.4f}")
            
            with col3:
                st.metric("R² Score", f"{r2:.4f}")
            
            # Residual Analysis
            residuals = y_test - y_pred_rf
            
            fig_residuals = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Residuals vs Predicted", "Residuals Distribution", 
                              "Actual vs Predicted", "Residuals Over Time"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Residuals vs Predicted
            fig_residuals.add_trace(
                go.Scatter(x=y_pred_rf, y=residuals, mode='markers', name='Residuals'),
                row=1, col=1
            )
            fig_residuals.add_trace(
                go.Scatter(x=y_pred_rf, y=[0]*len(y_pred_rf), mode='lines', name='Zero Line'),
                row=1, col=1
            )
            
            # Residuals Distribution
            fig_residuals.add_trace(
                go.Histogram(x=residuals, name='Residuals Distribution'),
                row=1, col=2
            )
            
            # Actual vs Predicted
            fig_residuals.add_trace(
                go.Scatter(x=y_test, y=y_pred_rf, mode='markers', name='Predictions'),
                row=2, col=1
            )
            fig_residuals.add_trace(
                go.Scatter(x=y_test, y=y_test, mode='lines', name='Perfect Fit'),
                row=2, col=1
            )
            
            # Residuals Over Time
            fig_residuals.add_trace(
                go.Scatter(x=test_dates, y=residuals, mode='lines+markers', name='Residuals'),
                row=2, col=2
            )
            
            fig_residuals.update_layout(height=800, showlegend=False)
            st.plotly_chart(fig_residuals, use_container_width=True)
    
    with tab5:
        # Detail Data
        st.subheader("Detail Data Ekonomi")
        
        # Data summary
        st.write("Ringkasan Dataset:")
        st.write(f"Total Baris: {len(filtered_df)}")
        st.write(f"Total Kolom: {len(filtered_df.columns)}")
        st.write(f"Rentang Tanggal: {filtered_df['Date'].min().date()} hingga {filtered_df['Date'].max().date()}")
        
        # Data table with sorting
        st.write("Data Lengkap:")
        
        sort_column = st.selectbox("Pilih kolom untuk sorting", filtered_df.columns)
        sort_order = st.radio("Urutan", ("Ascending", "Descending"))
        
        if sort_order == "Ascending":
            sorted_df = filtered_df.sort_values(by=sort_column, ascending=True)
        else:
            sorted_df = filtered_df.sort_values(by=sort_column, ascending=False)
        
        # Pagination
        page_size = 10
        page = st.number_input("Halaman", min_value=1, max_value=(len(sorted_df) // page_size) + 1, value=1)
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        st.dataframe(sorted_df.iloc[start_idx:end_idx], use_container_width=True)
        
        # Download option
        csv = sorted_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name='filtered_economic_data.csv',
            mime='text/csv'
        )

else:
    st.error("Tidak ada data yang tersedia. Silakan upload file CSV atau pastikan file 'finance_economics_dataset.csv' ada di direktori yang sama.")

# Footer
st.markdown("---")
st.markdown("GDP Forecasting Dashboard © 2024")
