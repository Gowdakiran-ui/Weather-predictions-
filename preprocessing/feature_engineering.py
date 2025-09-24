#!/usr/bin/env python3
"""
Feature Engineering Script for Weather Prediction Dataset
Author: Kiran gowda.A
Purpose: Advanced feature engineering for weather prediction models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_cleaned_data():
    """Load the cleaned datasets"""
    print("Loading cleaned datasets...")
    
    weather_df = pd.read_csv('data/weather_cleaned.csv')
    bbq_df = pd.read_csv('data/bbq_labels_cleaned.csv')
    
    # Convert DATE back to datetime
    weather_df['DATE'] = pd.to_datetime(weather_df['DATE'])
    bbq_df['DATE'] = pd.to_datetime(bbq_df['DATE'])
    
    print(f"Weather dataset shape: {weather_df.shape}")
    print(f"BBQ labels dataset shape: {bbq_df.shape}")
    
    return weather_df, bbq_df

def create_temporal_features(df):
    """Create advanced temporal features"""
    print(f"\n{'='*50}")
    print("CREATING TEMPORAL FEATURES")
    print(f"{'='*50}")
    
    df_featured = df.copy()
    
    # Cyclical encoding for temporal features
    print("1. Creating cyclical temporal features...")
    
    # Month cyclical features
    df_featured['MONTH_sin'] = np.sin(2 * np.pi * df_featured['MONTH'] / 12)
    df_featured['MONTH_cos'] = np.cos(2 * np.pi * df_featured['MONTH'] / 12)
    
    # Day of year cyclical features
    df_featured['DAY_OF_YEAR_sin'] = np.sin(2 * np.pi * df_featured['DAY_OF_YEAR'] / 365)
    df_featured['DAY_OF_YEAR_cos'] = np.cos(2 * np.pi * df_featured['DAY_OF_YEAR'] / 365)
    
    # Week of year cyclical features
    df_featured['WEEK_sin'] = np.sin(2 * np.pi * df_featured['WEEK_OF_YEAR'] / 52)
    df_featured['WEEK_cos'] = np.cos(2 * np.pi * df_featured['WEEK_OF_YEAR'] / 52)
    
    # Season features
    print("2. Creating seasonal features...")
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'
    
    df_featured['SEASON'] = df_featured['MONTH'].apply(get_season)
    
    # One-hot encode seasons
    season_dummies = pd.get_dummies(df_featured['SEASON'], prefix='SEASON')
    df_featured = pd.concat([df_featured, season_dummies], axis=1)
    
    print(f"Added temporal features. New shape: {df_featured.shape}")
    return df_featured

def create_weather_features(df):
    """Create weather-specific engineered features"""
    print(f"\n{'='*50}")
    print("CREATING WEATHER-SPECIFIC FEATURES")
    print(f"{'='*50}")
    
    df_featured = df.copy()
    
    # Get all city names
    cities = []
    for col in df.columns:
        if '_' in col and col not in ['DATE', 'MONTH', 'YEAR', 'DAY_OF_YEAR', 'QUARTER', 'WEEK_OF_YEAR']:
            city = col.split('_')[0]
            if city not in cities and city not in ['MONTH', 'DAY', 'YEAR', 'QUARTER', 'WEEK', 'SEASON']:
                cities.append(city)
    
    print(f"Found {len(cities)} cities: {cities}")
    
    # Create features for each city
    for city in cities:
        print(f"Creating features for {city}...")
        
        # Temperature features
        temp_cols = [col for col in df.columns if col.startswith(f'{city}_temp')]
        if len(temp_cols) >= 2:
            # Temperature range
            if f'{city}_temp_max' in df.columns and f'{city}_temp_min' in df.columns:
                df_featured[f'{city}_temp_range'] = df[f'{city}_temp_max'] - df[f'{city}_temp_min']
            
            # Temperature anomaly (deviation from mean)
            if f'{city}_temp_mean' in df.columns:
                df_featured[f'{city}_temp_anomaly'] = df[f'{city}_temp_mean'] - df[f'{city}_temp_mean'].mean()
        
        # Wind features
        if f'{city}_wind_speed' in df.columns and f'{city}_wind_gust' in df.columns:
            # Wind gust factor
            df_featured[f'{city}_wind_gust_factor'] = df[f'{city}_wind_gust'] / (df[f'{city}_wind_speed'] + 0.001)
        
        # Comfort index (simple heat index approximation)
        if f'{city}_temp_mean' in df.columns and f'{city}_humidity' in df.columns:
            df_featured[f'{city}_comfort_index'] = df[f'{city}_temp_mean'] - (df[f'{city}_humidity'] * 0.1)
        
        # Precipitation intensity
        if f'{city}_precipitation' in df.columns:
            df_featured[f'{city}_precip_intensity'] = np.where(
                df[f'{city}_precipitation'] > 0, 
                np.log1p(df[f'{city}_precipitation']), 
                0
            )
        
        # Weather favorability score (for BBQ weather prediction)
        weather_score_components = []
        
        if f'{city}_temp_mean' in df.columns:
            # Temperature component (optimal around 20-25°C)
            temp_score = 1 - np.abs(df[f'{city}_temp_mean'] - 22.5) / 22.5
            weather_score_components.append(temp_score.clip(0, 1))
        
        if f'{city}_precipitation' in df.columns:
            # Precipitation component (less is better)
            precip_score = np.exp(-df[f'{city}_precipitation'])
            weather_score_components.append(precip_score)
        
        if f'{city}_cloud_cover' in df.columns:
            # Cloud cover component (less is better)
            cloud_score = 1 - (df[f'{city}_cloud_cover'] / 8)
            weather_score_components.append(cloud_score.clip(0, 1))
        
        if f'{city}_humidity' in df.columns:
            # Humidity component (moderate is better)
            humidity_score = 1 - np.abs(df[f'{city}_humidity'] - 0.6) / 0.6
            weather_score_components.append(humidity_score.clip(0, 1))
        
        if weather_score_components:
            df_featured[f'{city}_weather_favorability'] = np.mean(weather_score_components, axis=0)
    
    print(f"Weather features created. New shape: {df_featured.shape}")
    return df_featured

def create_rolling_features(df, window_sizes=[3, 7, 14]):
    """Create rolling window features"""
    print(f"\n{'='*50}")
    print("CREATING ROLLING WINDOW FEATURES")
    print(f"{'='*50}")
    
    df_featured = df.copy()
    df_featured = df_featured.sort_values('DATE').reset_index(drop=True)
    
    # Get numeric columns (excluding engineered features)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    base_cols = [col for col in numeric_cols if not any(suffix in col for suffix in 
                ['_sin', '_cos', '_range', '_anomaly', '_factor', '_index', '_intensity', '_favorability'])]
    
    # Select key columns for rolling features (to avoid too many features)
    key_patterns = ['temp_mean', 'temp_max', 'temp_min', 'precipitation', 'humidity', 'pressure']
    rolling_cols = []
    for pattern in key_patterns:
        rolling_cols.extend([col for col in base_cols if pattern in col])
    
    print(f"Creating rolling features for {len(rolling_cols)} columns with windows: {window_sizes}")
    
    for window in window_sizes:
        print(f"Processing {window}-day rolling window...")
        for col in rolling_cols[:20]:  # Limit to prevent too many features
            # Rolling mean
            df_featured[f'{col}_rolling_mean_{window}d'] = df_featured[col].rolling(window=window, min_periods=1).mean()
            
            # Rolling standard deviation
            df_featured[f'{col}_rolling_std_{window}d'] = df_featured[col].rolling(window=window, min_periods=1).std()
            
            # Rolling min and max for temperature columns
            if 'temp' in col:
                df_featured[f'{col}_rolling_min_{window}d'] = df_featured[col].rolling(window=window, min_periods=1).min()
                df_featured[f'{col}_rolling_max_{window}d'] = df_featured[col].rolling(window=window, min_periods=1).max()
    
    print(f"Rolling features created. New shape: {df_featured.shape}")
    return df_featured

def create_lag_features(df, lags=[1, 3, 7]):
    """Create lag features for time series prediction"""
    print(f"\n{'='*50}")
    print("CREATING LAG FEATURES")
    print(f"{'='*50}")
    
    df_featured = df.copy()
    df_featured = df_featured.sort_values('DATE').reset_index(drop=True)
    
    # Select key columns for lag features
    key_patterns = ['temp_mean', 'precipitation', 'pressure', 'humidity']
    lag_cols = []
    for pattern in key_patterns:
        lag_cols.extend([col for col in df.columns if pattern in col and '_rolling_' not in col])
    
    print(f"Creating lag features for {len(lag_cols)} columns with lags: {lags}")
    
    for lag in lags:
        print(f"Processing {lag}-day lag...")
        for col in lag_cols[:15]:  # Limit to prevent too many features
            df_featured[f'{col}_lag_{lag}d'] = df_featured[col].shift(lag)
    
    # Fill NaN values created by lagging with forward fill
    df_featured = df_featured.fillna(method='ffill').fillna(method='bfill')
    
    print(f"Lag features created. New shape: {df_featured.shape}")
    return df_featured

def create_interaction_features(df):
    """Create interaction features between key variables"""
    print(f"\n{'='*50}")
    print("CREATING INTERACTION FEATURES")
    print(f"{'='*50}")
    
    df_featured = df.copy()
    
    # Find temperature and humidity columns for each city
    cities = []
    for col in df.columns:
        if '_temp_mean' in col:
            city = col.replace('_temp_mean', '')
            cities.append(city)
    
    print(f"Creating interaction features for cities: {cities[:5]}...")  # Show first 5
    
    for city in cities[:5]:  # Limit to first 5 cities to prevent feature explosion
        # Temperature-Humidity interaction
        temp_col = f'{city}_temp_mean'
        humidity_col = f'{city}_humidity'
        
        if temp_col in df.columns and humidity_col in df.columns:
            df_featured[f'{city}_temp_humidity_interaction'] = df[temp_col] * df[humidity_col]
        
        # Temperature-Pressure interaction
        pressure_col = f'{city}_pressure'
        if temp_col in df.columns and pressure_col in df.columns:
            df_featured[f'{city}_temp_pressure_interaction'] = df[temp_col] * df[pressure_col]
    
    print(f"Interaction features created. New shape: {df_featured.shape}")
    return df_featured

def apply_scaling(df, scaling_method='standard'):
    """Apply feature scaling"""
    print(f"\n{'='*50}")
    print(f"APPLYING {scaling_method.upper()} SCALING")
    print(f"{'='*50}")
    
    df_scaled = df.copy()
    
    # Select numeric columns for scaling (exclude DATE and boolean columns)
    numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_scale = [col for col in numeric_cols if col not in ['YEAR', 'MONTH', 'DAY_OF_YEAR', 'QUARTER', 'WEEK_OF_YEAR']]
    
    print(f"Scaling {len(cols_to_scale)} numeric features...")
    
    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
    else:
        print("Invalid scaling method. Using StandardScaler.")
        scaler = StandardScaler()
    
    # Fit and transform
    df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])
    
    # Save scaler for future use
    import joblib
    joblib.dump(scaler, f'data/{scaling_method}_scaler.pkl')
    print(f"Scaler saved as: data/{scaling_method}_scaler.pkl")
    
    return df_scaled, scaler

def perform_feature_selection(df, target_col=None, method='mutual_info', k=50):
    """Perform feature selection"""
    print(f"\n{'='*50}")
    print("PERFORMING FEATURE SELECTION")
    print(f"{'='*50}")
    
    if target_col is None or target_col not in df.columns:
        print("No valid target column provided. Skipping feature selection.")
        return df, None
    
    # Prepare features and target - only use numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in ['DATE', target_col]]
    X = df[feature_cols]
    y = df[target_col]
    
    # Convert boolean target to numeric if needed
    if y.dtype == 'bool':
        y = y.astype(int)
    
    print(f"Selecting top {k} features from {len(feature_cols)} using {method}...")
    
    if method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_regression, k=min(k, len(feature_cols)))
    else:
        selector = SelectKBest(score_func=f_regression, k=min(k, len(feature_cols)))
    
    # Fit selector
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    support_indices = selector.get_support(indices=True)
    selected_features = [feature_cols[i] for i in support_indices] if support_indices is not None else []
    
    # Create dataframe with selected features - include non-numeric columns too
    non_numeric_cols = [col for col in df.columns if col not in numeric_cols and col != target_col]
    df_selected = df[['DATE'] + non_numeric_cols + selected_features + [target_col]].copy()
    
    # Save feature importance scores
    feature_scores = pd.DataFrame({
        'Feature': feature_cols,
        'Score': selector.scores_
    }).sort_values('Score', ascending=False)
    
    feature_scores.to_csv('data/feature_importance_scores.csv', index=False)
    
    print(f"Selected {len(selected_features)} features")
    print("Top 10 features:")
    print(feature_scores.head(10))
    
    return df_selected, selected_features

def create_pca_features(df, n_components=10):
    """Create PCA features"""
    print(f"\n{'='*50}")
    print("CREATING PCA FEATURES")
    print(f"{'='*50}")
    
    # Select numeric columns for PCA
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    pca_cols = [col for col in numeric_cols if col not in ['YEAR', 'MONTH', 'DAY_OF_YEAR', 'QUARTER', 'WEEK_OF_YEAR']]
    
    print(f"Applying PCA to {len(pca_cols)} features, extracting {n_components} components...")
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(df[pca_cols])
    
    # Create PCA feature dataframe
    pca_columns = [f'PCA_{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(pca_features, columns=pca_columns)
    
    # Add PCA features to original dataframe
    df_with_pca = pd.concat([df.reset_index(drop=True), pca_df], axis=1)
    
    # Save PCA model
    import joblib
    joblib.dump(pca, 'data/pca_model.pkl')
    
    # Print explained variance
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")
    
    return df_with_pca, pca

def merge_with_target(weather_df, bbq_df, target_city='TOURS'):
    """Merge weather data with BBQ target labels"""
    print(f"\n{'='*50}")
    print("MERGING WITH TARGET LABELS")
    print(f"{'='*50}")
    
    target_col = f'{target_city}_BBQ_weather'
    
    if target_col not in bbq_df.columns:
        print(f"Target column {target_col} not found. Available columns: {bbq_df.columns.tolist()}")
        return weather_df
    
    # Merge on DATE
    merged_df = weather_df.merge(bbq_df[['DATE', target_col]], on='DATE', how='left')
    
    print(f"Merged dataset shape: {merged_df.shape}")
    print(f"Target distribution:")
    print(merged_df[target_col].value_counts())
    
    return merged_df

def save_engineered_data(df, filename='weather_engineered.csv'):
    """Save the final engineered dataset"""
    print(f"\n{'='*50}")
    print("SAVING ENGINEERED DATA")
    print(f"{'='*50}")
    
    # Save the dataset
    filepath = f'data/{filename}'
    df.to_csv(filepath, index=False)
    
    print(f"Engineered dataset saved to: {filepath}")
    print(f"Final dataset shape: {df.shape}")
    
    # Create feature engineering report
    feature_report = f"""
FEATURE ENGINEERING SUMMARY REPORT
==================================

Final Dataset Shape: {df.shape}

Feature Categories Created:
1. Temporal Features:
   - Cyclical encoding (sin/cos) for months, days, weeks
   - Seasonal indicators
   - Date-based features

2. Weather-Specific Features:
   - Temperature ranges and anomalies
   - Wind gust factors
   - Comfort indices
   - Precipitation intensity
   - Weather favorability scores

3. Rolling Window Features:
   - 3, 7, and 14-day rolling means and standard deviations
   - Rolling min/max for temperature variables

4. Lag Features:
   - 1, 3, and 7-day lag features for key variables

5. Interaction Features:
   - Temperature-humidity interactions
   - Temperature-pressure interactions

6. PCA Features:
   - Principal components for dimensionality reduction

Data Processing:
- Feature scaling applied
- Missing values handled
- Feature selection performed
- Target variables merged

Files Created:
- {filepath}
- data/standard_scaler.pkl (if scaling applied)
- data/pca_model.pkl (if PCA applied)
- data/feature_importance_scores.csv (if feature selection applied)
- data/feature_engineering_report.txt

Total Features: {df.shape[1]}
"""
    
    with open('data/feature_engineering_report.txt', 'w') as f:
        f.write(feature_report)
    
    print("Feature engineering report saved to: data/feature_engineering_report.txt")

def main():
    """Main function to run feature engineering"""
    print("Starting Feature Engineering Process...")
    print("="*60)
    
    # Load cleaned data
    weather_df, bbq_df = load_cleaned_data()
    
    # Create temporal features
    weather_df = create_temporal_features(weather_df)
    
    # Create weather-specific features
    weather_df = create_weather_features(weather_df)
    
    # Create rolling features
    weather_df = create_rolling_features(weather_df, window_sizes=[3, 7])
    
    # Create lag features
    weather_df = create_lag_features(weather_df, lags=[1, 3])
    
    # Create interaction features
    weather_df = create_interaction_features(weather_df)
    
    # Merge with target labels
    final_df = merge_with_target(weather_df, bbq_df, target_city='TOURS')
    
    # Apply scaling
    final_df_scaled, scaler = apply_scaling(final_df, scaling_method='standard')
    
    # Create PCA features
    final_df_with_pca, pca = create_pca_features(final_df_scaled, n_components=10)
    
    # Perform feature selection (if target column exists)
    target_col = 'TOURS_BBQ_weather'
    if target_col in final_df_with_pca.columns:
        final_df_selected, selected_features = perform_feature_selection(
            final_df_with_pca, target_col=target_col, method='mutual_info', k=100
        )
        save_engineered_data(final_df_selected, 'weather_engineered_selected.csv')
    
    # Save full engineered dataset
    save_engineered_data(final_df_with_pca, 'weather_engineered_full.csv')
    
    print("\n" + "="*60)
    print("Feature Engineering Process Completed Successfully!")
    print("="*60)

if __name__ == "__main__":
    main()