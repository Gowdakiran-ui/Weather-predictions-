#!/usr/bin/env python3
"""
EDA and Data Cleaning Script for Weather Prediction Dataset
Author: Kiran gowda.A
Purpose: Exploratory Data Analysis and Data Cleaning for weather prediction data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the weather prediction datasets"""
    print("Loading datasets...")
    
    # Load main weather dataset
    weather_df = pd.read_csv('weather_extracted/weather_prediction_dataset.csv')
    
    # Load BBQ labels dataset
    bbq_df = pd.read_csv('weather_extracted/weather_prediction_bbq_labels.csv')
    
    print(f"Weather dataset shape: {weather_df.shape}")
    print(f"BBQ labels dataset shape: {bbq_df.shape}")
    
    return weather_df, bbq_df

def perform_eda(df, df_name="Dataset"):
    """Perform comprehensive EDA on the dataset"""
    print(f"\n{'='*50}")
    print(f"EXPLORATORY DATA ANALYSIS - {df_name}")
    print(f"{'='*50}")
    
    # Basic information
    print(f"\nDataset Shape: {df.shape}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data types
    print(f"\nData Types:")
    print(df.dtypes.value_counts())
    
    # Missing values
    print(f"\nMissing Values:")
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        missing_percent = (missing_values / len(df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing_values,
            'Missing Percentage': missing_percent
        }).sort_values('Missing Count', ascending=False)
        print(missing_df[missing_df['Missing Count'] > 0])
    else:
        print("No missing values found!")
    
    # Duplicate rows
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate Rows: {duplicates}")
    
    # Basic statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\nNumeric Columns Statistics:")
        print(df[numeric_cols].describe())
    
    return df

def detect_outliers(df, method='iqr'):
    """Detect outliers in numeric columns"""
    print(f"\n{'='*50}")
    print("OUTLIER DETECTION")
    print(f"{'='*50}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove DATE and MONTH from outlier detection
    numeric_cols = [col for col in numeric_cols if col not in ['DATE', 'MONTH']]
    
    outliers_info = {}
    
    for col in numeric_cols:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[col]))
            outliers = df[z_scores > 3]
        
        outliers_info[col] = len(outliers)
    
    # Display outlier summary
    outlier_summary = pd.DataFrame.from_dict(outliers_info, orient='index', columns=['Outlier Count'])
    outlier_summary['Outlier Percentage'] = (outlier_summary['Outlier Count'] / len(df)) * 100
    outlier_summary = outlier_summary.sort_values('Outlier Count', ascending=False)
    
    print(f"Outliers detected using {method.upper()} method:")
    print(outlier_summary.head(10))
    
    return outlier_summary

def create_visualizations(df):
    """Create visualizations for EDA"""
    print(f"\n{'='*50}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*50}")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure for multiple plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Weather Dataset EDA Visualizations', fontsize=16)
    
    # 1. Distribution of temperature variables (using TOURS as example)
    temp_cols = [col for col in df.columns if 'TOURS_temp' in col]
    if temp_cols:
        for i, col in enumerate(temp_cols[:3]):  # Plot first 3 temperature columns
            if i < 3:
                axes[0, 0].hist(df[col], alpha=0.7, label=col.split('_')[-1], bins=30)
        axes[0, 0].set_title('Temperature Distribution (Tours)')
        axes[0, 0].set_xlabel('Temperature (°C)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
    
    # 2. Monthly distribution
    if 'MONTH' in df.columns:
        month_counts = df['MONTH'].value_counts().sort_index()
        axes[0, 1].bar(month_counts.index, month_counts.values)
        axes[0, 1].set_title('Data Distribution by Month')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Count')
    
    # 3. Correlation heatmap (sample of variables)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    sample_cols = [col for col in numeric_cols if 'TOURS' in col][:10]  # Sample 10 TOURS columns
    if len(sample_cols) > 1:
        corr_matrix = df[sample_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   fmt='.2f', ax=axes[1, 0], cbar_kws={'shrink': 0.8})
        axes[1, 0].set_title('Correlation Matrix (Tours Variables)')
    
    # 4. Boxplot for outliers (sample variables)
    sample_cols_box = [col for col in df.columns if 'TOURS_temp' in col][:3]
    if sample_cols_box:
        df[sample_cols_box].boxplot(ax=axes[1, 1])
        axes[1, 1].set_title('Boxplot for Temperature Variables')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('data/eda_visualizations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizations saved to 'data/eda_visualizations.png'")

def clean_data(weather_df, bbq_df):
    """Clean the datasets"""
    print(f"\n{'='*50}")
    print("DATA CLEANING")
    print(f"{'='*50}")
    
    # Make copies for cleaning
    weather_clean = weather_df.copy()
    bbq_clean = bbq_df.copy()
    
    print("Original shapes:")
    print(f"Weather: {weather_df.shape}")
    print(f"BBQ: {bbq_df.shape}")
    
    # 1. Handle DATE column - convert to datetime
    print("\n1. Converting DATE to datetime...")
    weather_clean['DATE'] = pd.to_datetime(weather_clean['DATE'], format='%Y%m%d')
    bbq_clean['DATE'] = pd.to_datetime(bbq_clean['DATE'], format='%Y%m%d')
    
    # 2. Add derived date features
    print("2. Adding derived date features...")
    weather_clean['YEAR'] = weather_clean['DATE'].dt.year
    weather_clean['DAY_OF_YEAR'] = weather_clean['DATE'].dt.dayofyear
    weather_clean['QUARTER'] = weather_clean['DATE'].dt.quarter
    weather_clean['WEEK_OF_YEAR'] = weather_clean['DATE'].dt.isocalendar().week
    
    # 3. Handle missing values (though the dataset appears to have none)
    print("3. Checking for missing values...")
    missing_weather = weather_clean.isnull().sum().sum()
    missing_bbq = bbq_clean.isnull().sum().sum()
    print(f"Missing values in weather data: {missing_weather}")
    print(f"Missing values in BBQ data: {missing_bbq}")
    
    # 4. Remove duplicate rows
    print("4. Removing duplicates...")
    weather_clean = weather_clean.drop_duplicates()
    bbq_clean = bbq_clean.drop_duplicates()
    
    # 5. Data type optimization
    print("5. Optimizing data types...")
    # Convert boolean columns in BBQ dataset
    bool_cols = [col for col in bbq_clean.columns if 'BBQ_weather' in col]
    for col in bool_cols:
        bbq_clean[col] = bbq_clean[col].astype('bool')
    
    # 6. Handle extreme outliers (capping at 99.9 percentile)
    print("6. Handling extreme outliers...")
    numeric_cols = weather_clean.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude date-related columns from outlier treatment
    exclude_cols = ['DATE', 'MONTH', 'YEAR', 'DAY_OF_YEAR', 'QUARTER', 'WEEK_OF_YEAR']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    outlier_count = 0
    for col in numeric_cols:
        lower_cap = weather_clean[col].quantile(0.001)
        upper_cap = weather_clean[col].quantile(0.999)
        
        outliers_before = ((weather_clean[col] < lower_cap) | (weather_clean[col] > upper_cap)).sum()
        
        weather_clean[col] = weather_clean[col].clip(lower=lower_cap, upper=upper_cap)
        outlier_count += outliers_before
    
    print(f"Capped {outlier_count} extreme outliers")
    
    # 7. Feature scaling preparation (we'll do this in feature engineering)
    print("7. Data cleaning completed!")
    
    print("\nCleaned shapes:")
    print(f"Weather: {weather_clean.shape}")
    print(f"BBQ: {bbq_clean.shape}")
    
    return weather_clean, bbq_clean

def save_cleaned_data(weather_clean, bbq_clean):
    """Save cleaned datasets"""
    print(f"\n{'='*50}")
    print("SAVING CLEANED DATA")
    print(f"{'='*50}")
    
    # Save cleaned datasets
    weather_clean.to_csv('data/weather_cleaned.csv', index=False)
    bbq_clean.to_csv('data/bbq_labels_cleaned.csv', index=False)
    
    print("Cleaned datasets saved to:")
    print("- data/weather_cleaned.csv")
    print("- data/bbq_labels_cleaned.csv")
    
    # Create data summary report
    summary_report = f"""
DATA CLEANING SUMMARY REPORT
============================

Original Dataset Shapes:
- Weather Dataset: {weather_clean.shape}
- BBQ Labels Dataset: {bbq_clean.shape}

Data Types:
- Weather Dataset: {weather_clean.dtypes.value_counts().to_dict()}
- BBQ Dataset: {bbq_clean.dtypes.value_counts().to_dict()}

Date Range: {weather_clean['DATE'].min()} to {weather_clean['DATE'].max()}

Features Added:
- YEAR: Year extracted from DATE
- DAY_OF_YEAR: Day of year (1-365/366)
- QUARTER: Quarter of year (1-4)
- WEEK_OF_YEAR: Week of year (1-53)

Data Quality:
- No missing values in either dataset
- Extreme outliers capped at 0.1% and 99.9% percentiles
- Duplicate rows removed
- DATE columns converted to datetime format
- Boolean columns properly typed

Files Saved:
- data/weather_cleaned.csv
- data/bbq_labels_cleaned.csv
- data/eda_visualizations.png
- data/data_cleaning_report.txt
"""
    
    with open('data/data_cleaning_report.txt', 'w') as f:
        f.write(summary_report)
    
    print("Data cleaning report saved to: data/data_cleaning_report.txt")

def main():
    """Main function to run EDA and data cleaning"""
    print("Starting EDA and Data Cleaning Process...")
    print("="*60)
    
    # Load data
    weather_df, bbq_df = load_data()
    
    # Perform EDA
    weather_df = perform_eda(weather_df, "Weather Dataset")
    bbq_df = perform_eda(bbq_df, "BBQ Labels Dataset")
    
    # Detect outliers
    outliers = detect_outliers(weather_df)
    
    # Create visualizations
    create_visualizations(weather_df)
    
    # Clean data
    weather_clean, bbq_clean = clean_data(weather_df, bbq_df)
    
    # Save cleaned data
    save_cleaned_data(weather_clean, bbq_clean)
    
    print("\n" + "="*60)
    print("EDA and Data Cleaning Process Completed Successfully!")
    print("="*60)

if __name__ == "__main__":
    main()