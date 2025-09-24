#!/usr/bin/env python3
"""
MLflow Model Upload Script for Weather Prediction
Author: Kiran gowda.A
Purpose: Train best model and upload to MLflow with proper Windows configuration
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# MLflow
import mlflow
import mlflow.sklearn
import mlflow.xgboost

def setup_mlflow():
    """Setup MLflow with local tracking"""
    print("Setting up MLflow...")
    
    # Use local directory for tracking
    tracking_uri = "./mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    
    # Create experiment
    experiment_name = "Weather_BBQ_Prediction"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"Created new experiment: {experiment_name}")
        else:
            experiment_id = experiment.experiment_id
            print(f"Using existing experiment: {experiment_name}")
        
        mlflow.set_experiment(experiment_name)
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        print(f"Experiment ID: {experiment_id}")
        return True
        
    except Exception as e:
        print(f"Error setting up MLflow: {e}")
        return False

def load_and_prepare_data():
    """Load and prepare data quickly"""
    print("Loading data...")
    df = pd.read_csv('data/weather_engineered_selected.csv')
    
    # Remove date columns
    date_cols = [col for col in df.columns if 'DATE' in col]
    df = df.drop(columns=date_cols, errors='ignore')
    
    # Handle categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        le = LabelEncoder()
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Separate features and target
    X = df.drop(columns=['TOURS_BBQ_weather'])
    y = df['TOURS_BBQ_weather'].astype(int)
    
    return X, y

def train_best_models(X_train, X_test, y_train, y_test):
    """Train the top performing models"""
    print("Training best models...")
    
    models = {
        'XGBoost': XGBClassifier(
            n_estimators=200, 
            learning_rate=0.1, 
            max_depth=5, 
            random_state=42, 
            eval_metric='logloss'
        ),
        'Gradient_Boosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        results[name] = {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"{name} - AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1_score']:.4f}")
    
    return results

def log_model_to_mlflow(model_name, model, metrics, X_test, feature_names):
    """Log model to MLflow"""
    print(f"Logging {model_name} to MLflow...")
    
    with mlflow.start_run(run_name=f"{model_name}_Weather_Prediction"):
        # Log parameters
        if hasattr(model, 'get_params'):
            params = model.get_params()
            mlflow.log_params(params)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log additional info
        mlflow.log_param("model_type", type(model).__name__)
        mlflow.log_param("features_count", len(feature_names))
        mlflow.log_param("dataset_size", len(X_test))
        mlflow.log_param("training_date", datetime.now().isoformat())
        
        # Log model
        if 'XGB' in model_name:
            mlflow.xgboost.log_model(model, "model", 
                                   input_example=X_test.iloc[:5],
                                   signature=mlflow.models.infer_signature(X_test, model.predict(X_test)))
        else:
            mlflow.sklearn.log_model(model, "model",
                                   input_example=X_test.iloc[:5],
                                   signature=mlflow.models.infer_signature(X_test, model.predict(X_test)))
        
        # Log feature importance if available
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Save and log feature importance
            importance_file = f'data/{model_name.lower()}_feature_importance.csv'
            feature_importance.to_csv(importance_file, index=False)
            mlflow.log_artifact(importance_file)
        
        # Log model comparison results if exists
        if os.path.exists('data/model_comparison_results.csv'):
            mlflow.log_artifact('data/model_comparison_results.csv')
        
        run_id = mlflow.active_run().info.run_id
        artifact_uri = mlflow.get_artifact_uri()
        
        print(f"Model logged successfully!")
        print(f"Run ID: {run_id}")
        print(f"Artifact URI: {artifact_uri}")
        
        return run_id, artifact_uri

def save_production_model(model_name, model, metrics, feature_names):
    """Save model for production use"""
    print(f"Saving {model_name} for production...")
    
    # Create models directory
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model
    model_file = os.path.join(models_dir, f'{model_name.lower()}_production_model.pkl')
    joblib.dump(model, model_file)
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'model_type': type(model).__name__,
        'metrics': metrics,
        'feature_names': feature_names,
        'training_date': datetime.now().isoformat(),
        'model_file': model_file
    }
    
    metadata_file = os.path.join(models_dir, f'{model_name.lower()}_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"Production model saved: {model_file}")
    print(f"Metadata saved: {metadata_file}")
    
    return model_file, metadata_file

def start_mlflow_ui():
    """Start MLflow UI server"""
    print("\n" + "="*60)
    print("STARTING MLFLOW UI")
    print("="*60)
    print("To view the MLflow UI, run the following command in a new terminal:")
    print("mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000")
    print("Then open: http://127.0.0.1:5000")
    print("="*60)

def main():
    """Main function"""
    print("Weather Prediction MLflow Upload Pipeline")
    print("="*60)
    
    # Setup MLflow
    if not setup_mlflow():
        print("Failed to setup MLflow. Continuing without MLflow logging.")
        return
    
    # Load data
    X, y = load_and_prepare_data()
    feature_names = X.columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    print(f"Target balance: {y.mean():.3f}")
    
    # Train models
    results = train_best_models(X_train, X_test, y_train, y_test)
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['metrics']['roc_auc'])
    best_result = results[best_model_name]
    
    print(f"\nBest Model: {best_model_name}")
    print("Best Model Metrics:")
    for metric, value in best_result['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    # Log all models to MLflow
    run_info = {}
    for model_name, result in results.items():
        try:
            run_id, artifact_uri = log_model_to_mlflow(
                model_name, 
                result['model'], 
                result['metrics'], 
                X_test, 
                feature_names
            )
            run_info[model_name] = {'run_id': run_id, 'artifact_uri': artifact_uri}
        except Exception as e:
            print(f"Error logging {model_name}: {e}")
    
    # Save best model for production
    model_file, metadata_file = save_production_model(
        best_model_name,
        best_result['model'],
        best_result['metrics'],
        feature_names
    )
    
    # Final summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Best Model: {best_model_name}")
    print(f"Best AUC: {best_result['metrics']['roc_auc']:.4f}")
    print(f"Best F1: {best_result['metrics']['f1_score']:.4f}")
    print(f"Production Model: {model_file}")
    
    if run_info:
        print("\nMLflow Runs:")
        for model_name, info in run_info.items():
            print(f"  {model_name}: {info['run_id']}")
    
    # Instructions for MLflow UI
    start_mlflow_ui()
    
    return best_result, run_info

if __name__ == "__main__":
    best_model, runs = main()