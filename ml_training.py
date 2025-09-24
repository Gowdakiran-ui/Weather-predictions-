#!/usr/bin/env python3
"""
Machine Learning Training Script for Weather Prediction
Author: Kiran gowda.A
Purpose: Train and compare ML models for BBQ weather prediction with MLflow integration
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report, roc_curve)

# Advanced ML Libraries
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not installed. Install with: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM not installed. Install with: pip install lightgbm")

# MLflow
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    import mlflow.lightgbm
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    print("MLflow not installed. Install with: pip install mlflow")

import joblib
import os

def setup_mlflow():
    """Setup MLflow tracking"""
    if not HAS_MLFLOW:
        print("MLflow not available. Skipping MLflow setup.")
        return False
    
    # Set tracking URI to local directory
    mlflow_dir = os.path.join(os.getcwd(), "mlruns")
    mlflow.set_tracking_uri(f"file://{mlflow_dir}")
    
    # Set experiment
    experiment_name = "Weather_Prediction_BBQ"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        print(f"Error setting up MLflow experiment: {e}")
        return False
    
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"MLflow experiment: {experiment_name}")
    return True

def load_and_prepare_data():
    """Load and prepare the engineered dataset"""
    print("="*60)
    print("LOADING AND PREPARING DATA")
    print("="*60)
    
    # Load the selected features dataset
    df = pd.read_csv('data/weather_engineered_selected.csv')
    print(f"Loaded dataset shape: {df.shape}")
    
    # Handle date columns and non-numeric features
    date_cols = [col for col in df.columns if 'DATE' in col]
    print(f"Removing date columns: {date_cols}")
    
    # Remove date columns
    df = df.drop(columns=date_cols, errors='ignore')
    
    # Handle categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        print(f"Encoding categorical columns: {categorical_cols}")
        le = LabelEncoder()
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Separate features and target
    target_col = 'TOURS_BBQ_weather'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)  # Convert boolean to int
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:")
    print(y.value_counts())
    print(f"Target balance: {y.mean():.3f} positive rate")
    
    return X, y, df

def get_baseline_models():
    """Get a dictionary of baseline models to compare"""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
        'Extra Trees': ExtraTreesClassifier(random_state=42, n_estimators=100),
        'SVM': SVC(random_state=42, probability=True),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10)
    }
    
    # Add advanced models if available
    if HAS_XGBOOST:
        models['XGBoost'] = XGBClassifier(random_state=42, eval_metric='logloss')
    
    if HAS_LIGHTGBM:
        models['LightGBM'] = LGBMClassifier(random_state=42, verbose=-1)
    
    return models

def evaluate_models(X_train, X_test, y_train, y_test, models):
    """Evaluate multiple models and return results"""
    print("\n" + "="*60)
    print("MODEL EVALUATION AND COMPARISON")
    print("="*60)
    
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
        
        results[name] = {
            'model': model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"{name} - CV AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"{name} - Test AUC: {roc_auc:.4f}, F1: {f1:.4f}")
    
    return results

def select_best_model(results):
    """Select the best model based on ROC-AUC score"""
    print("\n" + "="*60)
    print("MODEL SELECTION")
    print("="*60)
    
    # Create comparison DataFrame
    comparison_data = []
    for name, result in results.items():
        comparison_data.append({
            'Model': name,
            'CV_AUC_Mean': result['cv_mean'],
            'CV_AUC_Std': result['cv_std'],
            'Test_AUC': result['roc_auc'],
            'Test_F1': result['f1'],
            'Test_Accuracy': result['accuracy'],
            'Test_Precision': result['precision'],
            'Test_Recall': result['recall']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Test_AUC', ascending=False)
    
    print("Model Comparison Results:")
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    
    # Select best model
    best_model_name = comparison_df.iloc[0]['Model']
    best_model_result = results[best_model_name]
    
    print(f"\nBest Model: {best_model_name}")
    print(f"Best Test AUC: {best_model_result['roc_auc']:.4f}")
    
    # Save comparison results
    comparison_df.to_csv('data/model_comparison_results.csv', index=False)
    print("Model comparison results saved to: data/model_comparison_results.csv")
    
    return best_model_name, best_model_result

def hyperparameter_tuning(best_model_name, X_train, y_train):
    """Perform hyperparameter tuning for the best model"""
    print("\n" + "="*60)
    print(f"HYPERPARAMETER TUNING - {best_model_name}")
    print("="*60)
    
    # Define parameter grids for different models
    param_grids = {
        'Random Forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'Logistic Regression': {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        } if HAS_XGBOOST else {},
        'LightGBM': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'num_leaves': [31, 50, 100]
        } if HAS_LIGHTGBM else {}
    }
    
    if best_model_name not in param_grids:
        print(f"No hyperparameter grid defined for {best_model_name}. Using default parameters.")
        return get_baseline_models()[best_model_name]
    
    # Get base model
    base_models = get_baseline_models()
    base_model = base_models[best_model_name]
    
    # Perform grid search
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced folds for speed
    grid_search = GridSearchCV(
        base_model, 
        param_grids[best_model_name], 
        cv=cv, 
        scoring='roc_auc', 
        n_jobs=-1, 
        verbose=1
    )
    
    print("Starting hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def create_visualizations(y_test, results, best_model_name):
    """Create model performance visualizations"""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Weather Prediction Model Performance', fontsize=16)
    
    # 1. Model Comparison (AUC scores)
    model_names = list(results.keys())
    auc_scores = [results[name]['roc_auc'] for name in model_names]
    
    axes[0, 0].bar(model_names, auc_scores)
    axes[0, 0].set_title('Model Comparison - ROC AUC Scores')
    axes[0, 0].set_ylabel('ROC AUC')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Confusion Matrix for best model
    best_result = results[best_model_name]
    cm = confusion_matrix(y_test, best_result['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title(f'Confusion Matrix - {best_model_name}')
    axes[0, 1].set_ylabel('True Label')
    axes[0, 1].set_xlabel('Predicted Label')
    
    # 3. ROC Curve for best model
    if best_result['y_pred_proba'] is not None:
        fpr, tpr, _ = roc_curve(y_test, best_result['y_pred_proba'])
        axes[1, 0].plot(fpr, tpr, label=f'{best_model_name} (AUC = {best_result["roc_auc"]:.3f})')
        axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].set_title('ROC Curve')
        axes[1, 0].legend()
    
    # 4. Feature Importance (if available)
    if hasattr(best_result['model'], 'feature_importances_'):
        # Get feature names (assuming X is available in scope)
        # For now, we'll just show top 10 features
        importances = best_result['model'].feature_importances_
        top_10_idx = np.argsort(importances)[-10:]
        top_10_importances = importances[top_10_idx]
        
        axes[1, 1].barh(range(len(top_10_importances)), top_10_importances)
        axes[1, 1].set_title('Top 10 Feature Importances')
        axes[1, 1].set_xlabel('Importance')
        axes[1, 1].set_ylabel('Feature Index')
    
    plt.tight_layout()
    plt.savefig('data/model_performance_visualizations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizations saved to: data/model_performance_visualizations.png")

def log_to_mlflow(model, model_name, metrics, X_test, y_test, feature_names):
    """Log model and metrics to MLflow"""
    if not HAS_MLFLOW:
        print("MLflow not available. Skipping MLflow logging.")
        return None
    
    print("\n" + "="*60)
    print("LOGGING TO MLFLOW")
    print("="*60)
    
    with mlflow.start_run(run_name=f"{model_name}_weather_prediction"):
        # Log parameters
        if hasattr(model, 'get_params'):
            mlflow.log_params(model.get_params())
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model based on type
        if 'XGBoost' in model_name and HAS_XGBOOST:
            mlflow.xgboost.log_model(model, "model")
        elif 'LightGBM' in model_name and HAS_LIGHTGBM:
            mlflow.lightgbm.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")
        
        # Log additional artifacts
        mlflow.log_artifact('data/model_comparison_results.csv')
        mlflow.log_artifact('data/model_performance_visualizations.png')
        
        # Create and log feature importance if available
        if hasattr(model, 'feature_importances_'):
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            feature_importance_df.to_csv('data/feature_importance.csv', index=False)
            mlflow.log_artifact('data/feature_importance.csv')
        
        # Log dataset info
        mlflow.log_param("dataset_shape", f"{X_test.shape[0] + len(y_test)}x{X_test.shape[1]}")
        mlflow.log_param("target_balance", f"{np.mean(y_test):.3f}")
        
        run_id = mlflow.active_run().info.run_id
        print(f"MLflow run ID: {run_id}")
        print(f"Model logged to MLflow successfully!")
        
        return run_id

def save_final_model(model, model_name, metrics, feature_names):
    """Save the final trained model and related artifacts"""
    print("\n" + "="*60)
    print("SAVING FINAL MODEL")
    print("="*60)
    
    # Create model directory
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, f'{model_name.lower().replace(" ", "_")}_weather_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save model metadata
    metadata = {
        'model_name': model_name,
        'model_type': type(model).__name__,
        'training_date': datetime.now().isoformat(),
        'metrics': metrics,
        'feature_names': feature_names,
        'model_parameters': model.get_params() if hasattr(model, 'get_params') else {}
    }
    
    metadata_path = os.path.join(model_dir, f'{model_name.lower().replace(" ", "_")}_metadata.json')
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Model metadata saved to: {metadata_path}")
    
    # Save feature names
    feature_names_path = os.path.join(model_dir, 'feature_names.txt')
    with open(feature_names_path, 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
    print(f"Feature names saved to: {feature_names_path}")
    
    return model_path, metadata_path

def main():
    """Main function to run the complete ML pipeline"""
    print("Starting Weather Prediction ML Training Pipeline...")
    print("="*80)
    
    # Setup MLflow
    mlflow_available = setup_mlflow()
    
    # Load and prepare data
    X, y, df = load_and_prepare_data()
    feature_names = X.columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Get and evaluate models
    models = get_baseline_models()
    results = evaluate_models(X_train, X_test, y_train, y_test, models)
    
    # Select best model
    best_model_name, best_result = select_best_model(results)
    
    # Hyperparameter tuning
    tuned_model = hyperparameter_tuning(best_model_name, X_train, y_train)
    
    # Evaluate tuned model
    print(f"\nEvaluating tuned {best_model_name}...")
    tuned_model.fit(X_train, y_train)
    y_pred_tuned = tuned_model.predict(X_test)
    y_pred_proba_tuned = (tuned_model.predict_proba(X_test)[:, 1] 
                         if hasattr(tuned_model, 'predict_proba') else None)
    
    # Final metrics
    final_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_tuned),
        'precision': precision_score(y_test, y_pred_tuned),
        'recall': recall_score(y_test, y_pred_tuned),
        'f1_score': f1_score(y_test, y_pred_tuned),
        'roc_auc': roc_auc_score(y_test, y_pred_proba_tuned) if y_pred_proba_tuned is not None else 0
    }
    
    print("\nFinal Tuned Model Performance:")
    for metric, value in final_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # Update results with tuned model
    results[f"{best_model_name} (Tuned)"] = {
        'model': tuned_model,
        'roc_auc': final_metrics['roc_auc'],
        'f1': final_metrics['f1_score'],
        'y_pred': y_pred_tuned,
        'y_pred_proba': y_pred_proba_tuned
    }
    
    # Create visualizations
    create_visualizations(y_test, results, f"{best_model_name} (Tuned)")
    
    # Log to MLflow
    run_id = None
    if mlflow_available:
        run_id = log_to_mlflow(tuned_model, best_model_name, final_metrics, X_test, y_test, feature_names)
    
    # Save final model
    model_path, metadata_path = save_final_model(tuned_model, best_model_name, final_metrics, feature_names)
    
    # Final summary
    print("\n" + "="*80)
    print("TRAINING PIPELINE COMPLETED!")
    print("="*80)
    print(f"Best Model: {best_model_name}")
    print(f"Final ROC-AUC: {final_metrics['roc_auc']:.4f}")
    print(f"Final F1-Score: {final_metrics['f1_score']:.4f}")
    print(f"Model saved to: {model_path}")
    if run_id:
        print(f"MLflow run ID: {run_id}")
    print("="*80)
    
    return tuned_model, final_metrics, feature_names

if __name__ == "__main__":
    model, metrics, features = main()