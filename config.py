"""
Production Configuration for Weather Prediction App
"""
import os

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'weather-prediction-secret-key'
    
    # Model paths
    MODEL_PATH = os.environ.get('MODEL_PATH') or '../models/xgboost_production_model.pkl'
    SCALER_PATH = os.environ.get('SCALER_PATH') or '../data/standard_scaler.pkl'
    METADATA_PATH = os.environ.get('METADATA_PATH') or '../models/xgboost_metadata.json'
    CLEANED_DATA_PATH = os.environ.get('CLEANED_DATA_PATH') or '../data/weather_cleaned.csv'
    
    # Application settings
    DEBUG = False
    TESTING = False
    
    # Server settings
    HOST = os.environ.get('HOST') or '0.0.0.0'
    PORT = int(os.environ.get('PORT') or 5001)
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL') or 'INFO'
    
    # CORS settings
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    LOG_LEVEL = 'WARNING'

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}