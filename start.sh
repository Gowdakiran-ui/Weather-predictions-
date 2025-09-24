#!/bin/bash

# Startup script for Weather Prediction App

echo "Starting Weather Prediction ML Application..."
echo "============================================="

# Set production environment
export FLASK_ENV=production
export PYTHONUNBUFFERED=1

# Navigate to deployment directory
cd /app/deployment

# Start the application with gunicorn
if [ "$FLASK_ENV" = "production" ]; then
    echo "Starting with Gunicorn (Production Mode)..."
    exec gunicorn --bind 0.0.0.0:5001 \
                  --workers 4 \
                  --timeout 120 \
                  --keepalive 5 \
                  --max-requests 1000 \
                  --access-logfile - \
                  --error-logfile - \
                  --log-level info \
                  app:app
else
    echo "Starting with Flask dev server..."
    exec python app.py
fi