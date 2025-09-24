#!/bin/bash

# Weather Prediction App Deployment Script
# This script helps deploy the application using Docker or Kubernetes

set -e

echo "🌤️ Weather Prediction ML Application Deployment"
echo "================================================"

# Function to build Docker image
build_docker() {
    echo "🐳 Building Docker image..."
    docker build -t weather-prediction:latest .
    echo "✅ Docker image built successfully!"
}

# Function to run with Docker
run_docker() {
    echo "🚀 Running with Docker..."
    docker run -d \
        --name weather-prediction-app \
        -p 5001:5001 \
        --restart unless-stopped \
        weather-prediction:latest
    echo "✅ Application started at http://localhost:5001"
}

# Function to deploy to Kubernetes
deploy_k8s() {
    echo "☸️ Deploying to Kubernetes..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        echo "❌ kubectl not found. Please install kubectl first."
        exit 1
    fi
    
    # Apply all Kubernetes manifests
    kubectl apply -f k8s/
    
    echo "✅ Deployed to Kubernetes!"
    echo "📋 Check status with: kubectl get pods -l app=weather-prediction"
    echo "🌐 Port forward with: kubectl port-forward service/weather-prediction-service 8080:80"
}

# Function to stop Docker container
stop_docker() {
    echo "🛑 Stopping Docker container..."
    docker stop weather-prediction-app || true
    docker rm weather-prediction-app || true
    echo "✅ Docker container stopped!"
}

# Function to remove Kubernetes deployment
remove_k8s() {
    echo "🗑️ Removing Kubernetes deployment..."
    kubectl delete -f k8s/ || true
    echo "✅ Kubernetes deployment removed!"
}

# Function to show logs
show_logs() {
    if [ "$1" = "docker" ]; then
        echo "📋 Docker logs:"
        docker logs -f weather-prediction-app
    elif [ "$1" = "k8s" ]; then
        echo "📋 Kubernetes logs:"
        kubectl logs -f deployment/weather-prediction-app
    fi
}

# Main menu
case "$1" in
    "build")
        build_docker
        ;;
    "run")
        build_docker
        run_docker
        ;;
    "k8s")
        deploy_k8s
        ;;
    "stop")
        stop_docker
        ;;
    "remove-k8s")
        remove_k8s
        ;;
    "logs")
        show_logs "$2"
        ;;
    "help"|*)
        echo "Usage: $0 {build|run|k8s|stop|remove-k8s|logs}"
        echo ""
        echo "Commands:"
        echo "  build        Build Docker image"
        echo "  run          Build and run with Docker"
        echo "  k8s          Deploy to Kubernetes"
        echo "  stop         Stop Docker container"
        echo "  remove-k8s   Remove Kubernetes deployment"
        echo "  logs docker  Show Docker logs"
        echo "  logs k8s     Show Kubernetes logs"
        echo ""
        echo "Examples:"
        echo "  $0 run              # Run with Docker"
        echo "  $0 k8s              # Deploy to Kubernetes"
        echo "  $0 logs docker      # Show Docker logs"
        exit 1
        ;;
esac