#!/bin/bash

# OpenManus Docker Deployment Script
# This script helps deploy OpenManus in different environments

set -e

# Default values
ENVIRONMENT="production"
COMPOSE_FILES="docker-compose.yml"
BUILD_ARGS=""
PULL_IMAGES=false
BACKUP_BEFORE_DEPLOY=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy OpenManus using Docker Compose

OPTIONS:
    -e, --environment ENV    Environment to deploy (development, production) [default: production]
    -p, --pull              Pull latest images before deployment
    -b, --backup            Create backup before deployment (production only)
    -h, --help              Show this help message

EXAMPLES:
    $0                                    # Deploy production environment
    $0 -e development                     # Deploy development environment
    $0 -e production -p -b               # Deploy production with pull and backup
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -p|--pull)
            PULL_IMAGES=true
            shift
            ;;
        -b|--backup)
            BACKUP_BEFORE_DEPLOY=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate environment
case $ENVIRONMENT in
    development|dev)
        ENVIRONMENT="development"
        COMPOSE_FILES="docker-compose.yml -f docker-compose.dev.yml"
        ;;
    production|prod)
        ENVIRONMENT="production"
        COMPOSE_FILES="docker-compose.yml -f docker-compose.prod.yml"
        ;;
    *)
        print_error "Invalid environment: $ENVIRONMENT"
        print_error "Supported environments: development, production"
        exit 1
        ;;
esac

print_status "Deploying OpenManus in $ENVIRONMENT environment"

# Check if Docker and Docker Compose are available
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed or not in PATH"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed or not in PATH"
    exit 1
fi

# Check if .env file exists
if [[ ! -f .env ]]; then
    print_warning ".env file not found"
    if [[ -f .env.example ]]; then
        print_status "Copying .env.example to .env"
        cp .env.example .env
        print_warning "Please edit .env file with your configuration before continuing"
        exit 1
    else
        print_error "No .env.example file found. Please create .env file manually"
        exit 1
    fi
fi

# Set build arguments
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
VERSION=$(git describe --tags --always 2>/dev/null || echo "latest")

export BUILD_DATE
export VCS_REF
export VERSION

print_status "Build info: Version=$VERSION, Ref=$VCS_REF, Date=$BUILD_DATE"

# Create backup if requested (production only)
if [[ $BACKUP_BEFORE_DEPLOY == true && $ENVIRONMENT == "production" ]]; then
    print_status "Creating backup before deployment..."

    # Check if backup service is available
    if docker-compose -f $COMPOSE_FILES ps backup &> /dev/null; then
        docker-compose -f $COMPOSE_FILES exec backup /backup.sh
        print_success "Backup completed"
    else
        print_warning "Backup service not available, skipping backup"
    fi
fi

# Pull images if requested
if [[ $PULL_IMAGES == true ]]; then
    print_status "Pulling latest images..."
    docker-compose -f $COMPOSE_FILES pull
    print_success "Images pulled successfully"
fi

# Build and start services
print_status "Building and starting services..."

# Stop existing services
print_status "Stopping existing services..."
docker-compose -f $COMPOSE_FILES down

# Build images
print_status "Building OpenManus image..."
docker-compose -f $COMPOSE_FILES build openmanus

# Start services
print_status "Starting services..."
docker-compose -f $COMPOSE_FILES up -d

# Wait for services to be healthy
print_status "Waiting for services to be healthy..."
sleep 10

# Check health
print_status "Checking service health..."
HEALTH_CHECK_URL="http://localhost:${HEALTH_PORT:-8080}/health/live"

for i in {1..30}; do
    if curl -f -s $HEALTH_CHECK_URL > /dev/null 2>&1; then
        print_success "OpenManus is healthy and ready!"
        break
    else
        if [[ $i -eq 30 ]]; then
            print_error "Health check failed after 30 attempts"
            print_error "Check logs with: docker-compose -f $COMPOSE_FILES logs openmanus"
            exit 1
        fi
        print_status "Waiting for health check... (attempt $i/30)"
        sleep 2
    fi
done

# Show running services
print_status "Running services:"
docker-compose -f $COMPOSE_FILES ps

# Show useful commands
print_success "Deployment completed successfully!"
echo
print_status "Useful commands:"
echo "  View logs:           docker-compose -f $COMPOSE_FILES logs -f"
echo "  Check status:        docker-compose -f $COMPOSE_FILES ps"
echo "  Stop services:       docker-compose -f $COMPOSE_FILES down"
echo "  Restart service:     docker-compose -f $COMPOSE_FILES restart openmanus"
echo "  Health check:        curl http://localhost:${HEALTH_PORT:-8080}/health"
echo "  Application:         http://localhost:${OPENMANUS_PORT:-8000}"

if [[ $ENVIRONMENT == "development" ]]; then
    echo "  Shell access:        docker-compose -f $COMPOSE_FILES exec openmanus bash"
    echo "  Run tests:           docker-compose -f $COMPOSE_FILES exec openmanus pytest"
fi
