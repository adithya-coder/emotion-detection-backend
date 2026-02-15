#!/bin/bash

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🐳 Emotion Detection Docker Setup${NC}\n"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker is installed${NC}"

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Docker is not running. Please start Docker.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker is running${NC}\n"

# Build the image
echo -e "${BLUE}🔨 Building Docker image...${NC}"
docker build -t emotion-detection-backend:latest .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Image built successfully${NC}\n"
else
    echo -e "${RED}❌ Failed to build image${NC}"
    exit 1
fi

# Stop and remove existing container
echo -e "${BLUE}🧹 Cleaning up existing containers...${NC}"
docker stop emotion-detection-app 2>/dev/null
docker rm emotion-detection-app 2>/dev/null

# Run the container
echo -e "${BLUE}🚀 Starting container...${NC}"
docker run -d \
    --name emotion-detection-app \
    -p 5000:5000 \
    -v "$(pwd)/models:/app/models" \
    --restart unless-stopped \
    emotion-detection-backend:latest

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Container started successfully${NC}\n"
    
    # Wait for service to be ready
    echo -e "${BLUE}⏳ Waiting for service to be ready...${NC}"
    sleep 5
    
    # Test the health endpoint
    if curl -s http://localhost:5000/ > /dev/null; then
        echo -e "${GREEN}✓ Service is healthy${NC}\n"
        echo -e "${BLUE}📊 Container Status:${NC}"
        docker ps | grep emotion-detection-app
        echo ""
        echo -e "${GREEN}✅ Setup Complete!${NC}"
        echo -e "${BLUE}🌐 Access the API at: http://localhost:5000${NC}"
        echo -e "${BLUE}📝 View logs: docker logs -f emotion-detection-app${NC}"
    else
        echo -e "${RED}⚠️  Service started but health check failed${NC}"
        echo -e "${BLUE}Check logs: docker logs emotion-detection-app${NC}"
    fi
else
    echo -e "${RED}❌ Failed to start container${NC}"
    exit 1
fi
