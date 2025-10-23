#!/bin/bash
# Setup Docker for Python tool execution

echo "="
echo "Docker Setup for gpt-oss Python Tool"
echo "="
echo ""

# Check if Docker is installed
if command -v docker &> /dev/null; then
    echo "✓ Docker is installed: $(docker --version)"
else
    echo "✗ Docker is not installed"
    echo ""
    echo "Please install Docker:"
    echo "  Ubuntu/Debian: sudo apt-get install docker.io && sudo systemctl start docker"
    echo "  Fedora: sudo dnf install docker && sudo systemctl start docker"
    echo "  Or visit: https://docs.docker.com/engine/install/"
    echo ""
    exit 1
fi

# Check if Docker daemon is running
if ! docker ps &> /dev/null; then
    echo "✗ Docker daemon is not running or you don't have permission"
    echo ""
    echo "To start Docker:"
    echo "  sudo systemctl start docker"
    echo ""
    echo "To add your user to docker group (recommended):"
    echo "  sudo usermod -aG docker $USER"
    echo "  newgrp docker  # Or log out and back in"
    echo ""
    exit 1
fi

echo "✓ Docker daemon is running"
echo ""

# Pull Python 3.11 image for the Python tool
echo "Pulling Python 3.11 image for code execution tool..."
if docker pull python:3.11; then
    echo "✓ Python 3.11 image downloaded successfully"
else
    echo "✗ Failed to pull Python image"
    exit 1
fi

echo ""
echo "="
echo "Docker setup complete!"
echo "The Python tool in gpt-oss can now execute code in containers."
echo "="
