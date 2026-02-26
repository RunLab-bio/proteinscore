#!/bin/bash
# =============================================================================
# Build Lambda Layers
# =============================================================================
# Creates the requests library layer for Lambda functions
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LAYERS_DIR="$PROJECT_DIR/layers"
BUILD_DIR="$PROJECT_DIR/.build"

echo "🔧 Building Lambda layers..."

# Create directories
mkdir -p "$LAYERS_DIR"
mkdir -p "$BUILD_DIR"

# Build requests layer
echo "📦 Building requests layer..."
LAYER_BUILD_DIR="$BUILD_DIR/requests-layer"
rm -rf "$LAYER_BUILD_DIR"
mkdir -p "$LAYER_BUILD_DIR/python"

# Install requests and dependencies
pip install requests -t "$LAYER_BUILD_DIR/python" --quiet --upgrade

# Create zip
cd "$LAYER_BUILD_DIR"
zip -r "$LAYERS_DIR/requests.zip" python -x "*.pyc" -x "*__pycache__*" > /dev/null

echo "✅ Layer created: $LAYERS_DIR/requests.zip"

# Cleanup
rm -rf "$LAYER_BUILD_DIR"

echo "🎉 All layers built successfully!"
