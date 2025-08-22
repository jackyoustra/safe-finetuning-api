#!/bin/bash
set -e  # Exit on error

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv package manager not found! You need to install it first."
    echo ""
    echo "Installation options:"
    echo "  • macOS/Linux:    curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  • Windows:        powershell -ExecutionPolicy ByPass -c \"irm https://astral.sh/uv/install.ps1 | iex\""
    echo "  • Using pip:      pip install uv"
    echo "  • Using pipx:     pipx install uv"
    echo ""
    echo "After installing, run this script again."
    exit 1
fi

echo "🚀 Initializing ft-robustness environment with flash-attn support..."

# Step 1: Create the environment without installing flash-attn
# This ensures torch is installed first, which flash-attn needs for compilation
echo "📦 Creating environment (skipping flash-attn for now)..."
uv sync --no-install-package flash-attn

# Step 2: Now install everything including flash-attn
# flash-attn will use the already installed torch for compilation
echo "🔧 Installing all dependencies including flash-attn..."
uv sync --no-build-isolation

# Step 3: Install the local package in development mode
echo "🔧 Installing ft-robustness package in development mode..."
uv pip install -e .

echo "✅ Environment initialization complete!"
echo "You can now use regular 'uv sync' commands for future updates" 