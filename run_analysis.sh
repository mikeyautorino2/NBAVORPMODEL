#!/bin/bash
# NBA VORP Model - Terminal Analysis Runner

echo "🏀 NBA VORP Model - Terminal Analysis"
echo "====================================="

# Navigate to project directory
cd "$(dirname "$0")"

# Activate virtual environment
echo "Activating virtual environment..."
source vorp-env/bin/activate

if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment activated"
else
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

# Run the analysis
echo ""
echo "🚀 Running VORP Analysis..."
echo ""

python run_vorp_model.py

echo ""
echo "👋 Analysis complete!"
echo "To view detailed analysis, run: ./start_jupyter.sh"