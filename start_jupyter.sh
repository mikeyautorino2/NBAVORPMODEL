#!/bin/bash
# NBA VORP Model - Jupyter Notebook Startup Script

echo "🏀 NBA VORP Model Environment Setup"
echo "=================================="

# Navigate to project directory
cd "$(dirname "$0")"

# Activate virtual environment
echo "Activating virtual environment..."
source vorp-env/bin/activate

# Check if activation was successful
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment activated: $(basename $VIRTUAL_ENV)"
else
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

# Display installed packages info
echo "📦 Key packages installed:"
python -c "
import pandas as pd, numpy as np, sklearn, matplotlib, seaborn as sns
print(f'  - pandas: {pd.__version__}')
print(f'  - numpy: {np.__version__}') 
print(f'  - scikit-learn: {sklearn.__version__}')
print(f'  - matplotlib: {matplotlib.__version__}')
print(f'  - seaborn: {sns.__version__}')
"

echo ""
echo "🚀 Starting Jupyter Notebook..."
echo "The notebook will open in your browser"
echo "Navigate to: analysis/playeranalylsis.ipynb"
echo ""
echo "To stop Jupyter: Press Ctrl+C twice"
echo "To deactivate environment: Run 'deactivate'"
echo ""

# Start Jupyter Notebook
jupyter notebook --ip=localhost --port=8888

echo ""
echo "👋 Jupyter Notebook stopped"
echo "Virtual environment is still active. Run 'deactivate' to exit."