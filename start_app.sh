#!/bin/bash

# Stop on first error
set -e

# 1. Activate virtual environment (if you have one)
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# 2. Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# 3. Start the Python app
echo "Starting the app..."
# Replace main.py with your entry point file
python3 main.py
