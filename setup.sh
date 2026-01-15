#!/bin/bash

# setup.sh: Prepare the RecurseZero environment

echo "Setting up RecurseZero..."

# Check for python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found."
    exit 1
fi

# Create venv if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install Dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Post-install messages
echo "Success! Environment is ready."
echo "To run training: source venv/bin/activate && python main.py"
