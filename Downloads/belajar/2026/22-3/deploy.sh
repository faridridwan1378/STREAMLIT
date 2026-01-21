#!/bin/bash

# Deployment script for analisa_saham_detail.py

echo "Starting deployment of analisa_saham_detail.py..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install Python3 first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Make script executable
chmod +x analisa_saham_detail.py

echo "Deployment completed successfully!"
echo ""
echo "To run the script:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the script with a ticker: python analisa_saham_detail.py BBRI.JK"
echo "   (Replace BBRI.JK with your desired stock ticker)"
echo ""
echo "For automated runs, you can use cron:"
echo "Example: Run daily at 9 AM: 0 9 * * * /path/to/project/venv/bin/python /path/to/project/analisa_saham_detail.py BBRI.JK"
