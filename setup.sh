#!/bin/bash

# Make script exit on first error
set -e

# Print commands before executing
set -x

# Create necessary directories
mkdir -p templates
mkdir -p static/css

# Create a basic CSS file
cat > static/css/styles.css << EOL
/* Add your custom styles here */
.chart-container {
    height: 400px;
    width: 100%;
}
EOL

# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables
echo "PYTHONPATH=$PYTHONPATH:$(pwd)" >> ~/.bashrc

# Create empty __init__.py files
touch templates/__init__.py

# Set file permissions
chmod +x setup.sh

echo "Setup complete! The trading bot is ready to use."
echo "Run 'python main.py' to start the application"
echo "Then visit http://localhost:8000 in your browser"