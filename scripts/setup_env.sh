#!/bin/bash
set -e
cd "$(dirname "$0")/.."

if [ -d "venv" ]; then
    echo "venv already exists, skipping creation."
else
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Setup complete."