#!/bin/bash
# Runs the ENTIRE study (data prep -> preprocessing -> all training ->
# all evaluation -> comparisons) in a single command.
set -e
cd "$(dirname "$0")/.."
source venv/bin/activate
python main.py