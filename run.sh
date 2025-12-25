#!/bin/bash
set -e

# Ensure assets directory exists
mkdir -p assets

# Ensure reference.wav exists (placeholder if not)
if [ ! -f "assets/reference.wav" ]; then
  echo "Warning: assets/reference.wav not found. TTS service requires a voice sample."
  echo "Please place a 6-10 second wav file at 'assets/reference.wav'."
fi

# Check if venv exists
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3.10 -m venv venv
  source venv/bin/activate
  echo "Upgrading pip..."
  pip install --upgrade pip
  echo "Installing dependencies..."
  pip install -r requirements.txt
else
  source venv/bin/activate
fi

# Set environment variables for Coqui
export COQUI_TOS_AGREED=1

# Optimization for CPU-only mode: Limit threads to avoid starving Go services
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
# Explicitly hide GPU as a secondary safeguard
export CUDA_VISIBLE_DEVICES=""

echo "Starting Dexter TTS Service on 127.0.0.1:8200..."
python main.py
