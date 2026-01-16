# Dexter TTS Service

A backend Text-to-Speech service for Dexter, powered by [Coqui TTS](https://github.com/coqui-ai/TTS) (XTTS-v2).

## Overview

This service exposes a FastAPI interface to generate high-quality audio from text using zero-shot voice cloning. It allows Dexter to "speak" with a consistent voice identity.

## Requirements

- **OS**: Linux (Ubuntu/Debian recommended)
- **Python**: 3.10 or higher
- **GPU**: NVIDIA GPU with CUDA 11.8+ (Strongly recommended for XTTS-v2)
- **RAM**: 8GB+ (16GB+ recommended)

## Installation

1.  **Install System Dependencies**:

    ```bash
    sudo apt-get install libsndfile1
    ```

2.  **Create Virtual Environment**:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Python Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

    _Note: This installs `TTS` which includes PyTorch. If you need a specific CUDA version of PyTorch, install it manually before `TTS`._

4.  **Accept Coqui License**:
    XTTS-v2 requires accepting the user agreement. The service will attempt to handle this, or you can run `coqui-tts` CLI once to accept it interactively.

## Usage

### Start the Service

```bash
./run.sh
```

The service runs on `http://127.0.0.1:8200`.

### API Endpoints

#### `GET /health`

Checks if the service and model are loaded.

#### `POST /generate`

Generates audio from text.

**Request:**

```json
{
  "text": "Hello, I am Dexter.",
  "speaker_wav": "/path/to/reference_voice.wav",
  "language": "en"
}
```

**Response:**
Returns a stream of audio data (`audio/wav`).

## Configuration

The service looks for a default reference audio file at `assets/reference.wav`. Place a clean, 6-10 second audio clip of the target voice there.
