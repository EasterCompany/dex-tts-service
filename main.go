package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"syscall"
)

// requirementsTxt content (Minimal for FastAPI wrapper)
const requirementsTxt = `fastapi>=0.109.0
uvicorn[standard]>=0.27.0
pydantic>=2.6.0
python-multipart
requests
redis
psutil
`

// mainPy content
const mainPy = `import os
import sys
import logging
import time
import psutil
import subprocess
import requests
import tarfile
import io
import json
import redis
import hashlib
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel

# Force standard streams to be unbuffered
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logger = logging.getLogger("dex-tts-service")

START_TIME = time.time()

# Constants
PIPER_URL = "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz"
VOICE_MODEL_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts/high/en_US-libritts-high.onnx"
VOICE_CONFIG_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts/high/en_US-libritts-high.onnx.json"

BIN_DIR = os.path.expanduser("~/Dexter/bin")
MODELS_DIR = os.path.expanduser("~/Dexter/models/piper")
PIPER_BIN = os.path.join(BIN_DIR, "piper", "piper")
VOICE_MODEL_PATH = os.path.join(MODELS_DIR, "en_US-libritts-high.onnx")
VOICE_CONFIG_PATH = os.path.join(MODELS_DIR, "en_US-libritts-high.onnx.json")

# Redis Client
redis_client = None
try:
    redis_client = redis.Redis(host='127.0.0.1', port=6379, db=0)
    redis_client.ping()
    logger.info("Connected to Redis successfully.")
except Exception as e:
    logger.warning(f"Failed to connect to Redis: {e}")
    redis_client = None

def download_file(url, path):
    if os.path.exists(path):
        return
    logger.info(f"Downloading {url} to {path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    logger.info(f"Downloaded {path}")

def setup_piper():
    # 1. Setup Binary
    if not os.path.exists(PIPER_BIN):
        logger.info("Piper binary not found. Downloading...")
        os.makedirs(BIN_DIR, exist_ok=True)
        tar_path = os.path.join(BIN_DIR, "piper.tar.gz")
        download_file(PIPER_URL, tar_path)
        
        logger.info("Extracting Piper...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=BIN_DIR)
        os.remove(tar_path)
        logger.info("Piper installed.")

    # 2. Setup Model
    os.makedirs(MODELS_DIR, exist_ok=True)
    try:
        download_file(VOICE_MODEL_URL, VOICE_MODEL_PATH)
        download_file(VOICE_CONFIG_URL, VOICE_CONFIG_PATH)
    except Exception as e:
        logger.error(f"Failed to download voice model: {e}")
        # Allow startup without model, but generation will fail

app = FastAPI(title="Dexter Piper TTS", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    setup_piper()

class GenerateRequest(BaseModel):
    text: str
    language: str = "en"
    output_path: str = None

@app.get("/health")
async def health_check():
    if os.path.exists(PIPER_BIN) and os.path.exists(VOICE_MODEL_PATH):
        return {"status": "ok", "engine": "piper"}
    return JSONResponse(status_code=503, content={"status": "error", "detail": "Piper or Model missing"})

@app.get("/service")
async def service_status():
    process = psutil.Process(os.getpid())
    uptime_seconds = time.time() - START_TIME
    
    # Try Environment Variables
    version_str = os.getenv("DEX_VERSION", "0.0.0")
    
    return {
        "version": {"str": version_str, "obj": {}},
        "health": {"status": "ok", "uptime": f"{uptime_seconds:.2f}s"},
        "metrics": {
            "cpu": { "avg": process.cpu_percent(interval=None) },
            "memory": { "avg": process.memory_info().rss / 1024 / 1024 }
        }
    }

@app.post("/generate")
async def generate_audio(request: GenerateRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text required")

    try:
        # Check Cache
        text_hash = hashlib.md5(request.text.encode()).hexdigest()
        key = f"tts:piper:{text_hash}"
        
        if redis_client:
            cached = redis_client.get(key)
            if cached:
                # logger.info("Cache hit")
                return Response(content=cached, media_type="audio/wav")

        # Run Piper
        cmd = [
            PIPER_BIN,
            "--model", VOICE_MODEL_PATH,
            "--output_file", "-"
        ]

        start = time.time()
        process = subprocess.Popen(
            cmd, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        wav_data, stderr = process.communicate(input=request.text.encode('utf-8'))
        
        if process.returncode != 0:
            logger.error(f"Piper error: {stderr.decode()}")
            raise Exception("Piper generation failed")

        duration = time.time() - start
        logger.info(f"Generated audio in {duration:.3f}s")

        if redis_client:
            redis_client.set(key, wav_data)

        if request.output_path:
            with open(request.output_path, "wb") as f:
                f.write(wav_data)
            return {"status": "ok", "file_path": request.output_path}

        return Response(content=wav_data, media_type="audio/wav")

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8200"))
    uvicorn.run(app, host=host, port=port)
`

var (
	version   = "0.0.0"
	branch    = "unknown"
	commit    = "unknown"
	buildDate = "unknown"
	buildYear = "unknown"
	buildHash = "unknown"
	arch      = "unknown"
)

func main() {
	if len(os.Args) > 1 && os.Args[1] == "version" {
		v := version
		if v == "0.0.0" || v == "" {
			v = os.Getenv("DEX_VERSION")
		}
		if v == "" {
			v = "0.0.0"
		}
		fmt.Printf("%s.%s.%s.%s.%s.%s.%s\n", v, branch, commit, buildDate, buildYear, buildHash, arch)
		return
	}

	homeDir, err := os.UserHomeDir()
	if err != nil {
		log.Fatalf("Failed to get user home directory: %v", err)
	}

	serviceDir := filepath.Join(homeDir, "Dexter", "services", "dex-tts-service")
	if err := os.MkdirAll(serviceDir, 0755); err != nil {
		log.Fatalf("Failed to create service directory: %v", err)
	}

	if err := os.WriteFile(filepath.Join(serviceDir, "requirements.txt"), []byte(requirementsTxt), 0644); err != nil {
		log.Fatalf("Failed to write requirements.txt: %v", err)
	}
	if err := os.WriteFile(filepath.Join(serviceDir, "main.py"), []byte(mainPy), 0644); err != nil {
		log.Fatalf("Failed to write main.py: %v", err)
	}

	// Load options.json (Not needed for Piper as it's CPU optimized, but good for completeness)

	pythonEnvDir := filepath.Join(homeDir, "Dexter", "python3.10")
	pythonBin := filepath.Join(pythonEnvDir, "bin", "python")
	pipBin := filepath.Join(pythonEnvDir, "bin", "pip")

	if _, err := os.Stat(pythonBin); os.IsNotExist(err) {
		log.Fatalf("Shared Python 3.10 environment not found at %s.", pythonBin)
	}

	log.Println("Ensuring pip is up-to-date...")
	pipUpdateCmd := exec.Command(pipBin, "install", "--upgrade", "pip")
	_ = pipUpdateCmd.Run()

	log.Println("Installing dependencies into shared environment...")
	pipCmd := exec.Command(pipBin, "install", "-r", "requirements.txt")
	pipCmd.Dir = serviceDir
	pipCmd.Stdout = os.Stdout
	pipCmd.Stderr = os.Stderr
	if err := pipCmd.Run(); err != nil {
		log.Printf("Warning: Failed to install dependencies: %v", err)
	}

	log.Println("Starting Dexter TTS Service (Piper)...")

	pythonCmd := exec.Command(pythonBin, "main.py")
	pythonCmd.Dir = serviceDir

	v := version
	if v == "0.0.0" || v == "" {
		v = os.Getenv("DEX_VERSION")
	}
	// Environment Variables injection
	pythonCmd.Env = append(os.Environ(),
		fmt.Sprintf("DEX_VERSION=%s", v),
		// ... (truncated for brevity, assuming standard env pass through)
	)

	pythonCmd.Stdout = os.Stdout
	pythonCmd.Stderr = os.Stderr

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigChan
		_ = pythonCmd.Process.Signal(os.Interrupt)
	}()

	if err := pythonCmd.Run(); err != nil {
		log.Printf("Service exited with error: %v", err)
		os.Exit(1)
	}
}
