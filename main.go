package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"syscall"
)

// requirementsTxt content
const requirementsTxt = `fastapi>=0.109.0
uvicorn[standard]>=0.27.0
pydantic>=2.6.0
python-multipart
transformers==4.41.2
git+https://github.com/coqui-ai/TTS.git
torch>=2.1.0 --extra-index-url https://download.pytorch.org/whl/cu118
torchcodec>=0.0.2
torchaudio>=2.1.0 --extra-index-url https://download.pytorch.org/whl/cu118
soundfile
requests
redis
`

// mainPy content (Generated)
const mainPy = `import os
import io
import logging
import sys
import time
import psutil
import subprocess
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
import torch
import contextlib
import redis
import hashlib
import numpy as np
import scipy.io.wavfile
import re

# Force standard streams to be unbuffered
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Setup logging to output to stdout/stderr so systemd captures it
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
# Mute uvicorn access logs
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logger = logging.getLogger("dex-tts-service")

START_TIME = time.time()

# Fix for PyTorch 2.6+ weights_only=True security change
@contextlib.contextmanager
def unsafe_torch_load():
    original_load = torch.load
    def unsafe_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    
    torch.load = unsafe_load
    try:
        yield
    finally:
        torch.load = original_load

# Global model variable
tts_model = None

# Redis Client
redis_client = None
try:
    redis_client = redis.Redis(host='127.0.0.1', port=6379, db=0)
    # Test connection
    redis_client.ping()
    logger.info("Connected to Redis successfully.")
except Exception as e:
    logger.warning(f"Failed to connect to Redis: {e}")
    redis_client = None

# Device Configuration
DEVICE = os.getenv("DEX_TTS_DEVICE", "cpu")
logger.info(f"TTS Service configured to use device: {DEVICE}")

# Constants
DEFAULT_SPEAKER_PATH = os.path.join(os.path.dirname(__file__), "assets", "reference.wav")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model():
    global tts_model
    logger.info(f"Starting TTS Service on device: {DEVICE}")
    try:
        # Set custom model path
        custom_model_path = os.path.expanduser("~/Dexter/models/xtts")
        os.makedirs(custom_model_path, exist_ok=True)
        os.environ["TTS_HOME"] = custom_model_path
        
        from TTS.api import TTS
        
        # Check if model files are already present in the custom directory (flat structure)
        config_path = os.path.join(custom_model_path, "config.json")
        if os.path.exists(config_path):
            logger.info(f"Found local model configuration at {config_path}. Loading directly...")
            with unsafe_torch_load():
                # Search for a .pth file in the directory to be safe.
                model_file = None
                for f in os.listdir(custom_model_path):
                    if f.endswith(".pth") and "model" in f: # Heuristic for model.pth
                        model_file = os.path.join(custom_model_path, f)
                        break
                
                if model_file:
                     tts_model = TTS(model_path=custom_model_path, config_path=config_path).to(DEVICE)
                else:
                     logger.warning("config.json found but no obvious .pth model file. Attempting load with directory...")
                     tts_model = TTS(model_path=custom_model_path, config_path=config_path).to(DEVICE)

        else:
            logger.info(f"Local model not found. Downloading/Loading via manager (TTS_HOME={custom_model_path})...")
            with unsafe_torch_load():
                tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)
            
        logger.info("XTTS-v2 model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load TTS model: {e}")
        tts_model = None

# Define lifespan event handler
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model synchronously during startup
    load_model()
    yield
    logger.info("TTS Service shutdown complete.")

app = FastAPI(title="Dexter TTS Service", version="1.0.0", lifespan=lifespan)

class GenerateRequest(BaseModel):
    text: str
    language: str = "en"
    speaker_wav: str = DEFAULT_SPEAKER_PATH

@app.get("/health")
async def health_check():
    if tts_model is None:
        return JSONResponse(
            status_code=503, 
            content={"status": "error", "detail": "Model not loaded"}
        )
    return {"status": "ok", "device": DEVICE, "model": "xtts_v2"}

@app.get("/service")
async def service_status():
    process = psutil.Process(os.getpid())
    uptime_seconds = time.time() - START_TIME
    
    m, s = divmod(uptime_seconds, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    
    parts = []
    if d > 0: parts.append(f"{int(d)}d")
    if h > 0: parts.append(f"{int(h)}h")
    if m > 0: parts.append(f"{int(m)}m")
    parts.append(f"{float(s):.3f}s")
    uptime_str = "".join(parts)

    # 1. Try Environment Variables (Injected by Go Wrapper)
    branch = os.getenv("DEX_BRANCH", "unknown")
    commit = os.getenv("DEX_COMMIT", "unknown")
    version_str = os.getenv("DEX_VERSION", "")
    build_date = os.getenv("DEX_BUILD_DATE", "unknown")

    # 2. Fallbacks
    if branch == "unknown":
        try:
            branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        except: pass

    if commit == "unknown":
        try:
            commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        except: pass

    if not version_str or version_str == "0.0.0":
        version_str = "0.0.1" # Default fallback
        if os.path.exists("version.txt"):
            try:
                with open("version.txt", "r") as f:
                    version_str = f.read().strip()
            except: pass
        else:
            try:
                # Try to get from git tags
                v_tag = subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"], stderr=subprocess.DEVNULL).decode().strip()
                if v_tag.startswith("v"):
                    version_str = v_tag[1:]
            except: pass

    version_parts = version_str.split('.')
    major = version_parts[0] if len(version_parts) > 0 else "0"
    minor = version_parts[1] if len(version_parts) > 1 else "0"
    patch = version_parts[2] if len(version_parts) > 2 else "0"

    import platform
    arch = platform.machine()
    arch = os.getenv("DEX_ARCH", arch)
    build_hash = os.getenv("DEX_BUILD_HASH", "unknown")

    full_version_str = f"{version_str}.{branch}.{commit}.{build_date}.{arch}"

    return {
        "version": {
            "str": full_version_str,
            "obj": {
                "major": major,
                "minor": minor,
                "patch": patch,
                "branch": branch,
                "commit": commit,
                "build_date": build_date,
                "arch": arch,
                "build_hash": build_hash
            }
        },
        "health": {
            "status": "ok" if tts_model is not None else "error",
            "uptime": uptime_str,
            "message": "TTS Model Loaded" if tts_model is not None else "TTS Model Not Loaded"
        },
        "metrics": {
            "cpu": { "avg": process.cpu_percent(interval=None) },
            "memory": { "avg": process.memory_info().rss / 1024 / 1024 }
        }
    }

@app.post("/generate")
async def generate_audio(request: GenerateRequest):
    global tts_model
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model is not available.")

    if not request.text:
        raise HTTPException(status_code=400, detail="Text is required.")

    speaker_wav = request.speaker_wav
    if not os.path.exists(speaker_wav):
        if speaker_wav == DEFAULT_SPEAKER_PATH:
             raise HTTPException(status_code=500, detail=f"Default reference voice not found at {DEFAULT_SPEAKER_PATH}. Please add a reference.wav file.")
        raise HTTPException(status_code=400, detail=f"Speaker wav file not found: {speaker_wav}")

    try:
        logger.info(f"Generating audio for: '{request.text[:30]}...' ")
        
        # 2. Split text into sentences for granular caching
        # Simple regex split on punctuation
        sentences = re.split(r'(?<=[.!?]) +', request.text)
        
        final_wav_parts = []
        speaker_hash = hashlib.md5(request.speaker_wav.encode()).hexdigest()
        
        for sent in sentences:
            sent = sent.strip()
            if not sent: continue
            
            # Cache key structure: tts:v1:speaker_hash:lang:md5(text)
            text_hash = hashlib.md5(sent.encode()).hexdigest()
            key = f"tts:v1:{speaker_hash}:{request.language}:{text_hash}"
            
            audio_chunk = None
            if redis_client:
                try:
                    cached = redis_client.get(key)
                    if cached:
                        # logger.info(f"Cache hit for: {sent[:20]}")
                        audio_chunk = np.frombuffer(cached, dtype=np.float32)
                except Exception as e:
                    logger.warning(f"Redis get failed: {e}")
            
            if audio_chunk is None:
                # logger.info(f"Cache miss for: {sent[:20]}")
                # Generate audio (returns list of floats)
                # Note: We pass speaker_wav each time. Latent extraction optimization 
                # was causing issues with the high-level API wrapper.
                out = tts_model.tts(
                    text=sent,
                    language=request.language,
                    speaker_wav=request.speaker_wav
                )
                audio_chunk = np.array(out, dtype=np.float32)
                
                # Cache the result
                if redis_client:
                    try:
                        redis_client.set(key, audio_chunk.tobytes())
                    except Exception as e:
                        logger.warning(f"Redis set failed: {e}")
            
            final_wav_parts.append(audio_chunk)
            
        if not final_wav_parts:
             raise HTTPException(status_code=400, detail="No audio generated from text.")

        full_audio = np.concatenate(final_wav_parts)

        # Convert to WAV in memory
        buffer = io.BytesIO()
        # XTTS v2 is 24000Hz
        scipy.io.wavfile.write(buffer, 24000, full_audio)

        return Response(content=buffer.getvalue(), media_type="audio/wav")

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8200"))
    
    # Explicitly configure logging to silence access logs
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["loggers"]["uvicorn.access"]["level"] = "WARNING"
    log_config["loggers"]["uvicorn.error"]["level"] = "INFO"
    
    uvicorn.run(app, host=host, port=port, log_config=log_config)
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
		// Priority: ldflags > environment variable > default
		v := version
		if v == "0.0.0" || v == "" {
			v = os.Getenv("DEX_VERSION")
		}
		if v == "" {
			v = "0.0.0"
		}
		fmt.Println(v)
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

	// Write files
	if err := os.WriteFile(filepath.Join(serviceDir, "requirements.txt"), []byte(requirementsTxt), 0644); err != nil {
		log.Fatalf("Failed to write requirements.txt: %v", err)
	}
	if err := os.WriteFile(filepath.Join(serviceDir, "main.py"), []byte(mainPy), 0644); err != nil {
		log.Fatalf("Failed to write main.py: %v", err)
	}

	// Create assets dir
	_ = os.MkdirAll(filepath.Join(serviceDir, "assets"), 0755)

	// Load options.json to get configuration
	device := "cpu" // Default
	optionsPath := filepath.Join(homeDir, "Dexter", "config", "options.json")
	if data, err := os.ReadFile(optionsPath); err == nil {
		var opts struct {
			Services map[string]map[string]interface{} `json:"services"`
		}
		if err := json.Unmarshal(data, &opts); err == nil {
			if svc, ok := opts.Services["tts"]; ok {
				if val, ok := svc["device"].(string); ok {
					device = val
				}
			}
		}
	}

	// Use shared Dexter Python 3.10 environment
	pythonEnvDir := filepath.Join(homeDir, "Dexter", "python3.10")
	pythonBin := filepath.Join(pythonEnvDir, "bin", "python")
	pipBin := filepath.Join(pythonEnvDir, "bin", "pip")

	// Ensure the shared environment exists
	if _, err := os.Stat(pythonBin); os.IsNotExist(err) {
		log.Fatalf("Shared Python 3.10 environment not found at %s. Run 'dex verify' or 'dex build' to fix.", pythonBin)
	}

	log.Println("Installing dependencies into shared environment...")
	pipCmd := exec.Command(pipBin, "install", "-r", "requirements.txt")
	pipCmd.Dir = serviceDir
	pipCmd.Stdout = os.Stdout
	pipCmd.Stderr = os.Stderr
	if err := pipCmd.Run(); err != nil {
		log.Printf("Warning: Failed to install dependencies: %v", err)
	}

	log.Println("Starting Dexter TTS Service...")

	pythonCmd := exec.Command(pythonBin, "main.py")
	pythonCmd.Dir = serviceDir

	// Inherit environment and add/override DEX variables
	v := version
	if v == "0.0.0" || v == "" {
		v = os.Getenv("DEX_VERSION")
	}

	b := branch
	if b == "unknown" || b == "" {
		b = os.Getenv("DEX_BRANCH")
	}

	c := commit
	if c == "unknown" || c == "" {
		c = os.Getenv("DEX_COMMIT")
	}

	pythonCmd.Env = append(os.Environ(),
		fmt.Sprintf("DEX_VERSION=%s", v),
		fmt.Sprintf("DEX_BRANCH=%s", b),
		fmt.Sprintf("DEX_COMMIT=%s", c),
		fmt.Sprintf("DEX_BUILD_DATE=%s", buildDate),
		fmt.Sprintf("DEX_BUILD_YEAR=%s", buildYear),
		fmt.Sprintf("DEX_ARCH=%s", arch),
		fmt.Sprintf("DEX_BUILD_HASH=%s", buildHash),
		fmt.Sprintf("DEX_TTS_DEVICE=%s", device),
	)

	pythonCmd.Stdout = os.Stdout
	pythonCmd.Stderr = os.Stderr
	// Handle signals
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
