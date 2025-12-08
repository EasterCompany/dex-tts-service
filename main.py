import os
import io
import logging
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
import torch
import contextlib # Import contextlib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dex-tts-service")

# Fix for PyTorch 2.6+ weights_only=True security change
# The "Nuclear Option": Monkey-patch torch.load to disable weights_only check
# during model initialization. We trust the source (Coqui official models).
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
# Dynamically select the best CUDA device, preferring 4060
if torch.cuda.is_available():
    best_device_index = 0
    best_capability = -1.0
    found_4060 = False
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"Found GPU {i}: {props.name} (CUDA Capability: {props.major}.{props.minor})")
        
        # Prioritize 4060 if found
        if "4060" in props.name:
            best_device_index = i
            found_4060 = True
            break
        
        # Fallback to highest capability if 4060 not explicitly named yet
        current_capability = float(f"{props.major}.{props.minor}")
        if current_capability > best_capability:
            best_capability = current_capability
            best_device_index = i
            
    DEVICE = f"cuda:{best_device_index}"
    logger.info(f"Selected device: {DEVICE} ({torch.cuda.get_device_name(best_device_index)})")
else:
    DEVICE = "cpu"
    logger.info("CUDA not available. Using CPU.")

# Constants
DEFAULT_SPEAKER_PATH = os.path.join(os.path.dirname(__file__), "assets", "reference.wav")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define lifespan event handler
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_model
    logger.info(f"Starting TTS Service on device: {DEVICE}")
    try:
        from TTS.api import TTS
        logger.info("Loading XTTS-v2 model (with weights_only=False override)...")
        
        # Use the context manager to allow unsafe globals during loading
        with unsafe_torch_load():
            tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)
            
        logger.info("XTTS-v2 model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load TTS model: {e}")
        tts_model = None
    yield
    # Cleanup / shutdown code if any
    logger.info("TTS Service shutdown complete.")

app = FastAPI(title="Dexter TTS Service", version="1.0.0", lifespan=lifespan) # Pass lifespan to FastAPI

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

@app.post("/generate")
async def generate_audio(request: GenerateRequest):
    global tts_model
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model is not available.")

    if not request.text:
        raise HTTPException(status_code=400, detail="Text is required.")

    # Validate speaker wav
    speaker_wav = request.speaker_wav
    if not os.path.exists(speaker_wav):
        # If default is missing, we can't clone
        if speaker_wav == DEFAULT_SPEAKER_PATH:
             raise HTTPException(status_code=500, detail=f"Default reference voice not found at {DEFAULT_SPEAKER_PATH}. Please add a reference.wav file.")
        
        # If custom path is missing, fallback or error
        raise HTTPException(status_code=400, detail=f"Speaker wav file not found: {speaker_wav}")

    try:
        logger.info(f"Generating audio for: '{request.text[:30]}...' ")
        
        # Generate to a temporary buffer/file
        # TTS API usually writes to file. We'll write to a unique temp file.
        import uuid
        filename = f"{uuid.uuid4()}.wav"
        output_path = os.path.join(OUTPUT_DIR, filename)

        tts_model.tts_to_file(
            text=request.text,
            file_path=output_path,
            speaker_wav=speaker_wav,
            language=request.language
        )

        # Read file back into memory
        with open(output_path, "rb") as f:
            audio_data = f.read()
        
        # Cleanup
        os.remove(output_path)

        return Response(content=audio_data, media_type="audio/wav")

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8200)
