# phi2_api.py
# Air-Gapped deployment: offline, cloud, serverless
# source venv/bin/activate
# tree -L 3
import os
import time
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn
from contextlib import asynccontextmanager

# Set your paths
LOCAL_MODEL_PATH = "/workspace/data/models/microsoft/phi-2"
HF_MODEL_PATH = "GrahamWall/phi2-finetune"
local_files_only_computed = None
trust_remote_code_computed = None

# Air-Gapped deployment: offline, cloud, serverless
print("[DEBUG] Starting phi2_api.py... 🚀")
print(f"[DEBUG] torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"[DEBUG] torch.cuda.device_count(): {torch.cuda.device_count()}")
print(f"[DEBUG] Environment Variables: {os.environ}")

print("Current working directory:", os.getcwd())

# Check for local files
def model_files_exist(path):
    required_files = ["added_tokens.json", 
                      "config.json", 
                      "generation_config.json", 
                      "model-00001-of-00002.safetensors", 
                      "model-00002-of-00002.safetensors", 
                      "model.safetensors.index.json", 
                      "special_tokens_map.json", 
                      "tokenizer_config.json", 
                      "tokenizer.json", 
                      "vocab.json"]
    return all(os.path.isfile(os.path.join(path, f)) for f in required_files)

# Decide where to load from
if os.path.isdir(LOCAL_MODEL_PATH) and model_files_exist(LOCAL_MODEL_PATH):
    print(f"✅ Loading model from local path: {LOCAL_MODEL_PATH}")
    model_path = LOCAL_MODEL_PATH
    local_files_only_computed = True
    trust_remote_code_computed = False
else:
    print(f"⚠️ Local model not found, loading from Hugging Face: {HF_MODEL_PATH}")
    model_path = HF_MODEL_PATH
    local_files_only_computed = False
    trust_remote_code_computed = True


# --- SETUP ---
# If running inside Docker container
#MODEL_PATH = "/app/models/microsoft/phi-2" if os.path.exists("/app/models/microsoft/phi-2") else "./phi2_model_full"
#model_name = "microsoft/phi-2"
# If running Docker container, but model  is on persistent storage in RunPod, not in container app
#MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/data/model/phi-2")

#print(f"[DEBUG] Loading model from: {MODEL_PATH}")

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
MAX_NUM_TOKENS = 100 if torch.cuda.is_available() else 1

print(f"[DEBUG] torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"[DEBUG] Dtype selected: {dtype}")

#print(f"Loading {MODEL_PATH} on {device}...")
#tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

#try:
#    print("[DEBUG] Attempting to load model...")
#    model = AutoModelForCausalLM.from_pretrained(
#        MODEL_PATH,
#        device_map="auto",
#        torch_dtype=dtype,
#        trust_remote_code=True,
#    )
#    print("[DEBUG] Model loaded successfully ✅")
#except Exception as e:
#    print(f"[ERROR] Exception during model loading: {str(e)}")
#    raise

#model = model.to(device)

def generate_text(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=MAX_NUM_TOKENS)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

model = None
tokenizer = None

# --- FASTAPI SETUP ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    print("[DEBUG] Starting up app")
    try:
        print("[DEBUG] Attempting to load model...")
        #model_path = os.getenv("MODEL_PATH", "/workspace/data/models/microsoft/phi-2")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=dtype,
            local_files_only=local_files_only_computed,
            trust_remote_code=trust_remote_code_computed,
        )
        print("[DEBUG] Model loaded successfully ✅")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            local_files_only=local_files_only_computed, 
            trust_remote_code=trust_remote_code_computed,
        )
        print("[DEBUG] Tokenizer loaded ✅")
        test_output = generate_text("Hello, world!")
        print(f"[DEBUG] Startup generation success: {test_output}")
        # if we loaded from HF successfully, save to local path for next time
        if model_path == HF_MODEL_PATH:
            # Create directory if it doesn't exist
            os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
            model.save_pretrained(LOCAL_MODEL_PATH)
            tokenizer.save_pretrained(LOCAL_MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Exception during model loading: {str(e)}")
        raise
    yield
    print("[DEBUG] Shutting down app")

app = FastAPI(lifespan=lifespan)
print("[DEBUG] FastAPI app created ✅")

class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50

def generate_text_sync(prompt: str, max_new_tokens: int = 50) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# curl -X POST http://157.157.221.29:34378/runsync \
#  -H "Content-Type: application/json" \
#  -d '{"prompt": "Hello, world!", "max_new_tokens": 50}'
@app.post("/runsync")
def runsync(req: GenerationRequest):
    try:
        print(f"[DEBUG] Received prompt: {req.prompt}")
        start = time.time()
        output = generate_text_sync(req.prompt, req.max_new_tokens)
        print(f"[DEBUG] Response generated ✅ in {time.time() - start:.2f}s: {output}")
        return {"output": output}
    except Exception as e:
        print(f"[ERROR] Exception during generation: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# *** note that there is no trailing slash in the endpoint URL, by convention
#@app.post("/runsync")
#async def runpod_handler(request: Request):
#    print("[DEBUG] /runsync endpoint HIT ✅")

#    try:
#        # Read and log JSON payload
#        data = await request.json()
#        print(f"[DEBUG] Received data: {data}")
#        print(f"[DEBUG] Request headers: {request.headers}")
#
#        # Extract input object
#        input_data = data.get("input", {})
#        print(f"[DEBUG] Input data: {input_data}")
#
#        # Handle /healthz
#        if input_data.get("path") == "/healthz":
#            print("[DEBUG] /healthz endpoint was called ✅")
#            return {"status": "ok ✅"}
#
#        # Handle prompt
#        elif "prompt" in input_data:
#            prompt = input_data["prompt"]
#            print(f"[DEBUG] Received prompt: {prompt}")
#
#            # Tokenize and generate response
#            response = generate_text(prompt)
#
#            print("[DEBUG] Response generated ✅")
#            return {"response": response}
#
#        else:
#            print("[WARN] 'prompt' not found in input.")
#            return JSONResponse(status_code=400, content={"error": "Missing 'prompt' field in input"})
#
#    except Exception as e:
#        print(f"[ERROR] Exception while handling /runsync: {e}")
#        return JSONResponse(status_code=400, content={"error": "Invalid input format or internal error"})

# e.g. curl http://157.157.221.29:34378/health
@app.get("/health")
def health_check():
    return {"status": "ok"}

# e.g. curl http://157.157.221.29:34378/
@app.get("/")
def read_root():
    return {"message": "LLM server is running"}

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    import traceback
    print("[ERROR]", traceback.format_exc())
    return JSONResponse(status_code=500, content={"error": str(exc)})

# note: this code will not run in RunPod since we start uvicorn from the dockerfile
# but it is useful for local testing
# will start server using gunicorn as an external process in production
if __name__ == "__main__":
    import uvicorn
    print("[DEBUG] Starting Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
