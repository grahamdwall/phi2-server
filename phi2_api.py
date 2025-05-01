# phi2_api.py
# Air-Gapped deployment: offline, cloud, serverless
# source venv/bin/activate
# tree -L 3
import os
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
#from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn
from contextlib import asynccontextmanager

# Air-Gapped deployment: offline, cloud, serverless
print("[DEBUG] Starting phi2_api.py... 🚀")
print(f"[DEBUG] torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"[DEBUG] torch.cuda.device_count(): {torch.cuda.device_count()}")
print(f"[DEBUG] Environment Variables: {os.environ}")

# --- SETUP ---
# If running inside Docker container
MODEL_PATH = "/app/models/microsoft/phi-2" if os.path.exists("/app/models/microsoft/phi-2") else "./phi2_model_full"
#model_name = "microsoft/phi-2"

print(f"[DEBUG] Loading model from: {MODEL_PATH}")

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
MAX_NUM_TOKENS = 100 if torch.cuda.is_available() else 1

print(f"[DEBUG] torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"[DEBUG] Dtype selected: {dtype}")

print(f"Loading {MODEL_PATH} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("[DEBUG] Tokenizer loaded ✅")

try:
    print("[DEBUG] Attempting to load model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    print("[DEBUG] Model loaded successfully ✅")
except Exception as e:
    print(f"[ERROR] Exception during model loading: {str(e)}")
    raise

#model = model.to(device)

def generate_text(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=MAX_NUM_TOKENS)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- FASTAPI SETUP ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[DEBUG] Starting up app")
    try:
        test_output = generate_text("Hello, world!")
        print(f"[DEBUG] Startup generation success: {test_output}")
    except Exception as e:
        print(f"[ERROR] Model warm-up failed: {e}")
    yield
    print("[DEBUG] Shutting down app")

app = FastAPI(lifespan=lifespan)

print("[DEBUG] FastAPI app created ✅")

# implementing as a serverless function, not a web server
# *** note that there is no trailing slash in the endpoint URL, by convention for a serverless function
@app.post("/runsync")
async def runpod_handler(request: Request):
    print("[DEBUG] /runsync endpoint HIT ✅")

    try:
        # Read and log JSON payload
        data = await request.json()
        print(f"[DEBUG] Received data: {data}")
        print(f"[DEBUG] Request headers: {request.headers}")

        # Extract input object
        input_data = data.get("input", {})
        print(f"[DEBUG] Input data: {input_data}")

        # Handle /healthz
        if input_data.get("path") == "/healthz":
            print("[DEBUG] /healthz endpoint was called ✅")
            return {"status": "ok ✅"}

        # Handle prompt
        elif "prompt" in input_data:
            prompt = input_data["prompt"]
            print(f"[DEBUG] Received prompt: {prompt}")

            # Tokenize and generate response
            response = generate_text(prompt)

            print("[DEBUG] Response generated ✅")
            return {"response": response}

        else:
            print("[WARN] 'prompt' not found in input.")
            return JSONResponse(status_code=400, content={"error": "Missing 'prompt' field in input"})

    except Exception as e:
        print(f"[ERROR] Exception while handling /runsync: {e}")
        return JSONResponse(status_code=400, content={"error": "Invalid input format or internal error"})

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
if __name__ == "__main__":
    import uvicorn
    print("[DEBUG] Starting Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
