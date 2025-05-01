import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# SETTINGS
MODEL_NAME = "microsoft/phi-2"
DEVICE = "cpu"  # We'll assume CPU for now

print(f"Loading model {MODEL_NAME} on {DEVICE}...")

# CONFIGURE LOADING
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# LOAD MODEL
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",  # If RunPod, you can switch to "auto"
    quantization_config=quant_config,
    trust_remote_code=True,
)

# LOAD TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# SIMPLE INFERENCE EXAMPLE
prompt = "What is the weekly payment for a $500,000 loan at 6% interest over 25 years?"

inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
outputs = model.generate(**inputs, max_new_tokens=100)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Response:", response)