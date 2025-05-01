import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
import random

# -------------- SETTINGS ------------------
BASE_MODEL = "Qwen/Qwen1.5-1.8B-Chat"
OUTPUT_DIR = "./qwen-finetuned-mortgage"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
EPOCHS = 1
LEARNING_RATE = 2e-4

# Mortgage examples for evaluation and fine-tuning
data = [
    {"principal": 100000, "rate": 5.0, "expected_payment": 1347.13},
    {"principal": 250000, "rate": 3.5, "expected_payment": 1122.61},
    {"principal": 500000, "rate": 6.0, "expected_payment": 2998.57},
    {"principal": 350000, "rate": 4.5, "expected_payment": 1773.40},
]

# -------------- LOAD BASE MODEL ------------------
print(f"Loading model {BASE_MODEL} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="cpu",  # FORCE CPU instead of MPS
    torch_dtype=torch.float32,  # FORCE float32 to avoid bfloat16 crash    
    trust_remote_code=True
)

# -------------- BASELINE EVALUATION ------------------
def mortgage_prompt(principal, rate):
    return f"""
User: I want to calculate my mortgage. The principal amount is ${principal} and the annual interest rate is {rate}%. What is my weekly mortgage payment?
Assistant:
"""

def evaluate(model, tokenizer, examples):
    model.eval()
    correct = 0
    for ex in examples:
        prompt = mortgage_prompt(ex["principal"], ex["rate"])
        print(f"Prompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {response}")
        if str(round(ex["expected_payment"])) in response:
            correct += 1
    accuracy = correct / len(examples)
    return accuracy

print("\nRunning baseline evaluation...")
baseline_acc = evaluate(model, tokenizer, data)
print(f"Baseline Accuracy: {baseline_acc * 100:.2f}%")

# -------------- PREPARE FOR FINE-TUNING ------------------
print("\nSetting up LoRA fine-tuning...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Prepare dataset
train_texts = [
    mortgage_prompt(ex["principal"], ex["rate"]) + f" Weekly Payment: ${ex['expected_payment']}"
    for ex in data
]
train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt")
class MortgageDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return len(self.encodings["input_ids"])
    def __getitem__(self, idx):
        return {key: tensor[idx] for key, tensor in self.encodings.items()}

dataset = MortgageDataset(train_encodings)

# Fine-tuning arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    save_total_limit=1,
    save_steps=5,
    logging_steps=5,
    bf16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# -------------- FINE-TUNING ------------------
print("\nStarting fine-tuning...")
trainer.train()

# Save model
print("Saving fine-tuned model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# -------------- POST-FINETUNING EVALUATION ------------------
print("\nReloading fine-tuned model for evaluation...")
model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR, device_map="auto" if DEVICE == "cuda" else None)
model.eval()

print("\nRunning evaluation after fine-tuning...")
final_acc = evaluate(model, tokenizer, data)
print(f"Post-Fine-Tuning Accuracy: {final_acc * 100:.2f}%")

# -------------- SUMMARY ------------------
print("\n==================")
print(f"Baseline Accuracy: {baseline_acc * 100:.2f}%")
print(f"Post-Fine-Tuning Accuracy: {final_acc * 100:.2f}%")
print("==================")
