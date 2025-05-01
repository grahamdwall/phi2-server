import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import math
from peft import PeftModel

base_model = "mistralai/Mistral-7B-Instruct-v0.2"
lora_adapter = "GrahamWall/lora-mortgage-mistral7b"

tokenizer = AutoTokenizer.from_pretrained(base_model)
base = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype=torch.float32)
model = PeftModel.from_pretrained(base, lora_adapter)


def calculate_ground_truth(principal, annual_rate, years):
    r = (annual_rate / 100) / 12
    n = years * 12
    if r == 0:
        return principal / n
    return principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)

def extract_numbers(text):
    return [float(n.replace(',', '')) for n in re.findall(r"\$?([\d,.]+)", text)]

# Load prompts
prompts = []
with open("baseline_eval.jsonl", "r") as f:
    for line in f:
        prompts.append(json.loads(line)["prompt"])

results = {
    "total": 0,
    "exact_match": 0,
    "small_error": 0,
    "large_error": 0,
    "invalid": 0
}

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    predicted_nums = extract_numbers(response)
    if not predicted_nums:
        results["invalid"] += 1
        continue

    predicted_payment = predicted_nums[0]

    m = re.findall(r"\$?([\d,]+)", prompt)
    percents = re.findall(r"(\d+(\.\d+)?)%", prompt)
    years = re.findall(r"(\d+)[ ]?years?", prompt)

    if len(m) >= 1 and percents and years:
        principal = float(m[0].replace(',', ''))
        annual_rate = float(percents[0][0])
        term_years = int(years[0])

        true_payment = calculate_ground_truth(principal, annual_rate, term_years)

        rel_error = abs(predicted_payment - true_payment) / true_payment

        if abs(predicted_payment - true_payment) <= 1.0:
            results["exact_match"] += 1
        elif rel_error <= 0.01:
            results["small_error"] += 1
        else:
            results["large_error"] += 1
    else:
        results["invalid"] += 1

    results["total"] += 1

print(results)
print(f"Exact match rate: {results['exact_match'] / results['total']:.2%}")
print(f"Small error rate: {results['small_error'] / results['total']:.2%}")
print(f"Large error rate: {results['large_error'] / results['total']:.2%}")
print(f"Invalid rate: {results['invalid'] / results['total']:.2%}")
