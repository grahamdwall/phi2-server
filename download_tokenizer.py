from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

tokenizer.save_pretrained("./phi2_tokenizer")
