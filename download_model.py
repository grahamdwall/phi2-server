from huggingface_hub import snapshot_download

# This will download the full model and tokenizer
snapshot_download(repo_id="microsoft/phi-2", local_dir="./phi2_model_full", local_dir_use_symlinks=False)

