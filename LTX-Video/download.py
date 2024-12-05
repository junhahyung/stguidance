from huggingface_hub import snapshot_download

model_path = './weights'   # The local directory to save downloaded checkpoint
snapshot_download("Lightricks/LTX-Video", local_dir=model_path, local_dir_use_symlinks=False, repo_type='model')