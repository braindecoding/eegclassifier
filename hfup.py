from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="./checkpoints",
    repo_id="braindecoding/BANDWiseEMDHHT",
    repo_type="dataset",
)
