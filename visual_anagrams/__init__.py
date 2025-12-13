# src/__init__.py

# import os
# from dotenv import load_dotenv
# import wandb
# from pathlib import Path

# # # Set environment variables for caching
# # os.environ["HF_HOME"] = "/Data/shash/.cache"
# # os.environ["HF_HUB_CACHE"] = "/Data/shash/.cache/hub"

# # Load environment variables from .env file
# load_dotenv()

# # # Get API tokens
# # hf_token = os.getenv("HF_TOKEN")
# # wandb_token = os.getenv("WANDB_API_KEY")

# # # Login to Hugging Face
# # from huggingface_hub import login
# # if hf_token:
# #     login(hf_token)
# #     print("Logged in to Hugging Face successfully!")
# # else:
# #     print("HF_TOKEN not found in .env file.")

# # # Login to Weights & Biases
# # if wandb_token:
# #     wandb.login(key=wandb_token)
# #     print("Logged in to Weights & Biases successfully!")
# # else:
# #     print("WANDB_TOKEN not found in .env file.")

# # Define project paths
# root = Path(__file__).parent.parent
# outputs = root / "outputs"
# outputs.mkdir(parents=True, exist_ok=True)