import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

def check_hf_connection():
    token = os.getenv('HUGGING_FACE_HUB_TOKEN')
    
    if not token:
        print("[ERROR] No HUGGING_FACE_HUB_TOKEN found in environment")
        return False
    
    try:
        api = HfApi()
        user_info = api.whoami(token=token)
        print(f"[SUCCESS] Connected to Hugging Face as: {user_info['name']}")
        print(f"[INFO] Email: {user_info.get('email', 'N/A')}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to connect to Hugging Face: {e}")
        return False

if __name__ == "__main__":
    check_hf_connection()