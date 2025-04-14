import os
import logging
import fitz
import ollama
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from huggingface_hub import HfApi, create_repo, delete_repo
from huggingface_hub.utils import HfHubHTTPError


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def read_pdf_content(pdf_path):
    content_list = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            content_list.append(page.get_text())
    return content_list

def call_ollama(query: str, model: str = "cesarchamal/qa-expert") -> str:
    logging.debug(f"Calling Ollama with query: {query}, model: {model}")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Question: {query}."},
    ]
    try:
        response = ollama.chat(model=model, messages=messages, stream=False)
        logging.debug(f"Raw Response Text: {response}")
        if 'message' in response and 'content' in response['message']:
            return response['message']['content']
        return "No response or unexpected response structure."
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return "Error occurred while calling the Ollama API."

def prompt_engineered_api(text: str):
    prompt = f"""
        I have the following content: {text}
        I want to create a question-answer content that has the following format:

        ### Human:
        ### Assistant:

        Make sure to write question and answer based on the content I provided.
    """
    return call_ollama(prompt)

# Step 1: Read and preprocess PDF
scraped_content = ' '.join(read_pdf_content("./jvm_troubleshooting_guide.pdf"))
sentences = scraped_content.split('. ')
logging.info(f"Total segments: {len(sentences)}")

# Step 2: Generate QA pairs using Ollama
raw_content_for_train = []
for sentence in tqdm(sentences, desc="Generating Q&A"):
    raw_content_for_train.append(prompt_engineered_api(sentence))

# Step 3: Create datasets
train_data = {'text': raw_content_for_train[0:100]}
test_data = {'text': raw_content_for_train[100:]}
dataset_dict = DatasetDict({
    'train': Dataset.from_dict(train_data),
    'test': Dataset.from_dict(test_data)
})
print(dataset_dict)

# Step 4: Push to Hugging Face Hub
auth_token = os.getenv('HUGGING_FACE_HUB_TOKEN')
repo_name = 'jvm_troubleshooting_guide'
username = 'CesarChaMal'
app_id = f"{username}/{repo_name}"
api = HfApi()

# 🧹 Optional: delete existing repo if it exists
try:
    logging.info(f"Attempting to delete existing dataset repo: {app_id}")
    delete_repo(repo_id=app_id, token=auth_token, repo_type="dataset")
    logging.info(f"Deleted existing dataset: {app_id}")
except HfHubHTTPError as e:
    if e.response.status_code == 404:
        logging.info(f"No existing repo found: {app_id}, proceeding...")
    else:
        logging.warning(f"Could not delete repo: {e}")

# ✅ Recreate the repo
try:
    create_repo(repo_name, token=auth_token, private=False, repo_type="dataset")
    logging.info(f"Created dataset repo: {app_id}")
except HfHubHTTPError as e:
    if e.response.status_code == 409:
        logging.info(f"Repo {app_id} already exists, continuing...")
    else:
        raise

# Step 5: Push dataset
logging.info(f"Pushing dataset to: {app_id}")
dataset_dict.push_to_hub(app_id)
print(f"✅ Dataset successfully pushed to: https://huggingface.co/datasets/{app_id}")
