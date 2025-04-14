import os
import logging
import fitz
import ollama
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from huggingface_hub import HfApi, create_repo

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def read_pdf_content(pdf_path):
    """
    Reads a PDF and returns its content as a list of strings.

    Args:
    pdf_path (str): The file path to the PDF.

    Returns:
    list of str: A list where each element is the text content of a PDF page.
    """
    content_list = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            content_list.append(page.get_text())

    return content_list

def call_ollama(query: str, model: str = "cesarchamal/qa-expert") -> str:
    logging.debug(f"Calling Ollama with query: {query}, model: {model}")

    # Prepare the conversation context with system and user messages.
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Question: {query}."},
    ]

    content = ""  # Initialize a variable to accumulate content
    try:
        # Simulating multiple response objects for demonstration
        response = ollama.chat(
            model=model,
            messages=messages,
            stream=False
        )
        logging.debug(f"Raw Response Text: {response}")

        if 'message' in response and 'content' in response['message']:
            content = response['message']['content']
        else:
            content = "No response or unexpected response structure."
        logging.debug(f"Received response: {content}")

    except Exception as e:
        content = "Error occurred while calling the Ollama API."
        logging.error(f"Error: {str(e)}")

    return content

def prompt_engineered_api(text: str):

    prompt = f"""
        I have the following content: {text}

        I want to create a question-answer content that has the following format:

        ### Human:
        ### Assistant:

        Make sure to write question and answer based on the content I provided.

        The ### Human means it's a question, and the ### Assistant means it's an answer.
    """

    resp = call_ollama(prompt)

    return resp


scraped_content = read_pdf_content("./jvm_troubleshooting_guide.pdf")
print("\n")
print(scraped_content)

scraped_content = ' '.join(scraped_content)
print(scraped_content)

resp = prompt_engineered_api(scraped_content.split('. ')[0])


raw_content_for_train = []

for i in tqdm(range(len(scraped_content.split('. ')))):
    resp = prompt_engineered_api(scraped_content.split('. ')[i])
    raw_content_for_train.append(resp)

# Example data - replace these with your actual data
train_data = {'text': raw_content_for_train[0:100]}
test_data = {'text': raw_content_for_train[100::]}

# Create Dataset objects for training and testing
train_dataset = Dataset.from_dict(train_data)
test_dataset = Dataset.from_dict(test_data)

# Combine them into a DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

# Display the structure of the dataset
print(dataset_dict)

# Replace 'your_token_here' with your actual Hugging Face Auth token
# Replace 'youthless-homeless-shelter-web-scrape-dataset' with your desired repository name
auth_token = os.getenv('HUGGING_FACE_HUB_TOKEN')
repo_name = 'jvm_troubleshooting_guide'
username = 'CesarChaMal' # replace with your Hugging Face username

api = HfApi()
create_repo(repo_name, token=auth_token, private=False) # Set private=True if you want it to be a private dataset


app_id = f"{username}/{repo_name}"
print(app_id)

dataset_dict.push_to_hub(app_id)
