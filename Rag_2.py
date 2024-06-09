import os
import httpx
from sentence_transformers import SentenceTransformer, util
import torch
import warnings

warnings.filterwarnings("ignore")

api_token = "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"

headers = {
    "Authorization": f"Bearer {api_token}"
}

model_id = "google/flan-t5-xxl"

file_path = "input_file.txt"  
with open(file_path, "r", encoding="utf-8") as file:
    input_text = file.read()

chunks = input_text.split('\n')

# Load a pre-trained model for creating embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for each chunk
chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True)

# Ask the user to input the question in the terminal
question = input("Please enter your question: ")

# Create an embedding for the question
question_embedding = embedder.encode(question, convert_to_tensor=True)

# Compute cosine similarities between the question and each chunk
cosine_scores = util.pytorch_cos_sim(question_embedding, chunk_embeddings)[0]

# Find the index of the most similar chunk
top_k = min(5, len(chunks))  # Retrieve top 5 chunks
top_results = torch.topk(cosine_scores, k=top_k)

# Retrieve the most relevant chunks
retrieved_chunks = [chunks[idx] for idx in top_results[1]]

context = " ".join(retrieved_chunks)

# Combine the context and the question
input_text = f"Context: {context}\n\nQuestion: {question}"

# Calculate the max input tokens to fit within the token limit
max_input_tokens = 500

# The URL for the Hugging Face Inference API
api_url = f"https://api-inference.huggingface.co/models/{model_id}"

# The payload for the request
payload = {
    "inputs": input_text,
    "parameters": {
        "max_new_tokens": max_input_tokens, 
        "min_length": 50,     
        "do_sample": False 
    }
}

# Make the request to the API using httpx
response = httpx.post(api_url, headers=headers, json=payload)
if response.status_code == 200:
    generated_text = response.json()
    print("Generated text:", generated_text)
else:
    print("Request failed with status code:", response.status_code)
    print("Response:", response.text)
