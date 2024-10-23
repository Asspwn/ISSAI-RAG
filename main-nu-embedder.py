import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import re
import logging
from sentence_transformers import LoggingHandler
import time
import csv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
import multiprocessing
from tqdm import tqdm
from typing import List
import os

start_time = time.time()

# Setup logging
logging.basicConfig(
    format='%(asctime)s-%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[LoggingHandler()]
)

# Set Hugging Face API token
HF_TOKEN = "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"

# Set tokenizer parallelism to false
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set number of processes for multiprocessing
NUM_PROCESSES = 64

# Use the specified Llama tokenizer from Hugging Face, explicitly passing the token
tokenizer = AutoTokenizer.from_pretrained("nlp-team-issai/multilingual-e5-large-instruct-part-kkwiki-en-kk-ru-161024", use_auth_token=HF_TOKEN)

# Function to split texts into chunks using RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

# Function to split texts into chunks, ensuring sentence integrity
def process_chunk(args: tuple) -> List[tuple]:
    filename, text = args  # Only use filename and text
    sentences = sent_tokenize(text)  # Tokenize into sentences
    chunks = []
    current_chunk = []
    current_length = 0
    
    # Iterate over the sentences and combine them into chunks
    for sentence in sentences:
        sentence_length = len(tokenizer(sentence)["input_ids"])
        
        if current_length + sentence_length <= 512:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            # Save the current chunk as a single text
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
    
    # Append any remaining sentences as a final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    # Ensure we don't have extremely small chunks by merging them with previous chunks
    if len(chunks) > 1 and len(tokenizer(chunks[-1])["input_ids"]) < 100:
        chunks[-2] = chunks[-2] + " " + chunks[-1]
        chunks.pop(-1)
    
    # Return the filename with its respective text chunks
    return [(filename, chunk) for chunk in chunks]

# The rest of the code remains unchanged

# Function to process text batches in parallel using multiprocessing
BATCH_SIZE = 16

def process_batches(texts: List[tuple]) -> List[List[tuple]]:
    total_batches = len(texts) // BATCH_SIZE + (1 if len(texts) % BATCH_SIZE > 0 else 0)

    with multiprocessing.Pool(NUM_PROCESSES) as pool:
        results = list(tqdm(
            pool.imap(process_chunk, texts),
            total=total_batches,
            desc="Processing documents"
        ))

    return results

# Function to load and preprocess the entire CSV file
def load_and_preprocess_file(file_path):
    # Read the entire file into a DataFrame
    df = pd.read_csv(file_path)

    # Drop rows where 'text' column has NaN values
    df = df.dropna(subset=['text'])

    # Drop duplicates based on 'text' column only
    df = df.drop_duplicates(subset=['text'])

    return df

# List of data files to process
data_files = {
    'NU-regs': '/raid/aspandiyar_nurimanov/RAG-project/Aueskhan-RAG/NU-regulations-RAG/data/combined_nu_regulations.csv',
    'NU-guests': '/raid/aspandiyar_nurimanov/RAG-project/Aueskhan-RAG/NU-regulations-RAG/data/combined_nu_regulations.csv'
}

# Function to process a single CSV file
def process_file(file_key, file_path, model):
    # Load and preprocess the entire file
    df = load_and_preprocess_file(file_path)

    # Prepare text data with original indices to maintain order
#    indexed_texts = list(zip(df.index, df['filename'], df['text']))  # Get tuples of (index, title, text)
    indexed_texts = list(zip(df['filename'], df['text']))  # Get tuples of (filename, text)


    # Process the text data
    processed_results = process_batches(indexed_texts)
    flattened_results = [item for sublist in processed_results for item in sublist]

    # Create a new DataFrame from the processed, split texts
    chunked_df = pd.DataFrame(flattened_results, columns=['filename', 'text'])

    # Start the multi-process pool on all available GPUs
    pool = model.start_multi_process_pool(target_devices=['cuda:3', 'cuda:4', 'cuda:5', 'cuda:0', 'cuda:9', 'cuda:10', 'cuda:1'])

    # Computing the embeddings using the multi-process pool
    emb = model.encode_multi_process(chunked_df['text'].tolist(), pool)
    print(f'Embeddings computed for {file_key}. Shape:', emb.shape)

    # Optional: Stop the processes in the pool
    model.stop_multi_process_pool(pool)

    return chunked_df['filename'].tolist(), chunked_df['text'].tolist(), emb


if __name__ == '__main__':
    # Load the model
    model = SentenceTransformer('nlp-team-issai/multilingual-e5-large-instruct-part-kkwiki-en-kk-ru-161024')

    # Initialize lists to hold all the final titles, texts, and embeddings
    final_titles = []
    final_texts = []
    final_embeddings = []

    # Process each file and accumulate titles, texts, and embeddings
    for file_key, file_path in data_files.items():
        print(f"Processing file: {file_key}")
        titles, texts, embeddings = process_file(file_key, file_path, model)

        # Append to the final list of titles, texts, and embeddings
        final_titles.extend(titles)
        final_texts.extend(texts)
        final_embeddings.append(embeddings)

    # Concatenate all embeddings from all files
    final_embeddings = np.concatenate(final_embeddings, axis=0)

    # Save the final titles and texts to a CSV file
    with open('/raid/aspandiyar_nurimanov/RAG-project/Aueskhan-RAG/NU-regulations-RAG/embeddings/NU-v1-multilingual-e5-large.csv', mode='w', newline='') as text_file:
        writer = csv.writer(text_file)
        writer.writerow(['chunk_index', 'filename', 'text'])  # Header row
        for chunk_index, (title, text) in enumerate(zip(final_titles, final_texts)):
            writer.writerow([chunk_index, title, text])

    # Save the final embeddings to a .npy file
    np.save('/raid/aspandiyar_nurimanov/RAG-project/Aueskhan-RAG/NU-regulations-RAG/embeddings/NU-v1-multilingual-e5-large.npy', final_embeddings)

    print("--- %s seconds ---" % (time.time() - start_time))
