# ISSAI-RAG

Here's the updated README file, removing references to the `models/` and `outputs/` directories as per your request:

---

# NU-Embedder and Builder

## Project Overview
This project provides functionality for embedding and building vector representations from textual data using the `NU-builder.py` and `main-nu-embedder.py` scripts. The project is designed to preprocess large amounts of data and generate embeddings that can be used in downstream tasks such as information retrieval, classification, or clustering.

## Requirements
Ensure that you have the following libraries installed in your Python environment:
- `numpy`
- `pandas`
- `scikit-learn`
- `torch`
- `transformers`
- `sentence-transformers`
- `faiss-gpu`
- `tqdm`
- `json`

To install the required libraries, you can run:
```bash
pip install numpy pandas scikit-learn torch transformers sentence-transformers faiss-gpu tqdm json
```

## Installation Instructions

1. **Clone the repository** or copy the necessary Python files (`NU-builder.py` and `main-nu-embedder.py`) to your local machine.

2. **Create the following folder structure**:
   ```
   project-root/
   ├── data/
   │   ├── original/     # Original data files (input)
   ├── embeddings/       # Final embeddings stored as .npy and .csv files
   ```

   - `data/original`: Store your raw text files here (e.g., `.txt`, `.csv`).
   - `embeddings`: After running the embedding script, the `.npy` files (NumPy arrays) and `.csv` files containing embeddings will be stored here.

3. **Install the required Python libraries** (see the requirements section).

4. **Pretrained models**: The required models will be automatically downloaded from Hugging Face by the `sentence-transformers` library when the script is run. You do not need to manually download models.

## Configuration

- You may need to change the paths to your data and embedding directories in both the `NU-builder.py` and `main-nu-embedder.py` scripts.
- Update the variables in the scripts as follows:
  ```python
  data_dir = './data/original/'
  embeddings_dir = './embeddings/'
  ```
  to the appropriate paths based on where your data and embeddings are located.

- **Model Selection**: You can change the embedding model by editing the model loading section in `main-nu-embedder.py`. For example:
  ```python
  from sentence_transformers import SentenceTransformer
  model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
  ```

## How to Run

1. **Preprocessing the Data**:
   Use `NU-builder.py` to preprocess your raw text data:
   ```bash
   python NU-builder.py
   ```
   This script will clean, tokenize, and preprocess the data.

2. **Embedding Generation**:
   After preprocessing, run `main-nu-embedder.py` to generate embeddings:
   ```bash
   python main-nu-embedder.py
   ```
   This script will use the pretrained model to generate embeddings for each processed text and store the embeddings in both `.npy` and `.csv` format in the `embeddings/` folder.

## Example Usage

To preprocess a dataset and generate embeddings, follow these steps:

1. Place your raw text files in the `data/original/` folder.
2. Run the following commands:

   **Step 1**: Preprocess the text data:
   ```bash
   python NU-builder.py
   ```

   **Step 2**: Generate embeddings from the preprocessed data:
   ```bash
   python main-nu-embedder.py
   ```

3. The embeddings will be saved in the `embeddings/` folder as `.npy` files (containing the embedding vectors) and `.csv` files (containing the embeddings with corresponding identifiers).

## Folder Structure After Execution

After running the scripts, your folder structure should look like this:
   ```
   project-root/
   ├── data/
   │   ├── original/     # Contains original input data files
   ├── embeddings/       # Contains .npy and .csv files with embeddings
   ```

## Troubleshooting

- **Memory issues**: If you encounter memory issues with large datasets, consider batching the data in smaller chunks in both `NU-builder.py` and `main-nu-embedder.py`.
- **Model loading errors**: If there are issues loading models, ensure you have the correct Hugging Face models downloaded and that the internet connection is stable.

---

Let me know if you need further modifications!
