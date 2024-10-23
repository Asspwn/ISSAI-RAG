# ISSAI-RAG

To create a proper README file for your project, and provide a tutorial and guide for users, we'll need to include the following sections:

1. **Project Overview**:
   - A brief description of the project and its purpose.
   
2. **Requirements**:
   - Specify the required Python libraries and versions.
   
3. **Installation Instructions**:
   - Step-by-step guide on how to set up the project environment.

4. **Folder Structure**:
   - Description of the directory structure that users should set up.

5. **Configuration**:
   - Information on configuration settings users may need to change in the scripts.

6. **How to Run**:
   - Instructions on how to execute the scripts.

7. **Example Usage**:
   - Example commands and expected results.

8. **Troubleshooting**:
   - Common issues and solutions.

Here is a draft README file based on these sections:

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
   │   ├── raw/
   │   ├── processed/
   ├── models/
   └── outputs/
   ```

   - `data/raw`: Store your raw text files here.
   - `data/processed`: This is where the preprocessed data will be stored.
   - `models`: Pretrained or fine-tuned models will be stored here.
   - `outputs`: Store the final embeddings and logs.

3. **Install the required Python libraries** (see the requirements section).

4. **Pretrained models**: If you're using pretrained models (e.g., BERT, RoBERTa), download the necessary models from Hugging Face's model hub, such as `sentence-transformers/all-mpnet-base-v2`, and store them in the `models` directory.

## Configuration

- You may need to change the paths to your data and model directories in both the `NU-builder.py` and `main-nu-embedder.py` scripts.
- Update the variables:
  ```python
  data_dir = './data/raw/'
  processed_dir = './data/processed/'
  model_dir = './models/'
  output_dir = './outputs/'
  ```
  to the appropriate paths based on where your data and models are located.

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
   This will clean, tokenize, and preprocess the data, saving the output to `data/processed/`.

2. **Embedding Generation**:
   After preprocessing, run `main-nu-embedder.py` to generate embeddings:
   ```bash
   python main-nu-embedder.py
   ```
   This script will use the pretrained model to generate embeddings for each processed text and store the embeddings in the `outputs/` folder.

## Example Usage

To preprocess a dataset and generate embeddings, follow these steps:

1. Place your raw text files in the `data/raw/` folder.
2. Run the following commands:

   **Step 1**: Preprocess the text data:
   ```bash
   python NU-builder.py
   ```

   **Step 2**: Generate embeddings from the preprocessed data:
   ```bash
   python main-nu-embedder.py
   ```

3. The embeddings will be saved in the `outputs/` folder as a `.npy` file.

## Troubleshooting

- **Memory issues**: If you encounter memory issues with large datasets, consider batching the data in smaller chunks in both `NU-builder.py` and `main-nu-embedder.py`.
- **Model loading errors**: If there are issues loading models, ensure you have the correct Hugging Face models downloaded and that the internet connection is stable.

---

This is a general structure for your README. You may need to refine it further based on specific functionalities or requirements in the project. Let me know if you need any changes or additional guidance!
