import numpy as np
import pandas as pd
import faiss
from haystack import Document
from haystack.document_stores import FAISSDocumentStore
import time

start_time = time.time()

# Step 1: Load Precomputed Embeddings and Corresponding Texts
embeddings_path = "/raid/aspandiyar_nurimanov/RAG-project/Aueskhan-RAG/NU-regulations-RAG/embeddings/issai_161024_modified_embeddings_multilingual-e5-large.npy"
texts_path = "/raid/aspandiyar_nurimanov/RAG-project/Aueskhan-RAG/NU-regulations-RAG/embeddings/issai_161024_modified_corresponding_multilingual-e5-large.csv"

print("Loading precomputed embeddings and corresponding texts...")
precomputed_embeddings = np.load(embeddings_path)
corresponding_texts = pd.read_csv(texts_path)

# Check the dimensionality of embeddings
num_documents, embedding_dim = precomputed_embeddings.shape
print(f"Loaded {num_documents} documents with {embedding_dim}-dimensional embeddings.")

# Create the documents list with corresponding texts and titles as metadata
documents = []
for i in range(num_documents):
    content = corresponding_texts.iloc[i]['text']
    title = corresponding_texts.iloc[i]['filename']  # Assuming 'filename' is a column in your CSV
    documents.append(Document(content=content, id=str(i), meta={"filename": title}))

print(f"Time elapsed after Step 1: {time.time() - start_time:.2f} seconds")

# Step 2: Create and Train the FAISS Index (Flat Index)
print("Initializing FAISS flat index (no IVF, no PQ)...")

# Initialize GPU resources
res = faiss.StandardGpuResources()

d = embedding_dim  # Dimensionality of embeddings

# Create a flat index for inner product similarity (use IndexFlatIP for cosine or IndexFlatL2 for Euclidean)
index_flat_cpu = faiss.IndexFlatIP(d)

# Move the index to GPU 15 (example GPU ID, adjust as necessary)
index_flat = faiss.index_cpu_to_gpu(res, 15, index_flat_cpu)

# Add all embeddings directly to the index
print("Adding embeddings to the FAISS flat index on GPU 15...")
index_flat.add(precomputed_embeddings)

print(f"Time elapsed after Step 2: {time.time() - start_time:.2f} seconds")

# Step 3: Save the FAISS Index to Disk
print("Saving the FAISS flat index to disk...")
# Move the index back to CPU before saving
index_flat_cpu = faiss.index_gpu_to_cpu(index_flat)
faiss.write_index(index_flat_cpu, "faiss_flat_index.faiss")

print(f"Time elapsed after Step 3: {time.time() - start_time:.2f} seconds")

# Step 4: Initialize FAISSDocumentStore with the Pre-trained Index
print("Initializing FAISSDocumentStore with the pre-trained FAISS flat index...")
# Load the index from the file
faiss_index = faiss.read_index("faiss_flat_index.faiss")

# Initialize the FAISSDocumentStore
document_store = FAISSDocumentStore(
    sql_url="sqlite:///faiss_document_store.db",
    faiss_index=faiss_index,
    index="document",
    similarity="dot_product",
    embedding_dim=d,
    validate_index_sync=False  # Disable automatic index validation
)

print(f"Time elapsed after Step 4: {time.time() - start_time:.2f} seconds")

# Step 5: Write Documents to the Document Store with metadata
print("Writing documents to the FAISSDocumentStore...")
document_store.write_documents(documents)

print(f"Time elapsed after Step 5: {time.time() - start_time:.2f} seconds")

# Step 6: Update Vector IDs in the Document Store
print("Updating vector IDs in the FAISSDocumentStore...")
# Map internal IDs to document IDs
vector_id_map = {str(i): doc.id for i, doc in enumerate(documents)}

# Update the vector IDs in the document store
document_store.update_vector_ids(vector_id_map)

print(f"Time elapsed after Step 6: {time.time() - start_time:.2f} seconds")

# Step 7: Test the Retrieval Functionality
print("Testing retrieval from the FAISSDocumentStore...")
# Generate a random query embedding
query_embedding = np.random.rand(embedding_dim).astype(np.float32)

# Perform the query
retrieved_docs = document_store.query_by_embedding(query_embedding, top_k=2)

# Display the retrieved documents and similarity scores
for doc in retrieved_docs:
    print(f"Retrieved Document ID: {doc.id}")
    print(f"Title: {doc.meta['filename']}")  # Display the title metadata
    print(f"Content: {doc.content}")
    print(f"Similarity Score: {doc.score}")

print(f"Time elapsed after Step 7: {time.time() - start_time:.2f} seconds")
