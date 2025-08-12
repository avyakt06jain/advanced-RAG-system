import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import time
from tqdm import tqdm

# --- Configuration ---
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNKS_FILE_PATH = "chunks_with_metadata.json"
INDEX_FILE_PATH = "knowledge_base.index"
BATCH_SIZE = 32  # <-- ADD THIS LINE: Process 32 chunks at a time

def create_vector_store():
    """
    Loads text chunks, generates embeddings in batches, and creates a searchable FAISS index.
    """
    print("Step 3: Vectorization and Embedding")
    
    # --- 1. Load the clean, structured chunks ---
    if not os.path.exists(CHUNKS_FILE_PATH):
        print(f"Error: Chunks file not found at '{CHUNKS_FILE_PATH}'")
        return

    print(f"Loading chunks from '{CHUNKS_FILE_PATH}'...")
    with open(CHUNKS_FILE_PATH, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    texts_to_embed = [chunk['content'] for chunk in chunks]
    print(f"Found {len(texts_to_embed)} text chunks to embed.")

    # --- 2. Initialize the Sentence Transformer model ---
    print(f"Loading embedding model: '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    # --- 3. Generate embeddings in batches to conserve memory ---
    print(f"Generating embeddings in batches of {BATCH_SIZE}...")
    start_time = time.time()
    
    all_embeddings = []
    # Use tqdm for a manual progress bar over the batches
    for i in tqdm(range(0, len(texts_to_embed), BATCH_SIZE), desc="Embedding Batches"):
        batch = texts_to_embed[i:i + BATCH_SIZE]
        batch_embeddings = model.encode(batch, convert_to_numpy=True)
        all_embeddings.append(batch_embeddings)
        
    # Combine the batch embeddings into a single numpy matrix
    embeddings = np.vstack(all_embeddings)
    
    end_time = time.time()
    print(f"Embeddings generated in {end_time - start_time:.2f} seconds.")
    print(f"Shape of embedding matrix: {embeddings.shape}")

    # --- 4. Create and store a FAISS index ---
    d = embeddings.shape[1]
    print("Building FAISS index...")
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    
    print(f"Index built successfully. Total vectors in index: {index.ntotal}")
    faiss.write_index(index, INDEX_FILE_PATH)
    print(f"FAISS index saved to '{INDEX_FILE_PATH}'")
    
if __name__ == '__main__':
    # Try to free up RAM before starting
    import gc
    gc.collect()

    create_vector_store()
    print("\n✅ Vectorization complete. Your knowledge base is ready.")
    print(f"You now have two essential files:")
    print(f"1. '{CHUNKS_FILE_PATH}' - Your content and metadata store.")
    print(f"2. '{INDEX_FILE_PATH}' - Your searchable vector index.")