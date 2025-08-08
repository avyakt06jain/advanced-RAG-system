# semantic_searcher.py

# To run this script, you need to install the following libraries:
# pip install faiss-cpu sentence-transformers numpy pickle

import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import os

# Define the name of the embedding model we will use.
# It is CRITICAL that this is the same model used to create the index.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_index_and_documents(index_path, docs_path):
    """
    Loads the FAISS index and the corresponding document chunks from disk.

    Args:
        index_path (str): The file path to the saved FAISS index.
        docs_path (str): The file path to the saved list of document chunks.

    Returns:
        tuple: A tuple containing the loaded FAISS index and the documents.
    """
    try:
        # Load the FAISS index
        faiss_index = faiss.read_index(index_path)
        print(f"FAISS index loaded successfully from {index_path}")

        # Load the corresponding list of document chunks
        with open(docs_path, "rb") as f:
            documents = pickle.load(f)
        print(f"Documents loaded successfully from {docs_path}")

        return faiss_index, documents
    except FileNotFoundError as e:
        print(f"Error: Required file not found. {e}")
        return None, None


def semantic_search(query_keywords, faiss_index, documents, top_k=3):
    """
    Performs a semantic search on the FAISS index and returns relevant document chunks.

    Args:
        query_keywords (list or str): The keywords or full query from the user.
        faiss_index (faiss.Index): The loaded FAISS index.
        documents (tuple or list): The list of text chunks corresponding to the index.
                                   This function is now robust to a tuple wrapper.
        top_k (int): The number of top-k most similar chunks to retrieve.

    Returns:
        list: A list of the most relevant document chunks.
    """
    if not faiss_index or documents is None:
        print("Index or documents not loaded. Aborting search.")
        return []

    # Initialize the same embedding model used for indexing.
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded.")

    # Convert the query keywords into a single string for embedding.
    if isinstance(query_keywords, list):
        query_string = " ".join(query_keywords)
    else:
        query_string = query_keywords

    # Generate a vector embedding for the search query.
    query_vector = model.encode([query_string])
    query_vector = np.array(query_vector).astype("float32")

    print(f"Searching for top {top_k} similar documents...")
    distances, indices = faiss_index.search(query_vector, k=top_k)
    print("Search completed.")

    # --- FIX: Ensure 'documents' is a list before indexing. ---
    # This check handles the case where 'documents.pkl' contains a tuple wrapper.
    if isinstance(documents, tuple):
        # We assume the list of documents is the first item in the tuple,
        # and that the tuple is not empty.
        if len(documents) > 0:
            doc_list = documents[0]
        else:
            print("Error: Loaded documents tuple is empty. Cannot retrieve chunks.")
            return []
    else:
        # If it's already a list or other iterable, use it directly.
        doc_list = documents

    # Use the indices to retrieve the original document chunks.
    retrieved_chunks = [doc_list[i] for i in indices[0]]

    return retrieved_chunks


# This is the main execution block of the script.
if __name__ == "__main__":
    index_file_path = "faiss_index.faiss"
    docs_file_path = "documents.pkl"

    # Load the pre-built index and documents.
    index, docs = load_index_and_documents(index_file_path, docs_file_path)

    if index and docs:
        # Example keywords from your previous script.
        user_keywords = ["health", "benefits", "green tea"]

        # Perform the semantic search.
        relevant_chunks = semantic_search(user_keywords, index, docs, top_k=2)

        print("\n--- Retrieved Relevant Chunks ---")
        for i, chunk in enumerate(relevant_chunks):
            print(f"\nChunk {i + 1}:")
            print(chunk)
    else:
        print("\nPlease generate the FAISS index and documents files first.")
