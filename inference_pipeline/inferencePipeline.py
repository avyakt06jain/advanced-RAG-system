import os
import json
import faiss
import numpy as np
import google.generativeai as genai
from typing import List, Dict

# This file contains the logic for the inference pipeline.
# It's designed to be called by a main application like app.py.

def _semantic_search(query: str, embedding_model, faiss_index, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Performs semantic search to find the most relevant chunks.
    """
    print(f"Performing semantic search for query: '{query[:50]}...'")
    query_vector = embedding_model.encode([query], convert_to_numpy=True)
    
    # Search the FAISS index
    distances, indices = faiss_index.search(query_vector, k=top_k)
    
    # Retrieve the actual chunks and their metadata
    retrieved_chunks = [chunks[i] for i in indices[0]]
    
    # You could also add the score to each chunk if needed
    # for i, chunk in enumerate(retrieved_chunks):
    #     chunk['score'] = distances[0][i]
        
    return retrieved_chunks

def _generate_answer_with_gemini(query: str, context_chunks: List[Dict], generative_model) -> str:
    """
    Generates an answer using the Gemini model based on the provided context.
    """
    if not context_chunks:
        return "I do not have enough information to answer this question based on the provided document."

    context = "\n\n---\n\n".join([chunk['content'] for chunk in context_chunks])
    
    prompt = f"""
    **INSTRUCTION:**
    You are an expert Q&A assistant for documents.
    Your task is to answer the user's question based *only* on the provided context.
    Do not use any external knowledge. If the answer is not found in the context, state that clearly.
    Provide a direct and concise answer.

    **CONTEXT:**
    {context}

    **QUESTION:**
    {query}

    **ANSWER:**
    """
    
    try:
        response = generative_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return "Sorry, an error occurred while generating the answer with the language model."

# --- Main Pipeline Function ---

def run_inference_pipeline(
    queries: List[str],
    doc_hash: str,
    cache_dir: str,
    embedding_model,
    generative_model,
    loaded_indexes: Dict,
    loaded_chunks: Dict
) -> List[str]:
    """
    The complete inference pipeline for a list of queries.

    Args:
        queries (List[str]): List of user questions.
        doc_hash (str): The hash of the document to query against.
        cache_dir (str): The directory where knowledge base files are stored.
        embedding_model: The pre-loaded sentence-transformer model.
        generative_model: The pre-loaded Gemini model.
        loaded_indexes (Dict): A dictionary to cache loaded FAISS indexes.
        loaded_chunks (Dict): A dictionary to cache loaded chunk data.

    Returns:
        List[str]: A list of answers corresponding to the input queries.
    """
    index_path = os.path.join(cache_dir, f"{doc_hash}.index")
    chunks_path = os.path.join(cache_dir, f"{doc_hash}.json")

    # --- Step 1: Load Knowledge Base (if not already in memory) ---
    if doc_hash not in loaded_indexes:
        print(f"Loading knowledge base for '{doc_hash}' into memory...")
        try:
            loaded_indexes[doc_hash] = faiss.read_index(index_path)
            with open(chunks_path, 'r', encoding='utf-8') as f:
                loaded_chunks[doc_hash] = json.load(f)
        except FileNotFoundError:
            return [f"Error: Knowledge base for document hash {doc_hash} not found." for _ in queries]

    faiss_index = loaded_indexes[doc_hash]
    chunks_data = loaded_chunks[doc_hash]
    
    final_answers = []
    for query in queries:
        # --- Step 2: Hybrid Search (Semantic + Keyword) ---
        # For now, we are only implementing semantic search as the primary retriever.
        # TODO: Add your keyword search logic here (e.g., using rank_bm25).
        retrieved_chunks = _semantic_search(query, embedding_model, faiss_index, chunks_data)
        
        # --- Step 3: Rerank Results ---
        # TODO: Add your reranking logic here (e.g., using a cross-encoder).
        # For now, the top results from semantic search are used directly.
        reranked_chunks = retrieved_chunks 
        
        # --- Step 4: Generate Answer ---
        answer = _generate_answer_with_gemini(query, reranked_chunks, generative_model)
        final_answers.append(answer)
        
    return final_answers