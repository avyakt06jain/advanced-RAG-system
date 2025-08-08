from keywords import extract_keywords
from docSearch import load_index_and_documents, semantic_search
from llmoutputretrival import create_llm_prompt, generate_answer_from_llm


def run_rag_pipeline(user_prompt):
    print("--- Starting RAG Pipeline ---")
    print(f"User Prompt: '{user_prompt}'")
    print("-" * 30)

    # --- Step 1: Parse the prompt and extract keywords ---
    print("Step 1: Extracting keywords...")
    query_keywords = extract_keywords(user_prompt)
    print(f"Extracted Keywords: {query_keywords}")
    print("-" * 30)

    # --- Step 2: Load the FAISS index and documents ---
    print("Step 2: Loading FAISS index and documents...")
    index_file_path = "faiss_index.faiss"
    docs_file_path = "documents.pkl"
    faiss_index, documents = load_index_and_documents(index_file_path, docs_file_path)

    if not faiss_index or not documents:
        print("Aborting pipeline due to missing files.")
        return

    print("-" * 30)

    # --- Step 3: Perform the semantic search ---
    print("Step 3: Performing semantic search...")
    top_k = 3
    relevant_chunks = semantic_search(
        query_keywords, faiss_index, documents, top_k=top_k
    )

    if not relevant_chunks:
        print("No relevant chunks found. Aborting pipeline.")
        return

    print("Retrieved Relevant Chunks for Answer Generation:")
    for i, chunk in enumerate(relevant_chunks):
        print(f"\nChunk {i + 1}:")
        print(chunk)
    print("-" * 30)

    # --- Step 4: Create the final LLM prompt ---
    print("Step 4: Creating LLM prompt...")
    llm_prompt = create_llm_prompt(user_prompt, relevant_chunks)
    print("--- Final Prompt for LLM ---")
    print(llm_prompt)
    print("-" * 30)

    # --- Step 5: Generate the final answer from the LLM ---
    print("Step 5: Generating final answer...")
    final_answer = generate_answer_from_llm(llm_prompt)

    print("\n--- RAG Pipeline Complete ---")
    print("\nFinal Generated Answer:")
    print(final_answer)
    print("-" * 30)


# Main execution block
if __name__ == "__main__":
    # Define a sample user query to test the entire pipeline
    query = "QueryPlaceHolder"
    run_rag_pipeline(query)
