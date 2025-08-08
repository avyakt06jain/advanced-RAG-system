from keywords import extract_keywords
from docSearch import load_index_and_documents, semantic_search


def main():
    # This is the input from the user that our system needs to process.
    user_prompt = "What are policy conditions for pre-disabled people."
    print(f"User Prompt: '{user_prompt}'")
    print("-" * 30)

    # Keywords
    query_keywords = extract_keywords(user_prompt)
    print(f"Extracted Keywords: {query_keywords}")
    print("-" * 30)

    # Define the file paths for your pre-built FAISS index and document chunks.
    index_file_path = "../faiss_index/index.faiss"
    docs_file_path = "../faiss_index/index.pkl"
    faiss_index, documents = load_index_and_documents(index_file_path, docs_file_path)

    # Check if the files were loaded successfully before proceeding.
    if faiss_index and documents:
        top_k = 3
        relevant_chunks = semantic_search(
            query_keywords, faiss_index, documents, top_k=top_k
        )

        print("\n--- Retrieved Relevant Chunks for Answer Generation ---")
        for i, chunk in enumerate(relevant_chunks):
            print(f"\nChunk {i + 1}:")
            print(chunk)
            print("-" * 30)
    else:
        print(
            "Could not proceed with semantic search. Please ensure the required files exist."
        )


# Run the main function when the script is executed.
if __name__ == "__main__":
    main()
