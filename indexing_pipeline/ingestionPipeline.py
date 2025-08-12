import os
import time

# Import the main functions from your existing files
from parse_pdf import run_ingestion_pipeline as parse_pdf_to_json
from chunk_pdf import create_intelligent_chunks
from embedder import create_vector_store

# --- Configuration ---
# Define the PDF you want to process and the names for the intermediate and final files.
PDF_INPUT_PATH = "CHOTGDP23004V012223.pdf"
RAW_JSON_OUTPUT_PATH = "output.json"
CHUNKS_METADATA_PATH = "chunks_with_metadata.json"
FAISS_INDEX_PATH = "knowledge_base.index"
SOURCE_DOCUMENT_NAME = os.path.basename(PDF_INPUT_PATH)


def run_full_pipeline():
    """
    Orchestrates the entire ingestion pipeline from PDF to searchable vector index.
    """
    print("🚀 Starting the full ingestion pipeline...")
    start_time = time.time()

    # --- Step 1: Parse the PDF into structured JSON ---
    print("\n[Step 1/3] Parsing PDF into structured elements...")
    if not os.path.exists(PDF_INPUT_PATH):
        print(f"Error: Input PDF not found at '{PDF_INPUT_PATH}'. Aborting.")
        return
        
    structured_data = parse_pdf_to_json(PDF_INPUT_PATH)
    with open(RAW_JSON_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(structured_data, f, indent=2, ensure_ascii=False)
    print(f"Raw elements saved to '{RAW_JSON_OUTPUT_PATH}'")

    # --- Step 2: Create intelligent, context-aware chunks ---
    print("\n[Step 2/3] Performing intelligent chunking...")
    chunked_data = create_intelligent_chunks(structured_data, SOURCE_DOCUMENT_NAME)
    with open(CHUNKS_METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(chunked_data, f, indent=2, ensure_ascii=False)
    print(f"Intelligent chunks saved to '{CHUNKS_METADATA_PATH}'")

    # --- Step 3: Vectorize chunks and create the knowledge base ---
    print("\n[Step 3/3] Vectorizing chunks and building FAISS index...")
    create_vector_store()

    end_time = time.time()
    print("\n----------------------------------------------------")
    print("🎉🎉🎉 Full Ingestion Pipeline Completed! 🎉🎉🎉")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    print("Your knowledge base is now ready for querying.")
    print(f"Final files created: '{CHUNKS_METADATA_PATH}' and '{FAISS_INDEX_PATH}'")
    print("----------------------------------------------------")


if __name__ == '__main__':
    # Before running, we need to import the json library for the chunking step
    # This is a small adjustment to make the files work together seamlessly.
    import json
    
    run_full_pipeline()