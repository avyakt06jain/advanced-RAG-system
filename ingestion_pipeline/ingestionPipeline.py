import os
import gc
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import fitz # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- Configuration Constants ---
HEADER_FOOTER_MARGIN = 0.08
REPEATING_CONTENT_THRESHOLD = 0.5
BATCH_SIZE = 32

# --- Helper Functions (Internal to this module) ---

def _process_page_for_elements(args: tuple) -> tuple:
    """
    Worker function to process a single PDF page.
    Extracts text blocks and identifies potential header/footer content.
    """
    pdf_path, page_num = args
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    
    page_elements = []
    header_footer_candidates = []
    
    page_height = page.rect.height
    header_margin = page_height * HEADER_FOOTER_MARGIN
    footer_margin = page_height * (1 - HEADER_FOOTER_MARGIN)

    words = page.get_text("words")
    doc.close()

    if not words:
        return [], []

    words.sort(key=lambda w: (w[1], w[0]))

    paragraphs = []
    current_para = [words[0]]
    for i in range(1, len(words)):
        prev_word, curr_word = words[i-1], words[i]
        vertical_gap = curr_word[1] - prev_word[3]
        same_line = abs(curr_word[1] - prev_word[1]) < 2
        line_height = max(10, prev_word[3] - prev_word[1])
        
        if not same_line and vertical_gap > (line_height * 0.7):
            para_text = " ".join(w[4] for w in current_para)
            paragraphs.append({"content": para_text, "page_num": page_num})
            current_para = [curr_word]
        else:
            current_para.append(curr_word)
    
    if current_para:
        para_text = " ".join(w[4] for w in current_para)
        paragraphs.append({"content": para_text, "page_num": page_num})

    for para in paragraphs:
        para_text = para["content"].strip()
        if not para_text:
            continue
        
        # Heuristic: Short, centered text might be a header/footer
        # For this pipeline, we rely on the post-processing count
        header_footer_candidates.append(para_text)
        page_elements.append(para)
            
    return page_elements, header_footer_candidates


def _create_intelligent_chunks(elements: list, suppression_list: set, source_filename: str) -> list:
    """
    Applies cleaning, chunking, and metadata enrichment.
    """
    proto_chunks = []
    for element in elements:
        content = element["content"].strip()
        if content and content not in suppression_list:
            proto_chunks.append({
                "content": content,
                "page_num": element.get("page_num")
            })

    if not proto_chunks: return []

    merged_chunks = []
    list_item_pattern = re.compile(r'^\s*(?:[a-z][\.\)]|[ivx]+\.|[•])', re.IGNORECASE)
    merged_chunks.append(proto_chunks[0])
    for i in range(1, len(proto_chunks)):
        current_content = proto_chunks[i]["content"]
        if list_item_pattern.match(current_content):
            merged_chunks[-1]["content"] += f"\n{current_content}"
        else:
            merged_chunks.append(proto_chunks[i])

    final_chunks = []
    current_section_header = "Preamble"
    major_header_pattern = re.compile(r'^(?:[IVXLCDM]+\.|[A-Z]\)|\d+\.)\s+([A-Z\s\-&]+:?)$')
    for chunk in merged_chunks:
        content = chunk["content"]
        header_match = major_header_pattern.match(content.split('\n')[0])
        if header_match:
            current_section_header = header_match.group(1).strip().replace(':', '')
        
        final_chunks.append({
            "content": content,
            "metadata": {
                "source_document": source_filename,
                "page_number": chunk["page_num"],
                "section_header": current_section_header,
                "chunk_type": "content"
            }
        })
    return final_chunks


# --- Main Pipeline Function ---

def run_ingestion_pipeline(pdf_path: str, doc_hash: str, embedding_model, cache_dir: str):
    """
    The complete, refactored ingestion pipeline for use with app.py.

    Args:
        pdf_path (str): Path to the temporary downloaded PDF file.
        doc_hash (str): The unique hash of the document content.
        embedding_model: The pre-loaded sentence-transformer model.
        cache_dir (str): The directory to save the final index and chunks files.
    """
    print(f"--- Starting Ingestion Pipeline for doc hash: {doc_hash} ---")
    
    # 1. Parse PDF to structured elements
    print("Step 1/3: Parsing PDF...")
    doc = fitz.open(pdf_path)
    num_pages = doc.page_count
    doc.close()

    if num_pages == 0: return

    page_args = [(pdf_path, i) for i in range(num_pages)]
    all_page_elements_nested = [None] * num_pages
    all_hf_candidates = []

    with ThreadPoolExecutor() as executor:
        results = executor.map(_process_page_for_elements, page_args)
        for i, (page_elements, hf_candidates) in enumerate(tqdm(results, total=num_pages, desc="Parsing pages")):
            all_page_elements_nested[i] = page_elements
            all_hf_candidates.extend(hf_candidates)
    
    all_elements_flat = [item for sublist in all_page_elements_nested for item in sublist]

    # Calculate repeating headers to create suppression list
    header_footer_counts = defaultdict(int)
    for text in all_hf_candidates:
        header_footer_counts[text] += 1
    suppression_list = {text for text, count in header_footer_counts.items() 
                        if count / num_pages > REPEATING_CONTENT_THRESHOLD}

    # 2. Create intelligent chunks
    print("Step 2/3: Chunking content...")
    intelligent_chunks = _create_intelligent_chunks(all_elements_flat, suppression_list, os.path.basename(pdf_path))
    
    chunks_path = os.path.join(cache_dir, f"{doc_hash}.json")
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(intelligent_chunks, f, indent=2, ensure_ascii=False)
    
    # 3. Vectorize chunks and create FAISS index
    print("Step 3/3: Vectorizing and creating index...")
    texts_to_embed = [chunk['content'] for chunk in intelligent_chunks]

    all_embeddings = []
    for i in tqdm(range(0, len(texts_to_embed), BATCH_SIZE), desc="Embedding Batches"):
        batch = texts_to_embed[i:i + BATCH_SIZE]
        batch_embeddings = embedding_model.encode(batch, convert_to_numpy=True)
        all_embeddings.append(batch_embeddings)
        
    if not all_embeddings:
        print("Warning: No content was left after chunking to create a vector index.")
        return

    embeddings = np.vstack(all_embeddings)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    
    index_path = os.path.join(cache_dir, f"{doc_hash}.index")
    faiss.write_index(index, index_path)
    
    gc.collect()
    print(f"--- Ingestion Pipeline Finished. Files saved for hash {doc_hash} ---")