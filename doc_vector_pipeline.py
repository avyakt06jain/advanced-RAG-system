"""
pdf_to_faiss_class.py

Full pipeline class to parse PDF into FAISS vector store:
- extracts tables and text from each page
- parallel parsing (ProcessPoolExecutor) WITHOUT passing embedding model to workers
- main-process embedding using HuggingFaceEmbeddings (default all-MiniLM-L6-v2)
- persistent FAISS vector store (load or create)
- persistent deduplication via SHA256 hashes stored in dedup.json
- retrieval method (similarity search with score threshold)
- merges headings + subsequent paragraphs into single chunks for better RAG quality

Usage:
    from pdf_to_faiss_class import DocVectorPipeline
    pipeline = DocVectorPipeline(persist_dir="faiss_index")
    pipeline.process_pdf("mydoc.pdf")
    pipeline.save()
    results = pipeline.retrieve("What is privacy policy for user data?", k=5, score_threshold=0.2)
"""

import os
import gc
import json
import time
import hashlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple

import fitz
import camelot
from tqdm import tqdm

from langchain_core.documents import Document

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def classify_pages(pdf_path: str) -> List[int]:
    """Detect pages that look like tables (multi-column). Returns list of page numbers (0-indexed)."""
    doc = fitz.open(pdf_path)
    table_pages = []
    TOLERANCE = 5
    MIN_COLUMNS = 2
    MIN_ITEMS_PER_COLUMN = 3

    for page in doc:
        column_counts = defaultdict(int)
        blocks = page.get_text("blocks")
        for block in blocks:
            if len(block) >= 7 and block[6] == 0: 
                x0 = round(block[0] / TOLERANCE) * TOLERANCE
                column_counts[x0] += 1

        valid_columns = sum(1 for count in column_counts.values() if count >= MIN_ITEMS_PER_COLUMN)
        if valid_columns >= MIN_COLUMNS:
            table_pages.append(page.number)

    doc.close()
    return table_pages


def extract_table_from_page(pdf_path: str, page_num: int) -> List[Dict[str, Any]]:
    """Return list of table element dicts (type='table') for the page."""
    table_elements = []
    try:
        # camelot expects 1-indexed pages
        tables = camelot.read_pdf(pdf_path, pages=str(page_num + 1), flavor='lattice')
        for table in tables:
            markdown_table = table.df.to_markdown(index=False)
            table_element = {
                "type": "table",
                "content": markdown_table,
                "page_num": page_num,
                "bbox": table._bbox
            }
            table_elements.append(table_element)
    except Exception as e:
        print(f"Warning: Camelot failed on page {page_num + 1}. Error: {e}")
    return table_elements


def extract_text_from_page(pdf_path: str, page_num: int, table_bboxes: List[Any]) -> List[Dict[str, Any]]:
    """Return list of text element dicts for a page, with type=heading|paragraph based on bold detection."""
    def is_bbox_inside_any(inner_bbox, outer_bboxes):
        return any(
            inner_bbox[0] >= ob[0] and inner_bbox[1] >= ob[1] and
            inner_bbox[2] <= ob[2] and inner_bbox[3] <= ob[3]
            for ob in outer_bboxes
        )

    text_elements = []
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    # header/ footer heuristics to skip repeated lines
    header_margin = page.rect.height * 0.08
    footer_margin = page.rect.height * (1 - 0.08)

    blocks = page.get_text("dict")["blocks"]
    for block in blocks:
        if block.get("type") == 0 and "lines" in block:
            block_bbox = block.get("bbox", None)
            if not block_bbox:
                continue
            # Skip if overlaps a detected table region
            if table_bboxes and is_bbox_inside_any(block_bbox, table_bboxes):
                continue
            # Skip header/footer repeated lines
            if block_bbox[1] < header_margin or block_bbox[3] > footer_margin:
                continue

            block_text = ""
            is_bold = False
            for line in block["lines"]:
                for span in line["spans"]:
                    span_text = span.get("text", "")
                    block_text += span_text + " "
                    fontname = span.get("font", "")
                    if isinstance(fontname, str) and "bold" in fontname.lower():
                        is_bold = True

            cleaned = block_text.strip()
            if cleaned:
                text_elements.append({
                    "type": "heading" if is_bold else "paragraph",
                    "content": cleaned,
                    "page_num": page_num,
                    "bbox": block_bbox
                })

    doc.close()
    return text_elements


def merge_headings_with_paragraphs(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Given a list of elements for a single page (ordered top->bottom),
    merge a heading with subsequent paragraphs until the next heading/table.
    Returns a list of merged elements (dicts).
    """
    merged = []
    i = 0
    n = len(elements)
    while i < n:
        el = elements[i]
        if el["type"] == "heading":
            # start a chunk containing heading + following paragraphs
            chunk_text = el["content"]
            chunk_bbox = el.get("bbox")
            j = i + 1
            while j < n and elements[j]["type"] == "paragraph":
                chunk_text += "\n\n" + elements[j]["content"]
                j += 1
            merged.append({
                "type": "heading_paragraph",
                "content": chunk_text,
                "page_num": el["page_num"],
                "bbox": chunk_bbox
            })
            i = j
        else:
            # paragraph or table remains its own chunk
            merged.append(el)
            i += 1
    return merged


def sha256_hash_text(text: str) -> str:
    """Return SHA256 hex digest for given text (normalized)."""
    t = text.strip()
    # Normalize whitespace slightly
    t = " ".join(t.split())
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def process_page_worker(pdf_path: str, page_num: int, table_pages: List[int]) -> List[Dict[str, Any]]:
    """
    Parse page into a list of serializable element dicts (not Document objects).
    Return a list of dicts representing table/heading/paragraph elements.
    """
    try:
        table_elements = []
        table_bboxes = []
        if page_num in table_pages:
            table_elements = extract_table_from_page(pdf_path, page_num)
            table_bboxes = [el["bbox"] for el in table_elements]

        text_elements = extract_text_from_page(pdf_path, page_num, table_bboxes)
        all_elements = table_elements + text_elements
        # sort by vertical position (y0) so order is top->bottom
        all_elements.sort(key=lambda el: el["bbox"][1] if el.get("bbox") else 0)
        return all_elements
    except Exception as e:
        print(f"Error in process_page_worker (page {page_num + 1}): {e}")
        return []



# --- Configuration Constants ---
HEADER_FOOTER_MARGIN = 0.08
REPEATING_CONTENT_THRESHOLD = 0.5
TABLE_DETECTION_TOLERANCE = 5
MIN_COLUMNS_FOR_TABLE = 2
MIN_ITEMS_PER_COLUMN = 3
SPANNING_HEADER_FACTOR = 1.8
CAPTION_SEARCH_TOLERANCE = 15

def process_page_single_pass(args: tuple) -> tuple:
    """
    OPTIMIZED WORKER: Processes a single page in one pass.
    
    Returns a tuple containing:
    1. A list of all extracted page elements (text/tables).
    2. A list of potential header/footer texts found on this page.
    """
    pdf_path, page_num = args
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    
    page_elements = []
    header_footer_candidates = []
    
    # --- Part 1: PRECISE TEXT EXTRACTION & HEADER/FOOTER IDENTIFICATION ---
    page_height = page.rect.height
    header_margin = page_height * HEADER_FOOTER_MARGIN
    footer_margin = page_height * (1 - HEADER_FOOTER_MARGIN)

    words = page.get_text("words")
    if not words:
        doc.close()
        return [], []

    words.sort(key=lambda w: (w[1], w[0]))

    paragraphs = []
    current_para = [words[0]]
    for i in range(1, len(words)):
        prev_word, curr_word = words[i-1], words[i]
        vertical_gap = curr_word[1] - prev_word[3]
        same_line = abs(curr_word[1] - prev_word[1]) < 2
        line_height = max(10, prev_word[3] - prev_word[1]) # Use max to avoid division by zero for small elements
        
        if not same_line and vertical_gap > (line_height * 0.7):
            para_text = " ".join(w[4] for w in current_para)
            para_bbox = fitz.Rect(current_para[0][:4])
            for w in current_para[1:]:
                para_bbox.include_rect(w[:4])
            paragraphs.append({"text": para_text, "bbox": para_bbox.irect})
            current_para = [curr_word]
        else:
            current_para.append(curr_word)
    
    if current_para:
        para_text = " ".join(w[4] for w in current_para)
        para_bbox = fitz.Rect(current_para[0][:4])
        for w in current_para[1:]:
            para_bbox.include_rect(w[:4])
        paragraphs.append({"text": para_text, "bbox": para_bbox.irect, "source": pdf_path})

    # Now, categorize paragraphs as main content or potential headers/footers
    for para in paragraphs:
        para_text = para["text"].strip()
        if not para_text:
            continue
            
        para_bbox = para["bbox"]
        is_header_footer = para_bbox.y0 < header_margin or para_bbox.y1 > footer_margin
        
        if is_header_footer:
            header_footer_candidates.append(para_text)
        else:
            page_elements.append({
                "type": "text", "page_num": page_num, "content": para_text, "bbox": list(para_bbox), "source": pdf_path
            })
            
    doc.close()
    return sorted(page_elements, key=lambda x: x['bbox'][1]), header_footer_candidates



class DocVectorPipeline:
    def __init__(
        self,
        persist_dir: str = "faiss_index",
        dedup_path: str = "dedup.json",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 256,
        max_workers: Optional[int] = None
    ):
        """
        Initialize pipeline: loads or creates FAISS vectorstore and loads dedup metadata.
        - persist_dir: directory to save/load the FAISS data
        - dedup_path: JSON file path for deduplication hashes
        - embedding_model_name: HuggingFace embeddings model name
        - batch_size: number of docs per embedding shard
        - max_workers: number of parse workers for ProcessPoolExecutor
        """
        self.persist_dir = persist_dir
        self.dedup_path = dedup_path
        self.embedding_model_name = embedding_model_name
        self.batch_size = batch_size
        self.max_workers = max_workers


        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

        self.faiss: Optional[FAISS] = None
        if os.path.exists(self.persist_dir) and os.path.isdir(self.persist_dir):
            try:
                print(f"Loading existing FAISS index from {self.persist_dir} ...")
                self.faiss = FAISS.load_local(self.persist_dir, self.embedding_model, allow_dangerous_deserialization=True)
                print("FAISS index loaded.")
            except Exception as e:
                print(f"Warning: failed to load FAISS from {self.persist_dir}: {e}")
                self.faiss = None

        if self.faiss is None:
            print("Creating empty FAISS index (no documents yet).")
            self.faiss = None

        # Load dedup JSON (a set of SHA256 hashes)
        self.seen_hashes = set()
        if os.path.exists(self.dedup_path):
            try:
                with open(self.dedup_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.seen_hashes = set(data)
                    elif isinstance(data, dict) and "hashes" in data:
                        self.seen_hashes = set(data["hashes"])
                print(f"Loaded dedup metadata with {len(self.seen_hashes)} entries from {self.dedup_path}")
            except Exception as e:
                print(f"Warning: failed to load dedup metadata: {e}")
                self.seen_hashes = set()
        else:
            # If FAISS exists but dedup json doesn't, populate from FAISS
            if self.faiss is not None:
                try:
                    existing_docs = list(self.faiss._get_docs().values()) if hasattr(self.faiss, "_get_docs") else []

                    if not existing_docs and hasattr(self.faiss, "docstore") and hasattr(self.faiss.docstore, "_dict"):
                        existing_docs = list(self.faiss.docstore._dict.values())
                    for d in existing_docs:
                        text = getattr(d, "page_content", "") if d else ""
                        if text:
                            self.seen_hashes.add(sha256_hash_text(text))
                    print(f"Initialized dedup metadata from FAISS contents: {len(self.seen_hashes)} hashes.")
                except Exception:
                    pass

        self.total_docs = 0 if self.faiss is None else getattr(self.faiss, "index", None) and len(self.faiss.index.reconstruct_n(0, 0)) or 0

    # def parse_pdf_to_documents(self, pdf_path: str) -> List[Document]:
        # """
        # Parses the PDF using worker processes (parsing only), merges headings with paragraphs,
        # and returns a list of langchain_core.documents.Document objects ready for embedding.
        # """
        # if not os.path.exists(pdf_path):
        #     raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # print("Opening PDF to get page count...")
        # doc = fitz.open(pdf_path)
        # num_pages = len(doc)
        # doc.close()

        # print("Classifying pages for table detection...")
        # table_pages = classify_pages(pdf_path)
        # print(f"Detected table pages: {table_pages}")

        # parsed_elements_by_page: Dict[int, List[Dict[str, Any]]] = {}

        # print(f"Parsing {num_pages} pages in parallel (parsing only)...")
        # with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
        #     futures = {
        #         executor.submit(process_page_worker, pdf_path, pnum, table_pages): pnum
        #         for pnum in range(num_pages)
        #     }
        #     for future in tqdm(as_completed(futures), total=len(futures)):
        #         pnum = futures[future]
        #         try:
        #             elems = future.result()
        #             parsed_elements_by_page[pnum] = elems
        #         except Exception as e:
        #             print(f"Error parsing page {pnum + 1}: {e}")
        #             parsed_elements_by_page[pnum] = []

        # # Convert parsed elements into Document objects (after merging headings+paragraphs)
        # all_documents: List[Document] = []
        # for pnum in range(num_pages):
        #     elems = parsed_elements_by_page.get(pnum, [])
        #     if not elems:
        #         continue
        #     merged = merge_headings_with_paragraphs(elems)
        #     for el in merged:
        #         metadata = {
        #             "source": os.path.basename(pdf_path),
        #             "page_number": el["page_num"] + 1,
        #             "type": el["type"],
        #             "bounding_box": el.get("bbox")
        #         }
        #         content = el.get("content", "")

        #         if isinstance(content, bytes):
        #             content = content.decode("utf-8", errors="ignore")
        #         doc_obj = Document(page_content=content, metadata=metadata)
        #         all_documents.append(doc_obj)

        # print(f"Total Documents created from PDF: {len(all_documents)}")
        # return all_documents


    def parse_pdf_to_documents(self, pdf_path: str) -> List[Document]:
        doc = fitz.open(pdf_path)
        num_pages = doc.page_count
        doc.close()
        
        if num_pages == 0:
            return []
            
        page_args = [(pdf_path, i) for i in range(num_pages)]
        
        all_page_elements = [None] * num_pages
        all_header_footer_candidates = []
        
        with ThreadPoolExecutor() as executor:
            results = executor.map(process_page_single_pass, page_args)
            for i, (page_elements, hf_candidates) in enumerate(results):
                all_page_elements[i] = page_elements
                all_header_footer_candidates.extend(hf_candidates)

        # --- Post-processing: Calculate suppression list and filter ---
        header_footer_counts = defaultdict(int)
        for text in all_header_footer_candidates:
            header_footer_counts[text] += 1
            
        suppression_list = {
            text for text, count in header_footer_counts.items()
            if count / num_pages > REPEATING_CONTENT_THRESHOLD
        }
        
        # Final filtering pass (very fast)
        final_elements = []
        for page_elements in all_page_elements:
            for element in page_elements:
                if element["content"] not in suppression_list:
                    del element["bbox"]  # Clean up bbox as it's not needed
                    doc = Document(page_content=element["content"], metadata={
                        "page_number": element["page_num"] + 1,
                        "source": os.path.basename(pdf_path),
                        "type": element["type"],
                    })
                    final_elements.append(doc)

        gc.collect()

        # return json.dumps(final_elements, indent=2, ensure_ascii=False)
        return final_elements


    def add_documents_with_dedup(self, documents: List[Document], save_after_add: bool = True) -> Tuple[int, int]:
        """
        Adds documents to FAISS after deduplication.
        Returns (num_added, num_skipped)
        - Deduplication is done by computing SHA256 over normalized content.
        - Persist dedup metadata to dedup_path if save_after_add True.
        """
        if not documents:
            return 0, 0

        unique_docs = []
        skipped = 0
        for doc in documents:
            text = (doc.page_content or "")
            if not text.strip():
                skipped += 1
                continue
            h = sha256_hash_text(text)
            if h in self.seen_hashes:
                skipped += 1
                continue
            self.seen_hashes.add(h)
            unique_docs.append(doc)

        if not unique_docs:
            print("No new unique documents to add after deduplication.")
            return 0, skipped

        shards = []
        for batch in chunk_list(unique_docs, self.batch_size):
            shard = FAISS.from_documents(batch, self.embedding_model)
            shards.append(shard)

        # Merge with existing FAISS
        if self.faiss is None and shards:
            self.faiss = shards[0]
            rest = shards[1:]
        else:
            rest = shards

        for shard in rest:
            try:
                self.faiss.merge_from(shard)
            except Exception as e:
                print(f"Warning: merge failed: {e}. Attempting to add documents directly.")
                try:
                    docs_to_add = [d for d in unique_docs]  # fallback: add all unique docs
                    self.faiss.add_documents(docs_to_add)
                    break
                except Exception as e2:
                    print(f"Failed to add documents in fallback: {e2}")
                    raise

        num_added = len(unique_docs)
        self.total_docs += num_added

        if save_after_add:
            self.save()

        print(f"Added {num_added} new documents to FAISS. Skipped {skipped} duplicates/empties.")
        return num_added, skipped


    def build_from_pdf(self, pdf_path: str):
        """
        High-level convenience: parse the PDF and add docs with deduplication.
        """
        docs = self.parse_pdf_to_documents(pdf_path)
        return self.add_documents_with_dedup(docs)

    # def retrieve(self, query: str, k: int = 5, score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Retrieve top-k results for the given query.
        Returns a list of dicts: { 'page_content', 'metadata', 'score' }
        - score_threshold: if provided, filters out results with score > threshold (depending on FAISS scoring sign)
        Note: langchain FAISS returns (doc, score) with distance; smaller distance -> more similar for many metrics.
        """
        if self.faiss is None:
            print("FAISS index is empty. No retrieval possible.")
            return []

        # Use FAISS similarity search with scores
        try:
            results = self.faiss.similarity_search_with_score(query, k=k)
        except Exception as e:
            print(f"Error during similarity search: {e}")
            return []

        formatted = []
        for doc, score in results:
            entry = {
                "page_content": getattr(doc, "page_content", None),
                "metadata": getattr(doc, "metadata", None),
                "score": score
            }
            formatted.append(entry)

        if score_threshold is not None:
            filtered = [f for f in formatted if f["score"] <= score_threshold]
            if not filtered:
                return formatted
            return filtered

        return formatted

    def retrieve(
        self,
        query: str,
        source_pdf: str,
        k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k results for the given query, restricted to the same PDF.
        - source_pdf: PDF filename or unique source identifier to match in metadata['source'].
        - score_threshold: if provided, filters out results with score > threshold 
        (depending on FAISS scoring sign; smaller is better for cosine/L2).
        Returns: List[dict] with keys: page_content, metadata, score.
        """
        if self.faiss is None:
            print("FAISS index is empty. No retrieval possible.")
            return []

        try:
            results = self.faiss.similarity_search_with_score(query, k=20)  # get more for filtering
        except Exception as e:
            print(f"Error during similarity search: {e}")
            return []

        # Filter by the same PDF source
        filtered_by_source = [
            (doc, score) for doc, score in results
            if doc.metadata.get("source") == source_pdf
        ]

        # Apply score threshold if given
        if score_threshold is not None:
            filtered_by_source = [
                (doc, score) for doc, score in filtered_by_source
                if score <= score_threshold
            ]

        # Limit to top-k
        filtered_by_source = filtered_by_source[:k]

        # Format output
        return [
            {
                "page_content": getattr(doc, "page_content", None),
                "metadata": getattr(doc, "metadata", None),
                "score": score
            }
            for doc, score in filtered_by_source
        ]


    def save(self):
        """
        Save FAISS index and dedup metadata (hashes).
        """
        # Save dedup json
        try:
            tmp = list(self.seen_hashes)
            with open(self.dedup_path, "w", encoding="utf-8") as f:
                json.dump({"hashes": tmp}, f, indent=2)
            print(f"Saved dedup metadata to {self.dedup_path} ({len(tmp)} hashes).")
        except Exception as e:
            print(f"Warning: failed to save dedup metadata: {e}")

        # Save FAISS
        if self.faiss is not None:
            try:
                os.makedirs(self.persist_dir, exist_ok=True)
                self.faiss.save_local(self.persist_dir)
                print(f"Saved FAISS index to {self.persist_dir}")
            except Exception as e:
                print(f"Warning: failed to save FAISS index: {e}")
        else:
            print("No FAISS index to save.")

    def save_parsed_documents(self, documents: List[Document], path: str = "parsed_documents.json"):
        '''
        Save parsed documents to a JSON file.
        '''
        try:
            serial = []
            for d in documents:
                serial.append({
                    "page_content": d.page_content,
                    "metadata": d.metadata
                })
            with open(path, "w", encoding="utf-8") as f:
                json.dump(serial, f, indent=2, ensure_ascii=False)
            print(f"Saved parsed documents to {path}")
        except Exception as e:
            print(f"Warning: could not save parsed documents: {e}")


def chunk_list(lst: List[Any], chunk_size: int):
    '''
        chunk list generator
    '''
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


if __name__ == "__main__":
    PDF_FILE = "policy.pdf"
    # PDF_FILE = "CHOTGDP23004V012223.pdf"
    PERSIST_DIR = "faiss_index"
    DEDUP_PATH = "dedup.json"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    BATCH_SIZE = 256
    MAX_WORKERS = None

    pipeline = DocVectorPipeline(
        persist_dir=PERSIST_DIR,
        dedup_path=DEDUP_PATH,
        embedding_model_name=EMBEDDING_MODEL,
        batch_size=BATCH_SIZE,
        max_workers=MAX_WORKERS
    )

    t0 = time.time()
    added, skipped = pipeline.build_from_pdf(PDF_FILE)
    t1 = time.time()
    print(f"Pipeline finished: added {added} docs, skipped {skipped} duplicates. Time: {t1 - t0:.2f}s")

    pipeline.save()

    # Example retrieval
    query = "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?"
    results = pipeline.retrieve(query, PDF_FILE, k=5, score_threshold=None)
    print("Top retrieval results:")
    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} (score={r['score']}) ---")
        print("Metadata:", r["metadata"])
        snippet = (r["page_content"] or "")[:800].replace("\n", " ")
        print(snippet)
