import fitz  # PyMuPDF
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import gc

# --- Configuration Constants ---
# (Constants remain the same)
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
        paragraphs.append({"text": para_text, "bbox": para_bbox.irect})

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
                "type": "text", "page_num": page_num, "content": para_text, "bbox": list(para_bbox)
            })
            
    doc.close()
    return sorted(page_elements, key=lambda x: x['bbox'][1]), header_footer_candidates


def run_pdf_parser(pdf_path: str) -> list:
    """
    OPTIMIZED Main Orchestrator: Runs the single-pass parallel pipeline.
    """
    
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
                final_elements.append(element)

    gc.collect()
    
    print(final_elements)

    return json.dumps(final_elements, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    parsed_data = run_pdf_parser('CHOTGDP23004V012223.pdf')
    print(parsed_data)
    print(len(parsed_data))
