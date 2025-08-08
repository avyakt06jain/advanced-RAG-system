import fitz
import json
import camelot
from collections import defaultdict
from langchain_core.documents import Document
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time
import os

def classify_pages(pdf_path: str):
    doc = fitz.open(pdf_path)
    table_pages = []
    TOLERANCE = 5
    MIN_COLUMNS = 2
    MIN_ITEMS_PER_COLUMN = 3

    for page in doc:
        column_counts = defaultdict(int)
        blocks = page.get_text("blocks")

        for block in blocks:
            if block[6] == 0:
                x0 = round(block[0] / TOLERANCE) * TOLERANCE
                column_counts[x0] += 1

        valid_columns = sum(1 for count in column_counts.values() if count >= MIN_ITEMS_PER_COLUMN)
        if valid_columns >= MIN_COLUMNS:
            table_pages.append(page.number)

    return table_pages

def extract_table_from_page(pdf_path: str, page_num: int):
    table_elements = []
    try:
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

def extract_text_from_page(pdf_path: str, page_num: int, table_bboxes: list):
    def is_bbox_inside_any(inner_bbox, outer_bboxes):
        return any(
            inner_bbox[0] >= ob[0] and inner_bbox[1] >= ob[1] and
            inner_bbox[2] <= ob[2] and inner_bbox[3] <= ob[3]
            for ob in outer_bboxes
        )

    text_elements = []
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    header_margin = page.rect.height * 0.08
    footer_margin = page.rect.height * (1 - 0.08)

    text_blocks = page.get_text("dict")["blocks"]
    for block in text_blocks:
        if block.get("type") == 0 and "lines" in block:
            block_bbox = block["bbox"]
            if is_bbox_inside_any(block_bbox, table_bboxes):
                continue
            if block_bbox[1] < header_margin or block_bbox[3] > footer_margin:
                continue

            block_text = ""
            is_bold = False
            for line in block["lines"]:
                for span in line["spans"]:
                    block_text += span["text"] + " "
                    if "bold" in span["font"].lower():
                        is_bold = True

            cleaned_text = block_text.strip()
            if cleaned_text:
                text_elements.append({
                    "type": "heading" if is_bold else "paragraph",
                    "content": cleaned_text,
                    "page_num": page_num,
                    "bbox": block_bbox
                })

    return text_elements

def process_page(pdf_path, page_num, table_pages):
    table_elements = []
    table_bboxes = []
    if page_num in table_pages:
        table_elements = extract_table_from_page(pdf_path, page_num)
        table_bboxes = [el["bbox"] for el in table_elements]

    text_elements = extract_text_from_page(pdf_path, page_num, table_bboxes)
    return table_elements + text_elements

def create_langchain_document(pdf_path: str):
    all_elements = []

    print("Classifying pages...")
    table_pages = classify_pages(pdf_path)

    print("Processing pages in parallel...")
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_page, pdf_path, page.number, table_pages): page.number
            for page in fitz.open(pdf_path)
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                all_elements.extend(result)
            except Exception as e:
                print(f"Error processing page {futures[future]}: {e}")

    all_elements.sort(key=lambda el: (el['page_num'], el['bbox'][1]))

    langchain_docs = []
    for element in all_elements:
        metadata = {
            "source": os.path.basename(pdf_path),
            "page_number": element['page_num'] + 1,
            "type": element['type'],
            "bounding_box": element['bbox']
        }
        langchain_docs.append(Document(page_content=element['content'], metadata=metadata))

    return langchain_docs

if __name__ == "__main__":
    pdf_file = "CHOTGDP23004V012223.pdf"

    print("Starting PDF parsing...")
    time_start = time.time()

    docs = create_langchain_document(pdf_file)

    time_end = time.time()
    print(f"PDF parsing completed in {time_end - time_start:.2f} seconds.")
    print(f"Total documents created: {len(docs)}")

    with open("parsed_documents.json", "w") as f:
        json.dump([doc.dict() for doc in docs], f, indent=4)
