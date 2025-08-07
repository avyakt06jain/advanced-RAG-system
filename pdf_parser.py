import fitz
from collections import defaultdict
import camelot
from langchain_core.documents import Document

"""
PHASE 1: Extract the pages that potentially contains tables
- run pymupdf through the document and extract page numbers which potentially have tables

Input: document
Output: list of page numbers which contains table
"""

def classify_pages(doc: fitz.Document):
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

        valid_columns = 0
        for count in column_counts.values():
            if count >= MIN_ITEMS_PER_COLUMN:
                valid_columns += 1

        if valid_columns >= MIN_COLUMNS:
            table_pages.append(page.number)
            
    return table_pages

"""
PHASE 2: Extract tables from the pages
- Run camelot through the pages and extract data and formats them as elements

Input: pdf_path, page_num
Output: list of dictionary where each elements contains table data
"""

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

"""
PHASE 3: Extract text from pages
- run pymupdf on pages and extarct texts and format it as elements, filtering out tables, headers and footers

Input: page, table_bboxes
Output: list of dictionay where each element contains text data
"""

def extract_text_from_page(page: fitz.Page, table_bboxes: list):
    def is_bbox_inside_any(inner_bbox, outer_bboxes):
        for outer_bbox in outer_bboxes:
            if (inner_bbox[0] >= outer_bbox[0] and inner_bbox[1] >= outer_bbox[1] and
                inner_bbox[2] <= outer_bbox[2] and inner_bbox[3] <= outer_bbox[3]):
                return True
        return False

    text_elements = []

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
                element = {
                    "type": "heading" if is_bold else "paragraph",
                    "content": cleaned_text,
                    "page_num": page.number,
                    "bbox": block_bbox
                }
                text_elements.append(element)
                
    return text_elements

"""
PHASE 4: Creating lanchain document so they can be sent fro chunking
- Compiling all of the fucntions
- First run classfiy pages to get page numbers which potentially have tables
- Then run extract_table_from_page to get table data
- Then run extract_text_from_page to get text data

Input: pdf_path
Output: list of lanchain documents
"""

def create_langchain_document(pdf_path: str):
    all_elements = []

    doc = fitz.open(pdf_path)

    table_pages = classify_pages(doc)

    for page in doc:
        page_num = page.number

        table_bboxes = []
        if page_num in table_pages:
            table_elements = extract_table_from_page(pdf_path, page_num)
            all_elements.extend(table_elements)
            table_bboxes = [el['bbox'] for el in table_elements]

        text_elements = extract_text_from_page(page, table_bboxes)
        all_elements.extend(text_elements)

    all_elements.sort(key=lambda el: (el['page_num'], el['bbox'][1]))
    
    langchain_docs = []
    for element in all_elements:
        metadata = {
            "source": pdf_path.split('/')[-1],
            "page_number": element['page_num'] + 1,
            "type": element['type'],
            "bounding_box": element['bbox']
        }
        doc = Document(page_content=element['content'], metadata=metadata)
        langchain_docs.append(doc)
        
    return langchain_docs