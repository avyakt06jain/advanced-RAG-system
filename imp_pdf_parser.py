import fitz
import camelot
import pdfplumber
from collections import defaultdict
from langchain_core.documents import Document
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.cluster import DBSCAN

# Configure logging for backend notifications
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFExtractor:
    def __init__(self):
        self.COLUMN_TOLERANCE_PERCENT = 0.02  # 2% of page width
        self.MIN_COLUMNS = 2
        self.MIN_ITEMS_PER_COLUMN = 3
        self.HEADER_FOOTER_PERCENT = 0.08
        self.TABLE_CONFIDENCE_THRESHOLD = 0.6
        
    def _get_dynamic_margins(self, page: fitz.Page) -> Tuple[float, float]:
        """Dynamically calculate header and footer margins based on content distribution."""
        blocks = page.get_text("blocks")
        if not blocks:
            return page.rect.height * self.HEADER_FOOTER_PERCENT, page.rect.height * (1 - self.HEADER_FOOTER_PERCENT)
        
        y_positions = [block[1] for block in blocks if block[6] == 0]  # Text blocks only
        if len(y_positions) < 3:
            return page.rect.height * self.HEADER_FOOTER_PERCENT, page.rect.height * (1 - self.HEADER_FOOTER_PERCENT)
        
        y_positions.sort()
        
        # Find natural gaps for header/footer detection
        gaps = []
        for i in range(1, len(y_positions)):
            gap = y_positions[i] - y_positions[i-1]
            if gap > 20:  # Significant gap
                gaps.append((gap, y_positions[i-1], y_positions[i]))
        
        if gaps:
            gaps.sort(reverse=True)  # Largest gaps first
            header_margin = min(gaps[0][2], page.rect.height * 0.15)  # Cap at 15%
            footer_start = max(gaps[-1][1] if len(gaps) > 1 else y_positions[-1], 
                             page.rect.height * 0.85)  # Cap at 85%
        else:
            header_margin = page.rect.height * self.HEADER_FOOTER_PERCENT
            footer_start = page.rect.height * (1 - self.HEADER_FOOTER_PERCENT)
            
        return header_margin, footer_start

    def _cluster_columns(self, x_positions: List[float], page_width: float) -> int:
        """Use clustering to identify column positions more accurately."""
        if len(x_positions) < 2:
            return len(x_positions)
        
        # Convert to numpy array and reshape for DBSCAN
        X = np.array(x_positions).reshape(-1, 1)
        
        # Use dynamic epsilon based on page width
        eps = page_width * self.COLUMN_TOLERANCE_PERCENT
        clustering = DBSCAN(eps=eps, min_samples=self.MIN_ITEMS_PER_COLUMN).fit(X)
        
        # Count number of clusters (excluding noise points labeled as -1)
        n_clusters = len(set(clustering.labels_) - {-1})
        return n_clusters

    def classify_pages(self, doc: fitz.Document) -> List[int]:
        """Enhanced table detection with better column identification."""
        table_pages = []

        for page in doc:
            column_x_positions = []
            blocks = page.get_text("blocks")
            
            for block in blocks:
                if block[6] == 0:  # Text block
                    column_x_positions.append(block[0])

            if len(column_x_positions) < self.MIN_COLUMNS:
                continue
                
            # Use clustering for better column detection
            n_columns = self._cluster_columns(column_x_positions, page.rect.width)
            
            if n_columns >= self.MIN_COLUMNS:
                table_pages.append(page.number)
                logger.info(f"Page {page.number + 1}: Detected {n_columns} columns - marked as potential table page")
                
        return table_pages

    def _extract_table_with_camelot(self, pdf_path: str, page_num: int) -> List[Dict]:
        """Primary table extraction using Camelot."""
        table_elements = []
        try:
            # Try lattice first (for bordered tables)
            tables = camelot.read_pdf(pdf_path, pages=str(page_num + 1), flavor='lattice')
            
            # If no tables found with lattice, try stream (for borderless tables)
            if len(tables) == 0:
                tables = camelot.read_pdf(pdf_path, pages=str(page_num + 1), flavor='stream')
            
            for table in tables:
                # Check table quality
                if hasattr(table, 'accuracy') and table.accuracy < self.TABLE_CONFIDENCE_THRESHOLD:
                    logger.warning(f"Low confidence table on page {page_num + 1} (accuracy: {table.accuracy:.2f})")
                
                # Convert to structured format (better for RAG than markdown)
                table_data = {
                    "headers": table.df.columns.tolist(),
                    "rows": table.df.values.tolist(),
                    "raw_df": table.df  # Keep for fallback
                }

                table_element = {
                    "type": "table",
                    "content": table_data,
                    "markdown": table.df.to_markdown(index=False),  # Keep markdown for compatibility
                    "page_num": page_num,
                    "bbox": table._bbox,
                    "confidence": getattr(table, 'accuracy', 1.0)
                }
                table_elements.append(table_element)
                
        except Exception as e:
            logger.error(f"Camelot failed on page {page_num + 1}: {e}")
            raise  # Re-raise to trigger fallback
            
        return table_elements

    def _extract_table_with_pdfplumber_fallback(self, pdf_path: str, page_num: int) -> List[Dict]:
        """Fallback table extraction using pdfplumber."""
        table_elements = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_num < len(pdf.pages):
                    page = pdf.pages[page_num]
                    tables = page.extract_tables()
                    
                    for i, table in enumerate(tables):
                        if table and len(table) > 1:  # Valid table with headers and data
                            headers = table[0] if table[0] else [f"Column_{j}" for j in range(len(table[1]) if len(table) > 1 else 0)]
                            rows = table[1:] if len(table) > 1 else []
                            
                            # Create DataFrame equivalent structure
                            table_data = {
                                "headers": headers,
                                "rows": rows,
                                "raw_df": None
                            }
                            
                            # Estimate bbox (pdfplumber doesn't provide exact bbox)
                            bbox = (0, 0, page.width, page.height)  # Placeholder
                            
                            table_element = {
                                "type": "table",
                                "content": table_data,
                                "markdown": self._create_markdown_from_data(headers, rows),
                                "page_num": page_num,
                                "bbox": bbox,
                                "confidence": 0.5,  # Lower confidence for fallback
                                "extraction_method": "pdfplumber_fallback"
                            }
                            table_elements.append(table_element)
                            
        except Exception as e:
            logger.error(f"PDFplumber fallback failed on page {page_num + 1}: {e}")
            
        return table_elements

    def _create_markdown_from_data(self, headers: List[str], rows: List[List[str]]) -> str:
        """Create markdown table from headers and rows."""
        if not headers or not rows:
            return ""
            
        markdown_lines = []
        markdown_lines.append("| " + " | ".join(str(h) for h in headers) + " |")
        markdown_lines.append("| " + " | ".join("---" for _ in headers) + " |")
        
        for row in rows:
            if row:  # Skip empty rows
                markdown_lines.append("| " + " | ".join(str(cell) if cell else "" for cell in row) + " |")
                
        return "\n".join(markdown_lines)

    def _extract_table_pymupdf_fallback(self, page: fitz.Page) -> List[Dict]:
        """Last resort: extract table-like content as structured text using PyMuPDF."""
        logger.warning(f"Using PyMuPDF fallback for table extraction on page {page.number + 1}")
        
        blocks = page.get_text("blocks")
        table_blocks = []
        
        # Group blocks that might form a table
        for block in blocks:
            if block[6] == 0:  # Text block
                text = block[4].strip()
                if text and len(text.split()) > 2:  # Potential table row
                    table_blocks.append({
                        "text": text,
                        "bbox": block[:4],
                        "x": block[0],
                        "y": block[1]
                    })
        
        if not table_blocks:
            return []
        
        # Sort by y-position (top to bottom)
        table_blocks.sort(key=lambda x: x["y"])
        
        # Simple table structure
        table_element = {
            "type": "table",
            "content": {
                "headers": ["Content"],  # Simple single column
                "rows": [[block["text"]] for block in table_blocks],
                "raw_df": None
            },
            "markdown": "\n".join([f"| {block['text']} |" for block in table_blocks]),
            "page_num": page.number,
            "bbox": (
                min(b["bbox"][0] for b in table_blocks),
                min(b["bbox"][1] for b in table_blocks),
                max(b["bbox"][2] for b in table_blocks),
                max(b["bbox"][3] for b in table_blocks)
            ),
            "confidence": 0.3,  # Very low confidence
            "extraction_method": "pymupdf_fallback"
        }
        
        return [table_element]

    def extract_table_from_page(self, pdf_path: str, page_num: int) -> List[Dict]:
        """Enhanced table extraction with multiple fallback strategies."""
        # Try Camelot first
        try:
            table_elements = self._extract_table_with_camelot(pdf_path, page_num)
            if table_elements:
                logger.info(f"Successfully extracted {len(table_elements)} tables from page {page_num + 1} using Camelot")
                return table_elements
        except Exception as e:
            logger.warning(f"Camelot extraction failed on page {page_num + 1}, trying pdfplumber: {e}")
        
        # Try pdfplumber as fallback
        try:
            table_elements = self._extract_table_with_pdfplumber_fallback(pdf_path, page_num)
            if table_elements:
                logger.info(f"Successfully extracted {len(table_elements)} tables from page {page_num + 1} using pdfplumber fallback")
                return table_elements
        except Exception as e:
            logger.warning(f"PDFplumber fallback failed on page {page_num + 1}, using PyMuPDF: {e}")
        
        # Last resort: PyMuPDF
        try:
            with fitz.open(pdf_path) as doc:
                if page_num < len(doc):
                    page = doc[page_num]
                    table_elements = self._extract_table_pymupdf_fallback(page)
                    if table_elements:
                        logger.info(f"Extracted table-like content from page {page_num + 1} using PyMuPDF fallback")
                        return table_elements
        except Exception as e:
            logger.error(f"All table extraction methods failed on page {page_num + 1}: {e}")
        
        return []

    def _is_heading(self, span_info: Dict) -> bool:
        """Enhanced heading detection using multiple criteria."""
        font_name = span_info.get("font", "").lower()
        font_size = span_info.get("size", 0)
        font_flags = span_info.get("flags", 0)
        
        # Check for bold (flag 16 is bold in PyMuPDF)
        is_bold = "bold" in font_name or (font_flags & 16) != 0
        
        # Check for larger font size (relative to page average)
        is_large = font_size > 12  # Simple threshold, could be made dynamic
        
        # Check for specific font characteristics
        is_heading_font = any(word in font_name for word in ["heading", "title", "header"])
        
        return is_bold or is_large or is_heading_font

    def extract_text_from_page(self, page: fitz.Page, table_bboxes: List[Tuple]) -> List[Dict]:
        """Enhanced text extraction with better heading detection and margin calculation."""
        def is_bbox_inside_any(inner_bbox, outer_bboxes):
            for outer_bbox in outer_bboxes:
                if (inner_bbox[0] >= outer_bbox[0] - 5 and inner_bbox[1] >= outer_bbox[1] - 5 and
                    inner_bbox[2] <= outer_bbox[2] + 5 and inner_bbox[3] <= outer_bbox[3] + 5):
                    return True
            return False

        text_elements = []

        # Dynamic margin calculation
        header_margin, footer_start = self._get_dynamic_margins(page)

        text_blocks = page.get_text("dict")["blocks"]

        for block in text_blocks:
            if block.get("type") == 0 and "lines" in block:
                block_bbox = block["bbox"]

                # Skip if inside table
                if is_bbox_inside_any(block_bbox, table_bboxes):
                    continue

                # Skip headers and footers
                if block_bbox[1] < header_margin or block_bbox[3] > footer_start:
                    continue

                block_text = ""
                is_heading = False
                font_sizes = []
                
                for line in block["lines"]:
                    for span in line["spans"]:
                        block_text += span["text"] + " "
                        font_sizes.append(span.get("size", 0))
                        
                        # Enhanced heading detection
                        if self._is_heading(span):
                            is_heading = True
                
                cleaned_text = block_text.strip()
                if cleaned_text and len(cleaned_text) > 2:  # Minimum text length
                    avg_font_size = np.mean(font_sizes) if font_sizes else 0
                    
                    element = {
                        "type": "heading" if is_heading else "paragraph",
                        "content": cleaned_text,
                        "page_num": page.number,
                        "bbox": block_bbox,
                        "avg_font_size": avg_font_size
                    }
                    text_elements.append(element)
                    
        return text_elements

    def create_langchain_document(self, pdf_path: str) -> List[Document]:
        """Main orchestration function with comprehensive error handling."""
        all_elements = []
        extraction_stats = {
            "total_pages": 0,
            "table_pages": 0,
            "tables_extracted": 0,
            "camelot_success": 0,
            "pdfplumber_fallback": 0,
            "pymupdf_fallback": 0,
            "failed_extractions": 0
        }

        try:
            doc = fitz.open(pdf_path)
            extraction_stats["total_pages"] = len(doc)
            
            # Phase 1: Identify table pages
            table_pages = self.classify_pages(doc)
            extraction_stats["table_pages"] = len(table_pages)
            logger.info(f"Identified {len(table_pages)} potential table pages out of {len(doc)} total pages")

            # Phase 2 & 3: Extract content from each page
            for page in doc:
                page_num = page.number

                table_bboxes = []
                if page_num in table_pages:
                    table_elements = self.extract_table_from_page(pdf_path, page_num)
                    
                    if table_elements:
                        all_elements.extend(table_elements)
                        table_bboxes = [el['bbox'] for el in table_elements]
                        extraction_stats["tables_extracted"] += len(table_elements)
                        
                        # Count extraction methods used
                        for elem in table_elements:
                            method = elem.get("extraction_method", "camelot")
                            if method == "camelot":
                                extraction_stats["camelot_success"] += 1
                            elif method == "pdfplumber_fallback":
                                extraction_stats["pdfplumber_fallback"] += 1
                            elif method == "pymupdf_fallback":
                                extraction_stats["pymupdf_fallback"] += 1
                    else:
                        extraction_stats["failed_extractions"] += 1

                # Extract text content
                text_elements = self.extract_text_from_page(page, table_bboxes)
                all_elements.extend(text_elements)

            # Phase 4: Sort and create LangChain documents
            all_elements.sort(key=lambda el: (el['page_num'], el['bbox'][1]))
            
            langchain_docs = []
            for element in all_elements:
                # Enhanced metadata for better RAG performance
                metadata = {
                    "source": pdf_path.split('/')[-1],
                    "page_number": element['page_num'] + 1,
                    "type": element['type'],
                    "bounding_box": element['bbox']
                }
                
                # Add table-specific metadata
                if element['type'] == 'table':
                    metadata.update({
                        "table_confidence": element.get('confidence', 1.0),
                        "extraction_method": element.get('extraction_method', 'camelot'),
                        "table_structure": {
                            "headers": element['content'].get('headers', []),
                            "num_rows": len(element['content'].get('rows', []))
                        }
                    })
                    
                    # For RAG, use structured content rather than markdown
                    # This allows better semantic understanding
                    content = f"Table with {len(element['content'].get('headers', []))} columns:\n"
                    content += f"Headers: {', '.join(element['content'].get('headers', []))}\n"
                    content += element.get('markdown', '')
                else:
                    content = element['content']
                    if 'avg_font_size' in element:
                        metadata['avg_font_size'] = element['avg_font_size']
                
                doc_obj = Document(page_content=content, metadata=metadata)
                langchain_docs.append(doc_obj)

            # Log final statistics
            logger.info(f"Extraction completed for {pdf_path}")
            logger.info(f"Statistics: {extraction_stats}")
            
            return langchain_docs

        except Exception as e:
            logger.error(f"Critical error processing {pdf_path}: {e}")
            raise

# Usage function
def extract_pdf_content(pdf_path: str) -> List[Document]:
    """
    Main function to extract content from PDF for RAG system.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of LangChain Document objects ready for chunking and embedding
    """
    extractor = PDFExtractor()
    return extractor.create_langchain_document(pdf_path)