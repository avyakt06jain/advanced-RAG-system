import json
import re
import os

# --- Configuration ---
def create_intelligent_chunks(raw_data: json, source_filename: str) -> list:
    """
    Processes raw extracted text into semantically meaningful chunks with metadata.
    This revised version uses a multi-pass approach for better accuracy.
    """

    data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data

    # --- Pass 1: Clean and create a list of "proto-chunks" ---
    # Each text block from the original JSON is treated as a potential chunk.
    proto_chunks = []
    for element in data:
        content = element.get("content", "").strip()
        proto_chunks.append({
                "content": content,
                "page_num": element.get("page_num")
            })

    # --- Pass 2: Contextual Merging ---
    # Merge list items and continuations into their parent chunk.
    if not proto_chunks:
        return []

    merged_chunks = []
    # Pattern to identify the start of a list item (e.g., "a.", "b)", "i.", "•")
    list_item_pattern = re.compile(r'^\s*(?:[a-z][\.\)]|[ivx]+\.|[•])', re.IGNORECASE)

    # Start with the first chunk
    merged_chunks.append(proto_chunks[0])

    for i in range(1, len(proto_chunks)):
        current_content = proto_chunks[i]["content"]
        
        # Merge Condition: If the current line is a list item, merge it with the previous chunk.
        if list_item_pattern.match(current_content):
            # Append to the content of the last chunk in the merged list
            merged_chunks[-1]["content"] += f"\n{current_content}"
        else:
            # Otherwise, it's a new chunk
            merged_chunks.append(proto_chunks[i])

    # --- Pass 3: Metadata Enrichment and Finalization ---
    final_chunks = []
    current_section_header = "Preamble"
    # Pattern to capture the text of major section headers
    major_header_pattern = re.compile(r'^(?:[IVXLCDM]+\.|[A-Z]\)|\d+\.)\s+([A-Z\s\-&]+:?)$')

    for chunk in merged_chunks:
        content = chunk["content"]
        
        # Check if this chunk's content IS a major section header
        header_match = major_header_pattern.match(content.split('\n')[0])
        if header_match:
            # Update the current section header
            current_section_header = header_match.group(1).strip().replace(':', '')

        # Create the final chunk with enriched metadata
        final_chunk = {
            "content": content,
            "metadata": {
                "source_document": source_filename,
                "page_number": chunk["page_num"],
                "section_header": current_section_header,
                "chunk_type": "text"
            }
        }
        final_chunks.append(final_chunk)

    return json.dumps(final_chunks, indent=2, ensure_ascii=False)

