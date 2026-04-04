import os
import json
import time
import faiss
import numpy as np
from google import genai
from google.genai import errors as genai_errors
from typing import List, Dict
from rank_bm25 import BM25Okapi
import re

_cross_encoder_model = None


def _get_cross_encoder_model():
    global _cross_encoder_model
    if _cross_encoder_model is None:
        from sentence_transformers import CrossEncoder

        _cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder_model


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _bm25_search(query: str, chunks: List[Dict], top_k: int = 10) -> List[Dict]:
    tokenized_corpus = [_tokenize(chunk["content"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [{"chunk": chunks[i], "score": float(scores[i])} for i in top_indices]


def _semantic_search(
    query: str, embedding_model, faiss_index, chunks: List[Dict], top_k: int = 10
) -> List[Dict]:
    query_vector = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_vector, k=min(top_k, len(chunks)))
    retrieved = []
    for rank, i in enumerate(indices[0]):
        if i < len(chunks):
            retrieved.append(
                {"chunk": chunks[i], "score": float(1.0 / (1.0 + distances[0][rank]))}
            )
    return retrieved


def _reciprocal_rank_fusion(
    sem_results: List[Dict], bm25_results: List[Dict], k: int = 60
) -> List[Dict]:
    scores: Dict[int, float] = {}
    chunk_map: Dict[int, Dict] = {}
    for rank, item in enumerate(sem_results):
        idx = id(item["chunk"])
        scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)
        chunk_map[idx] = item["chunk"]
    for rank, item in enumerate(bm25_results):
        idx = id(item["chunk"])
        scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)
        chunk_map[idx] = item["chunk"]
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_map[idx] for idx, _ in ranked]


def _cross_encoder_rerank(query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
    try:
        model = _get_cross_encoder_model()
        pairs = [[query, c["content"]] for c in chunks]
        scores = model.predict(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[:top_k]]
    except Exception as e:
        print(f"Cross-encoder reranking failed, falling back to RRF order: {e}")
        return chunks[:top_k]


def _generate_answer_with_gemini(
    query: str,
    context_chunks: List[Dict],
    genai_client,
    model_name: str,
    max_retries: int = 3,
) -> str:
    if not context_chunks:
        return "I do not have enough information to answer this question based on the provided document."

    context_parts = []
    for chunk in context_chunks:
        meta = chunk.get("metadata", {})
        header = f"[Source: {meta.get('source_document', 'Unknown')}, Page {meta.get('page_number', '?')}, Section: {meta.get('section_header', 'N/A')}]"
        context_parts.append(f"{header}\n{chunk['content']}")
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are an expert AI assistant specialized in analyzing insurance policy documents. Answer the user's question based *only* on the provided context. Do not use external knowledge.

Instructions:
1. Formulate a clear, concise, direct answer using only the provided context.
2. For every statement, cite the source as [Source: document_name, Page X, Section: Y].
3. If the answer is not in the context, say: "I'm sorry, but the information required to answer your question is not available in the provided documents."

Context:
{context}

Question: {query}

Answer:"""

    last_error = None
    for attempt in range(max_retries):
        try:
            response = genai_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"temperature": 0},
            )
            return response.text.strip()
        except genai_errors.APIError as e:
            last_error = e
            if e.code == 429:
                retry_after = 30
                try:
                    for detail in getattr(e, "_details", []):
                        if isinstance(detail, dict) and "retryDelay" in detail:
                            delay_str = detail["retryDelay"]
                            retry_after = (
                                int("".join(filter(str.isdigit, delay_str))) + 1
                            )
                            break
                except Exception:
                    retry_after = 30 * (attempt + 1)
                wait = retry_after * (2**attempt)
                print(
                    f"Rate limited. Retrying in {wait}s (attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(wait)
            else:
                break
        except Exception as e:
            last_error = e
            break

    print(f"Error calling Gemini API after {max_retries} retries: {last_error}")
    return f"Sorry, an error occurred while generating the answer: {last_error}"


def run_inference_pipeline(
    queries: List[str],
    doc_hash: str,
    cache_dir: str,
    embedding_model,
    generative_model,
    model_name: str,
    loaded_indexes: Dict,
    loaded_chunks: Dict,
) -> List[str]:
    index_path = os.path.join(cache_dir, f"{doc_hash}.index")
    chunks_path = os.path.join(cache_dir, f"{doc_hash}.json")

    if doc_hash not in loaded_indexes:
        print(f"Loading knowledge base for '{doc_hash}' into memory...")
        try:
            loaded_indexes[doc_hash] = faiss.read_index(index_path)
            with open(chunks_path, "r", encoding="utf-8") as f:
                loaded_chunks[doc_hash] = json.load(f)
        except FileNotFoundError:
            return [
                f"Error: Knowledge base for document hash {doc_hash} not found."
                for _ in queries
            ]

    faiss_index = loaded_indexes[doc_hash]
    chunks_data = loaded_chunks[doc_hash]

    final_answers = []
    for query in queries:
        sem_results = _semantic_search(
            query, embedding_model, faiss_index, chunks_data, top_k=10
        )
        bm25_results = _bm25_search(query, chunks_data, top_k=10)
        fused_chunks = _reciprocal_rank_fusion(sem_results, bm25_results)
        reranked_chunks = _cross_encoder_rerank(query, fused_chunks, top_k=5)
        answer = _generate_answer_with_gemini(
            query, reranked_chunks, generative_model, model_name
        )
        final_answers.append(answer)

    return final_answers
