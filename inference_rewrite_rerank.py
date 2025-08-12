"""
inference_rewrite_rerank.py

Provides:
- Query rewriting (Flan-T5 small)
- Hybrid retrieval (semantic FAISS + optional keyword matching)
- Cross-Encoder re-ranking (sentence-transformers CrossEncoder)
- Example integration with a DocVectorPipeline-like index class

Notes:
- This file assumes you have a FAISS-backed vectorstore accessible via a pipeline object
  with a method `similarity_search_with_score(query, k)` that returns list[(Document, score)].
- Cross-encoder and rewrite models can be switched easily (model_name variables).
- Keep GPU usage in mind: CrossEncoder and Transformers will use available GPU if torch + CUDA are installed.
"""

from typing import List, Tuple, Dict, Any, Optional
import os
import math
import itertools
import time

from doc_vector_pipeline import DocVectorPipeline

# Transformers for query rewriting
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Cross-encoder for re-ranking
from sentence_transformers import CrossEncoder

# Document type - adapt if your implementation uses another Document class
from langchain_core.documents import Document

import spacy

# ---------- Utility: basic keyword extraction ----------
nlp = spacy.load("en_core_web_sm")

def extract_keywords(text: str, min_len: int = 3) -> List[str]:
    """
    Keyword extraction using spaCy noun chunks and important tokens.
    """
    doc = nlp(text)
    keywords = set()

    # Add noun phrases
    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip().lower()
        if len(phrase) >= min_len:
            keywords.add(phrase)

    # Add named entities
    for ent in doc.ents:
        phrase = ent.text.strip().lower()
        if len(phrase) >= min_len:
            keywords.add(phrase)

    # Add important tokens (nouns, verbs, adjectives)
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN", "ADJ"} and len(token.text) >= min_len:
            keywords.add(token.text.lower())

    return list(keywords)[:20]


# ---------- Inference Engine ----------
class InferenceEngine:
    """
    Inference Engine that contains:
    - query_rewriter: rewrites/paraphrases the user's original query
    - hybrid_retrieve: gets candidate chunks using semantic retrieval + optional keyword expansion
    - rerank_with_cross_encoder: re-scores candidates using a CrossEncoder and returns top-k
    """

    def __init__(
        self,
        index_pipeline,                 # your DocVectorPipeline or equivalent instance (must implement semantic search)
        rewrite_model_name: str = "google/flan-t5-small",
        cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rewrite_max_length: int = 64,
        rewrite_device: int = 0 if (os.getenv("CUDA_VISIBLE_DEVICES") is not None) else -1,
        cross_encoder_device: int = 0 if (os.getenv("CUDA_VISIBLE_DEVICES") is not None) else -1,
    ):
        """
        index_pipeline: your indexing pipeline instance (with a method `similarity_search_with_score(query, k)`)
        rewrite_model_name: seq2seq model used to rewrite/expand the query (Flan-T5 recommended)
        cross_encoder_model_name: Cross-Encoder model for re-ranking (sentence-transformers)
        rewrite_device/cross_encoder_device: device id for transformers / CrossEncoder (-1 = CPU)
        """
        self.index = index_pipeline
        self.rewrite_model_name = rewrite_model_name
        self.cross_encoder_model_name = cross_encoder_model_name

        # Initialize (lazy) rewrite generator and cross-encoder to avoid long init times if not needed
        self._rewrite_model = None
        self._rewrite_tokenizer = None
        self._rewrite_device = rewrite_device

        self._cross_encoder = None
        self._cross_encoder_device = cross_encoder_device

        self._init_cross_encoder()
        self._init_rewrite_model()

    # ---------- Query Rewriting ----------
    def _init_rewrite_model(self):
        """Initialize transformers seq2seq model / tokenizer for query rewriting."""
        if self._rewrite_model is None:
            print(f"[rewrite] Loading rewrite model: {self.rewrite_model_name}")
            # Use AutoModelForSeq2SeqLM + tokenizer for more control
            self._rewrite_tokenizer = AutoTokenizer.from_pretrained(self.rewrite_model_name)
            self._rewrite_model = AutoModelForSeq2SeqLM.from_pretrained(self.rewrite_model_name)
            if self._rewrite_device is not None and self._rewrite_device >= 0:
                try:
                    import torch
                    self._rewrite_model.to(self._rewrite_device)
                except Exception:
                    pass

    def rewrite_query(self, query: str, num_rewrites: int = 3, temperature: float = 0.0) -> List[str]:
        """
        Rewrites or expands the user's query using a seq2seq model (Flan-T5).
        Returns a list [original_query, rewrite1, rewrite2, ...] (original always first).
        Parameters:
            - num_rewrites: how many rewrites to produce (1..n). Total returned will be num_rewrites + 1.
            - temperature: controls diversity (0 = deterministic)
        """
        rewrites = [query]
        # simple heuristic: short query -> expand with "explain" or "what is" style prompts
        # self._init_rewrite_model()

        prompt = f"Rewrite the query to be more explicit and retrieval-oriented. Keep it short.\n\nQuery: {query}\nRewrite:"
        inputs = self._rewrite_tokenizer(prompt, return_tensors="pt", truncation=True).to(self._rewrite_model.device)

        outputs = self._rewrite_model.generate(
            **inputs,
            max_length=128,
            num_return_sequences=max(1, num_rewrites),
            num_beams=max(num_rewrites, 5),
            do_sample=(temperature > 0.0),
            # temperature=temperature,
            top_k=50 if temperature > 0 else None,
            top_p=0.95 if temperature > 0 else None,
            repetition_penalty=1.2,
        )

        decoded = [self._rewrite_tokenizer.decode(out, skip_special_tokens=True).strip() for out in outputs]
        # Keep unique and prepend original
        for s in decoded:
            if s and s not in rewrites:
                rewrites.append(s)
                if len(rewrites) >= (num_rewrites + 1):
                    break
        return rewrites

    # ---------- Hybrid Retrieval ----------
    def hybrid_retrieve(
        self,
        query: str,
        source_pdf: Optional[str] = None,
        semantic_k: int = 100,
        keyword_k: int = 50,
        top_candidates: int = 200,
        use_keywords: bool = True
    ) -> List[Tuple[Document, float]]:
        """
        Hybrid retrieval:
          - Uses the index's semantic search to get `semantic_k` candidates.
          - Optionally uses a lightweight keyword match to fetch keyword_k additional candidates
            (scanning docstore) — this is cheap if you have docstore in memory; otherwise use only semantic.
          - Merges candidate lists, dedupes, and returns up to `top_candidates` (Document, score)
        Parameters:
          - source_pdf: if provided, filter candidates to that source only (exact match to metadata['source'])
        """
        # 1) Semantic retrieval using the original query
        # Ask index for more candidates than we finally want to re-rank
        try:
            semantic_results = self.index.retrieve(query=query, k=semantic_k, source_pdf=source_pdf, score_threshold=None)
            # index.retrieve returns list[dict] with keys page_content, metadata, score (based on earlier design).
            # Convert to (Document, score)
            semantic_docs_scores = []
            for r in semantic_results:
                # create a lightweight Document wrapper if necessary
                doc_obj = Document(page_content=r.get("page_content"), metadata=r.get("metadata"))
                semantic_docs_scores.append((doc_obj, float(r.get("score", 0.0))))
        except Exception as e:
            print(f"[hybrid_retrieve] semantic search failed: {e}")
            semantic_docs_scores = []

        # 2) Keyword retrieval (optional)
        keyword_results = []
        if use_keywords:
            # extract keywords from the query
            keywords = extract_keywords(query)
            if keywords:
                # We'll scan the index's document store if available (docstore._dict)
                try:
                    candidate_docs = []
                    if hasattr(self.index, "faiss") and self.index.faiss is not None:
                        # try to access stored docs; fallback variations possible
                        # Try docstore._dict route:
                        store = None
                        try:
                            store = self.index.faiss.docstore._dict
                        except Exception:
                            # fallback: maybe index has docstore attribute directly
                            try:
                                store = getattr(self.index.faiss, "docstore", None)
                                if hasattr(store, "_dict"):
                                    store = store._dict
                                else:
                                    store = None
                            except Exception:
                                store = None

                        if store:
                            for doc_id, stored_doc in store.items():
                                # stored_doc likely has `page_content` and `metadata`
                                text = getattr(stored_doc, "page_content", "") or ""
                                meta = getattr(stored_doc, "metadata", {}) or {}
                                if source_pdf and meta.get("source") != source_pdf:
                                    continue
                                # cheap heuristic: count matched keywords
                                text_l = text.lower()
                                match_count = sum(1 for kw in keywords if kw in text_l)
                                if match_count > 0:
                                    # create Document wrapper
                                    cand = Document(page_content=text, metadata=meta)
                                    # Use negative match_count as score (we'll sort later descending)
                                    keyword_results.append((cand, float(-match_count)))
                                    if len(keyword_results) >= keyword_k:
                                        break
                except Exception as e:
                    # if docstore access fails, silently skip keyword retrieval
                    print(f"[hybrid_retrieve] keyword retrieval failed: {e}")
                    keyword_results = []

        # 3) Merge semantic + keyword candidates while deduping (by content)
        merged = []
        seen_texts = set()
        def add_candidate(doc_score_tuple):
            doc, score = doc_score_tuple
            text = (doc.page_content or "").strip()
            key = text[:4096]  # clamp
            if not key:
                return
            if key in seen_texts:
                return
            seen_texts.add(key)
            merged.append((doc, score))

        for ds in semantic_docs_scores:
            add_candidate(ds)
        for ds in keyword_results:
            add_candidate(ds)

        # Return up to top_candidates items
        return merged[:top_candidates]

    # ---------- Cross-encoder re-ranking ----------
    def _init_cross_encoder(self):
        if self._cross_encoder is None:
            print(f"[rerank] Loading cross-encoder model: {self.cross_encoder_model_name}")
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._cross_encoder_device = device

            self._cross_encoder = CrossEncoder(self.cross_encoder_model_name, device=self._cross_encoder_device)

    def rerank_with_cross_encoder(self, query: str, candidates: List[Tuple[Document, float]], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Rerank candidate (Document, score) pairs using a CrossEncoder.
        Returns list of top_k dicts: { 'page_content', 'metadata', 'cross_score', 'original_score' }
        CrossEncoder produces higher scores for better matches (for most pre-trained rerankers).
        """
        if not candidates:
            return []

        # self._init_cross_encoder()

        # Prepare pairs (query, candidate_text) for cross-encoder
        doc_texts = [ (doc.page_content or "") for doc, _ in candidates ]
        pairs = [ (query, txt if isinstance(txt, str) else str(txt)) for txt in doc_texts ]

        # Batch predict in manageable sizes
        batch_size = 64
        cross_scores = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i+batch_size]
            # CrossEncoder accepts list of pairs
            scores = self._cross_encoder.predict(batch_pairs)
            cross_scores.extend(scores)

        # Attach cross-scores to candidates and sort by descending cross_score
        merged = []
        for (doc, orig_score), cs in zip(candidates, cross_scores):
            merged.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "original_score": float(orig_score),
                "cross_score": float(cs)
            })

        # sort descending by cross_score (higher = better)
        merged.sort(key=lambda x: x["cross_score"], reverse=True)

        # return top_k
        return merged[:top_k]

    # ---------- Full flow helper: rewrite -> hybrid -> rerank ----------
    def answer_pipeline_stage_rewrite_and_rerank(
        self,
        query: str,
        source_pdf: Optional[str] = None,
        num_rewrites: int = 2,
        semantic_k: int = 100,
        top_candidates: int = 200,
        rerank_top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Full staged pipeline (focus on rewriting + re-ranking):
         1) rewrite query into multiple variations
         2) run hybrid retrieval for each rewrite and combine candidates
         3) re-rank combined candidates with CrossEncoder
        Returns:
          {
            "rewrites": [...],
            "candidates_count": N,
            "reranked": [ top results ],
            "timings": { ... }
          }
        """
        t0 = time.time()
        rewrites = self.rewrite_query(query, num_rewrites=num_rewrites)
        t1 = time.time()

        # Combine candidates from all rewrites
        all_candidates = []
        # Use a moderate per-rewrite semantic_k to control cost
        per_rewrite_k = max(50, semantic_k // max(1, len(rewrites)))
        for rw in rewrites:
            cands = self.hybrid_retrieve(rw, source_pdf=source_pdf, semantic_k=per_rewrite_k, top_candidates=top_candidates)
            all_candidates.extend(cands)

        # Deduplicate by page_content and keep highest original score
        dedup_map = {}
        for doc, score in all_candidates:
            key = (doc.page_content or "").strip()[:4096]
            if not key:
                continue
            if key not in dedup_map or float(score) < float(dedup_map[key][1]):
                dedup_map[key] = (doc, score)
        combined_candidates = list(dedup_map.values())

        t2 = time.time()
        reranked = self.rerank_with_cross_encoder(query, combined_candidates, top_k=rerank_top_k)
        t3 = time.time()

        return {
            "rewrites": rewrites,
            "timings": {
                "rewrite_s": t1 - t0,
                "retrieve_s": t2 - t1,
                "rerank_s": t3 - t2,
                "total_s": t3 - t0
            },
            "candidates_count": len(combined_candidates),
            "reranked": reranked
        }

# ---------- Example usage ----------
if __name__ == "__main__":
    class IndexStub:
        """
        Minimal stub for testing. Replace with your DocVectorPipeline instance.
        The 'retrieve' method must match the earlier interface:
        retrieve(query: str, source_pdf: Optional[str], k: int, score_threshold: Optional[float])
        -> List[dict(page_content, metadata, score)]
        """
        def __init__(self, faiss_pipeline, source_pdf):
            self.faiss = faiss_pipeline
            t0 = time.time()
            added, skipped = self.faiss.build_from_pdf(source_pdf)
            t1 = time.time()
            print(f"Pipeline finished: added {added} docs, skipped {skipped} duplicates. Time: {t1 - t0:.2f}s")
            pipeline.save()
            
        def retrieve(self, query, source_pdf=None, k=100, score_threshold=None):
            # If pipeline provides retrieve directly this could call that
            # Here we check if the pipeline has the method `retrieve`
            if hasattr(self, "faiss") and self.faiss is not None and hasattr(self.faiss, "retrieve"):
                # use the FAISS object's method signature (k param)
                return self.faiss.retrieve(query=query, source_pdf=source_pdf, k=k, score_threshold=score_threshold)
            # fallback empty
            return []

    t1 = time.time()
    pipeline = DocVectorPipeline()
    index_stub = IndexStub(pipeline, source_pdf="policy.pdf")
    engine = InferenceEngine(index_stub)
    out = engine.answer_pipeline_stage_rewrite_and_rerank(
        query="What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        source_pdf="policy.pdf",
        num_rewrites=2,
        semantic_k=150,
        top_candidates=300,
        rerank_top_k=5
    )
    t2 = time.time()
    # print(f"Total time for inference: {t2 - t1:.2f}s")
    # print(out['timings'])
    for r in out["reranked"]:
        print(r["cross_score"], r["metadata"])
        print(r["page_content"][:50])
