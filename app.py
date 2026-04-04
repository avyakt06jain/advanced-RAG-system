from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
import hashlib
import requests
import faiss

from ingestion_pipeline.ingestionPipeline import run_ingestion_pipeline
from inference_pipeline.inferencePipeline import run_inference_pipeline

app = FastAPI()
security = HTTPBearer()

load_dotenv()

# -- Environment Variables --
# Load API keys from environment variables
API_KEY = os.getenv("API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not all([API_KEY, GEMINI_API_KEY]):
    raise ValueError("Missing API KEY")

CACHE_DIR = "knowledge_base_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# -- RAG State Class --
# This class initializes the embedding and generative models used in the RAG pipeline.
class RAGState:
    def __init__(self, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.genai_client = genai.Client(api_key=GEMINI_API_KEY)
        self.model_name = "gemini-2.0-flash"

        # Dictionaries to cache loaded knowledge bases in memory
        self.loaded_indexes: Dict[str, faiss.Index] = {}
        self.loaded_chunks: Dict[str, List[Dict]] = {}


rag_state = RAGState()


# -- Pydantic Models for Request and Response --
class HealthResponse(BaseModel):
    status: str
    timestamp: str


class EndpointRequest(BaseModel):
    document: str = Field(..., description="URL for the document to be processed")
    queries: List[str] = Field(..., description="List of user queries")


class EndpointResponse(BaseModel):
    answers: List[str] = Field(..., description="Generated answers to the queries")


# --Helper Functions--
def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API KEY",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


def download_pdf(url: str, local_path: str) -> bool:
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading PDF: {e}")
        return False


def get_file_hash(file_path: str):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


# -- API Endpoints --
@app.get("/")
def root():
    return {"message": "Welcome to HackRX Hackathon API"}


@app.get("/health", response_model=HealthResponse)
async def check_health():
    return HealthResponse(
        status="Healthy",
        timestamp=datetime.now().isoformat(),
    )


@app.post(
    "/hackrx/run",
    response_model=EndpointResponse,
    dependencies=[Depends(verify_api_key)],
)
async def generate_response(request: EndpointRequest):
    try:
        temp_pdf_path = os.path.join(CACHE_DIR, "temp_document.pdf")

        # 1. Download and Hash the document
        if not download_pdf(request.document, temp_pdf_path):
            raise HTTPException(
                status_code=400, detail="Could not download document from URL."
            )

        doc_hash = get_file_hash(temp_pdf_path)
        index_path = os.path.join(CACHE_DIR, f"{doc_hash}.index")

        # 2. **CORRECTED LOGIC**: Check cache and run ingestion ONLY IF NEEDED
        if not os.path.exists(index_path):
            print(f"Cache miss for document: {request.document}")
            # Run the full ingestion pipeline for the new document
            run_ingestion_pipeline(
                pdf_path=temp_pdf_path,
                doc_hash=doc_hash,
                embedding_model=rag_state.embedding_model,
                cache_dir=CACHE_DIR,
            )
        else:
            print(f"Cache hit for document: {request.document}")

        # 3. Always run the inference pipeline after ensuring the KB exists
        answers = run_inference_pipeline(
            queries=request.queries,
            doc_hash=doc_hash,
            cache_dir=CACHE_DIR,
            embedding_model=rag_state.embedding_model,
            generative_model=rag_state.genai_client,
            model_name=rag_state.model_name,
            loaded_indexes=rag_state.loaded_indexes,
            loaded_chunks=rag_state.loaded_chunks,
        )

        # 4. Clean up the downloaded PDF
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

        return EndpointResponse(answers=answers)

    except HTTPException as e:
        raise e  # Re-raise known HTTP exceptions
    except Exception as e:
        import traceback

        traceback.print_exc()  # Print the full traceback to the server console for debugging
        raise HTTPException(
            status_code=500, detail=f"An internal server error occurred: {str(e)}"
        )


if __name__ == "__main__":
    port = 8000
    uvicorn.run("app:app", port=port, reload=False)
