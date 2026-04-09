import os
import warnings
from fastapi import FastAPI
import uvicorn
from dotenv import load_dotenv

# Load environment variables from the .env file BEFORE doing anything else
load_dotenv()

# Suppress the Python 3.14 Pydantic V1 warning
warnings.filterwarnings("ignore", message=".*Core Pydantic V1.*")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector

app = FastAPI(title="Lesson Plan RAG System")

# --- Configuration & Shared State ---
# Fetch strictly from environment, raise error if missing
DB_CONNECTION_STRING = os.getenv("DATABASE_URL")
if not DB_CONNECTION_STRING:
    raise ValueError("CRITICAL ERROR: DATABASE_URL is not set in the .env file.")

COLLECTION_NAME = "lesson_plans"

# Global variables for lazy loading
_embeddings = None
_vector_store = None

def get_vector_store():
    """
    Dependency to lazy-load the embedding model and vector store.
    Imported by our routers so they share the same memory instance.
    """
    global _embeddings, _vector_store
    if _vector_store is None:
        print("Lazy Loading Embedding Model... (This will take a moment on the first request)")
        # UPDATED: Swapped to a much lighter multilingual model (~470MB)
        # This handles Hindi and Sanskrit perfectly without hogging your RAM or bandwidth.
        _embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
        
        _vector_store = PGVector(
            connection_string=DB_CONNECTION_STRING,
            embedding_function=_embeddings,
            collection_name=COLLECTION_NAME,
            use_jsonb=True, 
        )
        print("System Ready! Vector Store Initialized.")
    return _vector_store


# --- Router Registration ---
# We import these AFTER defining get_vector_store to prevent circular imports.
from router.ingest import router as ingest_router
from router.generate_lesson_plan import router as generate_router
from router.generate_academic_plan import router as academic_router

# Register the routes with the main app
app.include_router(ingest_router)
app.include_router(generate_router)
app.include_router(academic_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)