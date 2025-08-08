#!/usr/bin/env python3
"""
LLM-Powered Intelligent Document Query System - FastAPI Server
This server provides an API for document loading, querying, and batch processing.
Refactored for LlamaIndex with a focus on performance optimization.
"""
import logging
import os
import sys
import requests
import tempfile
import asyncio
import uvicorn
import time
import inspect
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

# Ensure the project structure is correct for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import config
from src.query_processor import QueryProcessor
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import PropertyGraphIndex
from llama_index.core.ingestion import IngestionPipeline
from pinecone import Pinecone
import fitz  # PyMuPDF
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import Document
from llama_index.core.postprocessor import LongContextReorder
from neo4j import GraphDatabase
from src.graphiti_client import GraphitiClient

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Pydantic models (moved to the top to prevent NameErrors) ---
class QueryRequest(BaseModel):
    query: str
    document_path: Optional[str] = None

class BatchQueryRequest(BaseModel):
    documents: Optional[str] = Field(None, description="URL to document for indexing")
    questions: List[str] = Field(..., description="List of questions to answer")

class HackRxResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers to the questions")

# Global system components
query_processor: QueryProcessor = QueryProcessor()
pipeline: Optional[IngestionPipeline] = None
index: Optional[VectorStoreIndex] = None
pg_index: Optional[PropertyGraphIndex] = None
neo4j_driver = None
executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
pc_client = Pinecone(api_key=config.PINECONE_API_KEY)
graphiti_client = GraphitiClient()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the application.
    This replaces the deprecated @app.on_event("startup") and "shutdown" decorators.
    """
    global index, pipeline, pg_index, neo4j_driver
    logger.info("🚀 System initialization started on startup.")
    Settings.embed_model = OpenAIEmbedding(model=config.EMBEDDING_MODEL)
    Settings.llm = OpenAI(model=config.OPENAI_MODEL)
    Settings.chunk_size = config.CHUNK_SIZE
    Settings.chunk_overlap = config.CHUNK_OVERLAP

    try:
        # Fixed: Call names() method instead of treating it as a property
        existing_indexes = pc_client.list_indexes().names()
        if config.PINECONE_INDEX_NAME not in existing_indexes:
            logger.info(f"🔍 Pinecone index '{config.PINECONE_INDEX_NAME}' not found. Creating a new one...")
            pc_client.create_index(
                name=config.PINECONE_INDEX_NAME,
                dimension=1536,
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": config.PINECONE_ENVIRONMENT}}
            )
            logger.info(f"✅ Pinecone index '{config.PINECONE_INDEX_NAME}' created successfully.")
        else:
            logger.info(f"✅ Pinecone index '{config.PINECONE_INDEX_NAME}' already exists. Connecting to it.")

        pinecone_vector_store = PineconeVectorStore(
            pinecone_client=pc_client,
            index_name=config.PINECONE_INDEX_NAME,
            environment=config.PINECONE_ENVIRONMENT
        )
        storage_context = StorageContext.from_defaults(vector_store=pinecone_vector_store)
        index = VectorStoreIndex([], storage_context=storage_context, show_progress=True)
        
        # Initialize Neo4j graph store and property graph index
        try:
            # Initialize low-level driver for health checks
            neo4j_driver = GraphDatabase.driver(
                config.NEO4J_URI,
                auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD),
            )
            try:
                neo4j_driver.verify_connectivity()
                logger.info("✅ Neo4j connectivity verified.")
            except Exception as e:
                logger.warning(f"Neo4j connectivity check failed: {e}")

            graph_store = Neo4jPropertyGraphStore(
                username=config.NEO4J_USERNAME,
                password=config.NEO4J_PASSWORD,
                url=config.NEO4J_URI,
                database=config.NEO4J_DATABASE,
            )
            pg_index = PropertyGraphIndex.from_documents(
                documents=[],
                property_graph_store=graph_store,
                embed_model=Settings.embed_model,
                show_progress=False,
            )
            logger.info("✨ Neo4j graph store initialized.")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Neo4j graph store: {e}")

        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP),
                Settings.embed_model
            ],
            vector_store=pinecone_vector_store
        )
        logger.info("✨ LlamaIndex and Pinecone components initialized.")
        yield
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        raise
    finally:
        logger.info("🛑 Shutting down thread pool executor...")
        executor.shutdown(wait=True)
        logger.info("🛑 Thread pool executor shut down.")
        # Close Neo4j driver
        try:
            if neo4j_driver is not None:
                neo4j_driver.close()
                logger.info("🛑 Neo4j driver closed.")
        except Exception:
            pass

app = FastAPI(
    title="LLM Document Processing System",
    description="Intelligent query-retrieval system for unstructured documents, powered by LlamaIndex.",
    version="2.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)

def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)):
    """Verify Bearer token authentication."""
    SECRET_TOKEN = "52e10e56bc55ec56dd26783ea2cef3196cf8f7c6354a5b39d872559874bd29a5"
    if not credentials or credentials.credentials != SECRET_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing authentication token")
    return credentials.credentials

@app.get("/")
async def root():
    return {
        "message": "LLM Document Processing System",
        "version": "2.1.0 (LlamaIndex)",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    try:
        # Pinecone stats (v3 returns a dict)
        pinecone_index = pc_client.Index(config.PINECONE_INDEX_NAME)
        stats_obj = pinecone_index.describe_index_stats()
        # Normalize to plain dict if needed
        if isinstance(stats_obj, dict):
            stats = stats_obj
        elif hasattr(stats_obj, "to_dict"):
            stats = stats_obj.to_dict()
        elif hasattr(stats_obj, "model_dump"):
            stats = stats_obj.model_dump()
        else:
            stats = {}
        document_count = int(stats.get('total_vector_count', 0)) if isinstance(stats, dict) else 0
        is_healthy = True  # If stats call succeeds, treat as healthy

        # Neo4j health
        neo4j_ok = False
        try:
            if neo4j_driver is not None:
                with neo4j_driver.session(database=config.NEO4J_DATABASE) as session:
                    result = session.run("RETURN 1 as ok").single()
                    neo4j_ok = bool(result and result["ok"] == 1)
        except Exception as e:
            logger.warning(f"Neo4j health check failed: {e}")

        components = {
            'pinecone_index': True,
            'neo4j': neo4j_ok,
            'openai_api': True
        }

        system_status = {
            'is_healthy': is_healthy,
            'components': components,
            'document_count': document_count
        }
        
        logger.info("💚 Health check completed.")
        return system_status
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

async def _process_pdf_page(file_path: str, page_num: int) -> str:
    """Helper function to extract text from a single PDF page concurrently."""
    return await asyncio.to_thread(_thread_process_pdf_page_with_pymupdf_first, file_path, page_num)

def _thread_process_pdf_page_with_pymupdf_first(file_path: str, page_num: int) -> str:
    """Synchronous function using PyMuPDF for fast, accurate text extraction."""
    with fitz.open(file_path) as pdf_doc:
        page = pdf_doc.load_page(page_num)
        text = page.get_text("text")
        return text or ""

async def _load_and_index_from_url(document_url: str):
    """Helper function to load and index a document from a URL using LlamaIndex with optimizations."""
    global index, pipeline, pg_index
    if not pipeline:
         raise RuntimeError("Ingestion pipeline is not initialized.")
    logger.info(f"🌐 Downloading document from URL: {document_url}")
    
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / 'document.pdf'
    
    try:
        start_download = time.time()
        response = requests.get(document_url, timeout=60)
        response.raise_for_status()
        temp_path.write_bytes(response.content)
        logger.info(f"📁 Document downloaded in {time.time() - start_download:.2f}s.")
        
        start_processing = time.time()
        with fitz.open(str(temp_path)) as pdf_doc:
            num_pages = pdf_doc.page_count
        
        logger.info(f"⚙️ Starting parallel text extraction for {num_pages} pages...")
        processing_tasks = [_process_pdf_page(str(temp_path), i) for i in range(num_pages)]
        pages_content = await asyncio.gather(*processing_tasks)
        
        full_text = "\n".join(pages_content)
        logger.info(f"✅ Text extraction completed in {time.time() - start_processing:.2f}s. Total characters extracted: {len(full_text)}")

        # Create page-level documents for better citations and graph nodes
        documents = []
        for i, page_text in enumerate(pages_content):
            if not page_text:
                continue
            documents.append(
                Document(text=page_text, metadata={"source": document_url, "page": i + 1})
            )
        
        start_ingestion = time.time()
        logger.info("📦 Starting LlamaIndex ingestion pipeline...")
        try:
            await pipeline.arun(documents=documents)
        except Exception as e:
            logger.warning(f"Async ingestion failed: {e}; falling back to sync run.")
            pipeline.run(documents=documents)
        logger.info(f"🎉 Ingestion pipeline completed in {time.time() - start_ingestion:.2f}s.")
        
        # Fixed: Need to recreate index reference after ingestion
        pinecone_vector_store = PineconeVectorStore(
            pinecone_client=pc_client,
            index_name=config.PINECONE_INDEX_NAME,
            environment=config.PINECONE_ENVIRONMENT
        )
        storage_context = StorageContext.from_defaults(vector_store=pinecone_vector_store)
        try:
            index = VectorStoreIndex.from_vector_store(pinecone_vector_store, storage_context=storage_context)
        except Exception as e:
            logger.warning(f"Vector index refresh failed: {e}; rebuilding empty index handle.")
            index = VectorStoreIndex([], storage_context=storage_context, show_progress=False)

        # Also upsert into the property graph for schema-rich retrieval (best-effort)
        try:
            if pg_index is not None:
                # Prefer async API if available to avoid nested event loop issues
                if hasattr(pg_index, "arefresh_ref_docs"):
                    await pg_index.arefresh_ref_docs(documents)  # type: ignore[attr-defined]
                elif hasattr(pg_index, "refresh_ref_docs"):
                    await asyncio.to_thread(pg_index.refresh_ref_docs, documents)  # type: ignore[attr-defined]
                logger.info("🧠 Document added to Neo4j property graph.")
        except Exception as e:
            logger.warning(f"Graph ingestion skipped due to error: {e}")

        # Emit Graphiti episode for the ingestion event (best-effort)
        try:
            await graphiti_client.send_episode(
                title="document_ingested",
                content=f"{document_url}",
                metadata={
                    "type": "pdf_ingestion",
                    "pages": len(pages_content),
                    "chunks": len(documents),
                    "source": document_url,
                },
            )
        except Exception as e:
            logger.debug(f"Graphiti episode write failed: {e}")
        
        try:
            stats_obj = pc_client.Index(config.PINECONE_INDEX_NAME).describe_index_stats()
            if isinstance(stats_obj, dict):
                total_vecs = int(stats_obj.get('total_vector_count', 0))
            elif hasattr(stats_obj, 'to_dict'):
                total_vecs = int(stats_obj.to_dict().get('total_vector_count', 0))
            elif hasattr(stats_obj, 'model_dump'):
                total_vecs = int(stats_obj.model_dump().get('total_vector_count', 0))
            else:
                total_vecs = 0
            logger.info(f"🎉 Successfully loaded and indexed document. Total vectors: {total_vecs}")
        except Exception:
            logger.info("🎉 Successfully loaded and indexed document. (Vector count unavailable)")
        return True
    except Exception as e:
        logger.error(f"❌ Error loading and indexing document from URL: {e}")
        raise
    finally:
        if temp_path.exists():
            os.remove(temp_path)
            logger.info(f"🗑️ Deleted temporary file: {temp_path}")
        if Path(temp_dir).exists():
            os.rmdir(temp_dir)
            logger.info(f"🗑️ Deleted temporary directory: {temp_dir}")

@app.post("/load-and-index", tags=["Document Management"])
async def load_and_index_document(request: dict, token: str = Depends(verify_token)):
    """Loads and indexes a document from a URL into the Pinecone index."""
    document_url = request.get("document_url")
    if not document_url:
        raise HTTPException(status_code=400, detail="Missing 'document_url' in request body.")

    start_time = time.time()
    try:
        await _load_and_index_from_url(document_url)
        
        return {
            "message": "Document loaded and indexed successfully.",
            "url": document_url,
            "processing_time": time.time() - start_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading document: {str(e)}")

@app.post("/query", tags=["Querying"])
async def process_query(request: QueryRequest, token: str = Depends(verify_token)) -> Dict[str, Any]:
    """Processes a single query against the indexed documents."""
    global index
    if index is None:
        raise HTTPException(status_code=404, detail="No document index available. Please load a document first.")
    logger.info(f"➡️ Received single query: '{request.query}'")
    start_time = time.time()
    
    query_engine = index.as_query_engine(
        similarity_top_k=max(3, config.TOP_K_RESULTS),
        response_mode="compact",
        node_postprocessors=[LongContextReorder()]
    )

    try:
        async def vector_task():
            try:
                return await query_engine.aquery(request.query)
            except Exception as e:
                logger.error(f"Vector query failed: {e}")
                return None

        async def graph_task():
            try:
                if pg_index is None:
                    return None
                pg_query_engine = pg_index.as_query_engine()
                if hasattr(pg_query_engine, "aquery"):
                    return await pg_query_engine.aquery(request.query)
                # Run sync query in thread to avoid blocking loop
                return await asyncio.to_thread(pg_query_engine.query, request.query)
            except Exception as e:
                logger.debug(f"Graph query skipped: {e}")
                return None

        vector_resp, graph_resp = await asyncio.gather(vector_task(), graph_task())
        if vector_resp is None or str(vector_resp).strip() == "":
            # Fallback: ask LLM directly with minimal prompt if vector returns empty
            llm = Settings.llm
            direct_answer = await llm.acomplete(request.query)
            vector_resp = direct_answer.text if hasattr(direct_answer, 'text') else str(direct_answer)
        response = vector_resp
        graph_snippets: List[str] = [str(graph_resp)] if graph_resp else []
        
        sources: List[str] = []
        confidence: float = 0.0
        try:
            source_nodes = response.source_nodes  # type: ignore[attr-defined]
            sources = [node.metadata.get('source', 'Unknown') for node in source_nodes]
            confidence = sum(node.score for node in source_nodes) / len(source_nodes) if source_nodes else 0.0
        except Exception:
            pass

        end_time = time.time()
        logger.info(f"✅ Query processed successfully in {end_time - start_time:.2f}s.")
        logger.debug(f"🔍 Sources found: {sources}")
        logger.debug(f"📊 Confidence score: {confidence:.2f}")

        combined_answer = str(response)
        if graph_snippets:
            combined_answer = combined_answer + "\n\n[Graph Insight]\n" + "\n".join(graph_snippets)

        result = {
            "answer": combined_answer,
            "confidence": confidence,
            "sources": sources,
            "reasoning": "Hybrid vector+graph retrieval with LlamaIndex.",
            "metadata": {}
        }

        # Emit Graphiti episode for the query (best-effort)
        try:
            await graphiti_client.send_episode(
                title="query",
                content=request.query,
                metadata={
                    "sources": sources,
                    "confidence": confidence,
                    "has_graph": bool(graph_snippets),
                },
            )
        except Exception as e:
            logger.debug(f"Graphiti episode write failed: {e}")

        return result
    except Exception as e:
        logger.error(f"❌ Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

class GraphQueryRequest(BaseModel):
    query: str

@app.post("/graph/query", tags=["Querying"])
async def graph_query(request: GraphQueryRequest, token: str = Depends(verify_token)) -> Dict[str, Any]:
    """Runs a graph-only query using the property graph index (Neo4j)."""
    if pg_index is None:
        raise HTTPException(status_code=503, detail="Graph index not available.")
    try:
        pg_query_engine = pg_index.as_query_engine()
        if hasattr(pg_query_engine, "aquery"):
            resp = await pg_query_engine.aquery(request.query)
        else:
            resp = await asyncio.to_thread(pg_query_engine.query, request.query)
        return {"answer": str(resp)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph query failed: {str(e)}")

@app.post("/api/v1/hackrx/run", tags=["Hackathon"])
async def process_batch_queries(request: BatchQueryRequest, token: str = Depends(verify_token)) -> HackRxResponse:
    """
    **Optimized for speed.** This endpoint now handles both document loading and batch querying
    using LlamaIndex's powerful query engine and parallel processing capabilities.
    """
    global index
    start_time = time.time()

    if request.documents:
        logger.info(f"📥 A new document URL was provided. Loading and indexing: {request.documents}")
        try:
            await _load_and_index_from_url(request.documents)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading document from URL: {str(e)}")
    else:
        try:
            index_stats = pc_client.Index(config.PINECONE_INDEX_NAME).describe_index_stats()
            if index_stats.total_vector_count == 0:
                raise HTTPException(status_code=404, detail="No document index available. Please use the 'documents' field to provide a URL for a document to be indexed.")
            logger.info(f"📂 Using existing document index with {index_stats.total_vector_count} vectors.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not connect to Pinecone index: {str(e)}")

    logger.info(f"➡️ Processing a batch of {len(request.questions)} questions.")

    if index is None:
        pinecone_vector_store = PineconeVectorStore(
            pinecone_client=pc_client,
            index_name=config.PINECONE_INDEX_NAME,
            environment=config.PINECONE_ENVIRONMENT
        )
        storage_context = StorageContext.from_defaults(vector_store=pinecone_vector_store)
        try:
            index = VectorStoreIndex.from_vector_store(pinecone_vector_store, storage_context=storage_context)
        except Exception as e:
            logger.warning(f"Vector index open failed: {e}; creating handle to empty index (will still query backend)")
            index = VectorStoreIndex([], storage_context=storage_context, show_progress=False)

    query_engine = index.as_query_engine(
        similarity_top_k=max(3, config.TOP_K_RESULTS),
        response_mode="compact"
    )

    async def process_single_question(question: str):
        logger.info(f"  - Starting query for question: '{question}'")
        try:
            response = await query_engine.aquery(question)
            logger.info(f"  - Finished query for question: '{question}'")
            return str(response)
        except Exception as e:
            logger.error(f"  - Failed to process question '{question}': {e}")
            return "Error processing this question."

    tasks = [process_single_question(q) for q in request.questions]
    answers = await asyncio.gather(*tasks)

    processing_time = time.time() - start_time
    logger.info(f"🎉 Batch query processing completed for {len(request.questions)} questions in {processing_time:.2f}s.")
    
    return HackRxResponse(answers=answers)
    
if __name__ == "__main__":
    # Use the PORT environment variable if available, otherwise default to 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable reload in production
        log_level=config.LOG_LEVEL.lower()
    )