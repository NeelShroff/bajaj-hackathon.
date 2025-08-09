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
import json
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
# Utility to detect effectively empty answers from upstream libraries
def _is_empty_answer_text(text: str) -> bool:
    try:
        normalized = (text or "").strip().lower()
        if normalized in {"", "empty response", "no answer", "n/a", "na", "none", "null"}:
            return True
        # Treat extremely short answers as empty/noise
        return len(normalized) < 3
    except Exception:
        return False


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
neo4j_pg_store: Optional[Neo4jPropertyGraphStore] = None
neo4j_driver = None
executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
pc_client = Pinecone(api_key=config.PINECONE_API_KEY)
graphiti_client = GraphitiClient()

def _check_if_document_exists_in_neo4j(source_url: str) -> bool:
    """Check if a document source already exists in Neo4j."""
    if neo4j_driver is None:
        return False
    try:
        with neo4j_driver.session(database=config.NEO4J_DATABASE) as session:
            result = session.run("MATCH (d:Document {source: $source}) RETURN d LIMIT 1", source=source_url).single()
            return result is not None
    except Exception:
        return False

async def _clear_all_existing_data():
    """Clear all existing data from Neo4j and Pinecone."""
    logger.info("🗑️ Clearing all existing data from Neo4j and Pinecone...")
    try:
        if neo4j_driver:
            async with neo4j_driver.session(database=config.NEO4J_DATABASE) as session:
                await session.run("MATCH (n) DETACH DELETE n").consume()
            logger.info("✅ All data cleared from Neo4j.")
        else:
            logger.warning("Neo4j driver not initialized, cannot clear Neo4j data.")
    except Exception as e:
        logger.error(f"❌ Failed to clear Neo4j data: {e}")

    try:
        # Re-create Pinecone index to ensure full clear
        if config.PINECONE_INDEX_NAME in pc_client.list_indexes().names():
            pc_client.delete_index(config.PINECONE_INDEX_NAME)
            logger.info(f"✅ Pinecone index '{config.PINECONE_INDEX_NAME}' deleted.")
        pc_client.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec={"serverless": {"cloud": "aws", "region": config.PINECONE_ENVIRONMENT}}
        )
        logger.info(f"✅ Pinecone index '{config.PINECONE_INDEX_NAME}' re-created.")
    except Exception as e:
        logger.error(f"❌ Failed to clear/re-create Pinecone index: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the application.
    This replaces the deprecated @app.on_event("startup") and "shutdown" decorators.
    """
    global index, pipeline, pg_index, neo4j_driver, neo4j_pg_store
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

            neo4j_pg_store = Neo4jPropertyGraphStore(
                username=config.NEO4J_USERNAME,
                password=config.NEO4J_PASSWORD,
                url=config.NEO4J_URI,
                database=config.NEO4J_DATABASE,
            )
            pg_index = PropertyGraphIndex.from_documents(
                documents=[],
                property_graph_store=neo4j_pg_store,
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

# --------- Neo4j helpers ---------
def _is_empty_answer(ans: Any) -> bool:
    try:
        s = str(ans).strip()
    except Exception:
        return True
    return s == "" or s.lower() == "empty response"

def _neo4j_ensure_indexes() -> None:
    if neo4j_driver is None:
        return
    try:
        with neo4j_driver.session(database=config.NEO4J_DATABASE) as session:
            # Neo4j 5 schema DDL with IF NOT EXISTS
            session.run(
                "CREATE CONSTRAINT doc_source IF NOT EXISTS FOR (d:Document) REQUIRE d.source IS UNIQUE"
            )
            session.run(
                "CREATE FULLTEXT INDEX pageTextIndex IF NOT EXISTS FOR (p:Page) ON EACH [p.text]"
            )
    except Exception as e:
        logger.debug(f"Neo4j index creation warning: {e}")

def _neo4j_upsert_pages(source_url: str, pages: List[Document]) -> None:
    if neo4j_driver is None:
        return
    try:
        with neo4j_driver.session(database=config.NEO4J_DATABASE) as session:
            for doc in pages:
                page_no = int(doc.metadata.get("page", 0)) if doc.metadata else 0
                text = doc.text or ""
                session.run(
                    (
                        "MERGE (d:Document {source: $source}) "
                        "ON CREATE SET d.createdAt = timestamp() "
                        "MERGE (p:Page {source: $source, page: $page}) "
                        "SET p.text = $text "
                        "MERGE (d)-[:HAS_PAGE]->(p)"
                    ),
                    source=source_url,
                    page=page_no,
                    text=text,
                )
    except Exception as e:
        logger.warning(f"Neo4j upsert pages failed: {e}")


# --------- LLM-driven KG extraction (entities + relations) ---------
def _sanitize_label(raw: str) -> str:
    s = ''.join(ch if ch.isalnum() or ch in ['_'] else '_' for ch in (raw or 'Entity'))
    if not s:
        return 'Entity'
    return s

def _safe_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    s = text.strip()
    if s.startswith('```'):  # strip fences
        s = s.strip('`')
        if s.startswith('json'):
            s = s[4:]
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        return {}

async def _llm_extract_graph_for_page(page_text: str, source: str, page_num: int) -> Dict[str, Any]:
    """Ask the LLM to extract entities and relationships for a page."""
    if not page_text or len(page_text) < 40:
        return {}
    # Trim to avoid huge prompts
    snippet = page_text[:3500]
    prompt = f"""
You are a precise information extraction system for insurance and policy documents. From the page content below, extract a knowledge graph focusing on key insurance terms, conditions, benefits, exclusions, and procedures.

Return ONLY JSON with keys: nodes, edges.

Schema:
- nodes: Array of objects with fields: name (string), type (string), properties (object, optional)
- edges: Array of objects with fields: source (string: node.name), target (string: node.name), relation (string), properties (object, optional)

Focus on extracting:
- Insurance terms: "Any one Illness", "Moratorium Period", "BMI", "Pre-existing conditions", etc.
- Policy features: "In-patient cash benefit", "International coverage", "Portability", "Migration", etc.
- Conditions and requirements: age limits, BMI thresholds, waiting periods, etc.
- Exclusions and limitations: what's not covered, restrictions, etc.
- Procedures and processes: how to claim, approval processes, etc.

Rules:
- Use exact terms as they appear in the document
- Extract specific numbers, amounts, durations, and thresholds
- Include relationships like "requires", "excludes", "covers", "applies_to", "has_threshold"
- Limit to at most 15 nodes and 25 edges per page
- Be precise and avoid hallucination

Page meta: source={source}, page={page_num}
Page text:
"""
    prompt += snippet
    llm = Settings.llm
    resp = await llm.acomplete(prompt)
    content = getattr(resp, 'text', str(resp))
    return _safe_json(content)

def _upsert_graph_json(graph: Dict[str, Any], source: str, page_num: int) -> None:
    if not graph or neo4j_driver is None:
        return
    nodes = graph.get('nodes') or []
    edges = graph.get('edges') or []
    try:
        with neo4j_driver.session(database=config.NEO4J_DATABASE) as session:
            # Upsert nodes
            for n in nodes:
                name = str(n.get('name', '')).strip()
                ntype = str(n.get('type', 'Entity')).strip() or 'Entity'
                props = n.get('properties') or {}
                if not name:
                    continue
                label = _sanitize_label(ntype)
                cy = (
                    f"MERGE (e:Entity:`{label}` {{source:$source, name:$name, type:$type}}) "
                    f"ON CREATE SET e.firstSeenAt=timestamp() "
                    f"SET e.lastSeenAt=timestamp(), e += $props "
                    f"WITH e "
                    f"MERGE (p:Page {{source:$source, page:$page}}) "
                    f"MERGE (e)-[:MENTIONED_IN]->(p)"
                )
                session.run(cy, source=source, name=name, type=ntype, props=props, page=page_num)
            # Upsert edges
            for e in edges:
                sname = str(e.get('source', '')).strip()
                tname = str(e.get('target', '')).strip()
                rel = _sanitize_label(str(e.get('relation', 'RELATED_TO')) or 'RELATED_TO')
                eprops = e.get('properties') or {}
                if not sname or not tname:
                    continue
                cy = (
                    "MATCH (a:Entity {source:$source, name:$sname}), (b:Entity {source:$source, name:$tname}) "
                    f"MERGE (a)-[r:`{rel}` {{source:$source}}]->(b) "
                    "SET r += $props"
                )
                session.run(cy, source=source, sname=sname, tname=tname, props=eprops)
    except Exception as e:
        logger.warning(f"Neo4j upsert KG failed: {e}")

async def _extract_and_upsert_graph(documents: List[Document], source: str) -> None:
    # Process more pages for better coverage - up to 60 pages or all pages if less
    max_pages = min(len(documents), 60)
    step = max(1, len(documents) // 20)  # Sample every ~3 pages for very long documents
    
    tasks = []
    for i in range(0, max_pages, step):
        d = documents[i]
        page_num = int(d.metadata.get('page', i + 1)) if d.metadata else (i + 1)
        tasks.append(_llm_extract_graph_for_page(d.text, source, page_num))
    
    logger.info(f"🔄 Extracting graph data from {len(tasks)} pages (sampling every {step} pages)")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successful_extractions = 0
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            logger.debug(f"KG extract skip page {i*step+1}: {res}")
            continue
        if res and (res.get('nodes') or res.get('edges')):
            page_num = int(documents[i*step].metadata.get('page', i*step + 1)) if documents[i*step].metadata else (i*step + 1)
            _upsert_graph_json(res, source, page_num)
            successful_extractions += 1
    
    logger.info(f"✅ Successfully extracted graph data from {successful_extractions}/{len(tasks)} pages")

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
        
        # Check if document already exists
        is_document_new = not _check_if_document_exists_in_neo4j(document_url)
        
        if is_document_new:
            logger.info(f"🆕 New document detected: {document_url}. Clearing old data and performing full ingestion.")
            await _clear_all_existing_data()
            # Persist in Neo4j synchronously to guarantee graph availability
            _neo4j_ensure_indexes()
            _neo4j_upsert_pages(document_url, documents)
            # Extract higher-level nodes/relations from each page (LLM) and upsert to Neo4j
            try:
                await _extract_and_upsert_graph(documents, document_url)
                logger.info("🧩 LLM-driven KG extracted and stored in Neo4j.")
            except Exception as e:
                logger.warning(f"KG extraction failed: {e}")
        else:
            logger.info(f"🧩 Document already ingested: {document_url}. Skipping LLM KG extraction.")
            # Still persist pages for consistency
            _neo4j_ensure_indexes()
            _neo4j_upsert_pages(document_url, documents)
        
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

        # Neo4j graph persisted via Cypher; skip PropertyGraphIndex refresh to avoid async issues
        logger.info("🧠 Document persisted to Neo4j via Cypher upsert.")

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
                if neo4j_driver is None:
                    return None
                
                # Use query processor for hints
                processed_query = await query_processor.process_query(request.query)
                keywords = processed_query["enhanced_extraction"].get("key_entities", [])
                
                results = []
                async with neo4j_driver.session(database=config.NEO4J_DATABASE) as session:
                    # Get the document source from Neo4j
                    source_result = await session.run("MATCH (d:Document) RETURN d.source AS source LIMIT 1").single()
                    source = source_result["source"] if source_result else ""
                    
                    # 1. Fulltext search on pages (most important)
                    try:
                        fulltext_result = await session.run(
                            "CALL db.index.fulltext.queryNodes('pageTextIndex', $q) YIELD node, score RETURN node.text, node.page, score ORDER BY score DESC LIMIT 8",
                            q=request.query
                        ).data()
                        if fulltext_result:
                            results.append(f"[Fulltext Search] Found {len(fulltext_result)} relevant pages")
                            for r in fulltext_result:
                                page_text = r.get('text', '')[:300] + "..." if len(r.get('text', '')) > 300 else r.get('text', '')
                                results.append(f"Page {r.get('page')} (score: {r.get('score', 0):.2f}): {page_text}")
                    except Exception as e:
                        results.append(f"[Fulltext Error] {e}")
                    
                    # 2. Entity search with flexible matching
                    if keywords:
                        for kw in keywords:
                            try:
                                # Flexible entity search
                                entity_result = await session.run(
                                    """
                                    MATCH (e:Entity)-[:MENTIONED_IN]->(p:Page)
                                    WHERE toLower(e.name) CONTAINS toLower($kw) 
                                       OR toLower(e.type) CONTAINS toLower($kw)
                                       OR toLower($kw) CONTAINS toLower(e.name)
                                       OR toLower(e.name) CONTAINS toLower($kw)
                                    RETURN e.name, e.type, p.page, p.text
                                    ORDER BY p.page
                                    LIMIT 15
                                    """,
                                    kw=kw
                                ).data()
                                if entity_result:
                                    results.append(f"[Entity Search for '{kw}'] Found {len(entity_result)} matches")
                                    for r in entity_result:
                                        page_text = r.get('text', '')[:200] + "..." if len(r.get('text', '')) > 200 else r.get('text', '')
                                        results.append(f"- {r.get('name')} ({r.get('type')}) on page {r.get('page')}: {page_text}")
                            except Exception as e:
                                results.append(f"[Entity Search Error for '{kw}']: {e}")
                    
                    # 3. Direct page search for key terms
                    try:
                        # Search for key insurance terms directly in page text
                        key_terms = ["Any one Illness", "Moratorium Period", "BMI", "In-patient cash benefit", 
                                   "International coverage", "Portability", "Migration", "Pre-approval", 
                                   "Grace period", "Waiting period", "Exclusions", "Coverage"]
                        
                        for term in key_terms:
                            if term.lower() in request.query.lower():
                                term_result = await session.run(
                                    """
                                    MATCH (p:Page)
                                    WHERE toLower(p.text) CONTAINS toLower($term)
                                    RETURN p.page, p.text
                                    ORDER BY p.page
                                    LIMIT 5
                                    """,
                                    term=term
                                ).data()
                                if term_result:
                                    results.append(f"[Key Term '{term}'] Found on {len(term_result)} pages")
                                    for r in term_result:
                                        page_text = r.get('text', '')[:250] + "..." if len(r.get('text', '')) > 250 else r.get('text', '')
                                        results.append(f"Page {r.get('page')}: {page_text}")
                                break  # Only search for the first matching term
                    except Exception as e:
                        results.append(f"[Key Term Search Error]: {e}")
                    
                    # 4. General entity overview if no specific results
                    if len(results) <= 1:  # Only debug info
                        try:
                            general_entities = await session.run(
                                """
                                MATCH (e:Entity)-[:MENTIONED_IN]->(p:Page)
                                RETURN e.name, e.type, p.page
                                ORDER BY p.page
                                LIMIT 25
                                """
                            ).data()
                            if general_entities:
                                results.append(f"[General Entities] Found {len(general_entities)} entities across pages")
                                for r in general_entities:
                                    results.append(f"- {r.get('name')} ({r.get('type')}) on page {r.get('page')}")
                        except Exception as e:
                            results.append(f"[General Entity Error]: {e}")
                
                return "\n".join(results) if results else None
            except Exception as e:
                logger.debug(f"Graph query skipped: {e}")
                return None

        vector_resp, graph_resp = await asyncio.gather(vector_task(), graph_task())
        if (vector_resp is None) or _is_empty_answer(vector_resp):
            # Fallback: ask LLM directly with minimal prompt if vector returns empty
            llm = Settings.llm
            direct_answer = await llm.acomplete(request.query)
            vector_resp = direct_answer.text if hasattr(direct_answer, 'text') else str(direct_answer)
        response = vector_resp
        graph_snippets: List[str] = []
        if graph_resp:
            _g = str(graph_resp).strip()
            if _g and _g.lower() != "empty response":
                graph_snippets.append(_g)
        
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
        if _is_empty_answer(combined_answer):
            combined_answer = "No relevant content found in the document."
        if graph_snippets:
            combined_answer = combined_answer + "\n\n[Graph Insight]\n" + "\n".join(graph_snippets)

        # If empty/placeholder, try to fetch a previous answer from Neo4j, then set a clear default
        try:
            if _is_empty_answer_text(combined_answer) and neo4j_driver is not None:
                def _get_answer_from_neo4j(q: str) -> str:
                    try:
                        with neo4j_driver.session(database=config.NEO4J_DATABASE) as session:
                            rec = session.execute_read(
                                lambda tx: tx.run(
                                    """
                                    MATCH (q:HackRxQuestion)-[:HAS_ANSWER]->(a:HackRxAnswer)
                                    WHERE toLower(q.text) = toLower($q)
                                       OR toLower(q.text) CONTAINS toLower($q)
                                       OR toLower($q) CONTAINS toLower(q.text)
                                    RETURN a.text AS text, coalesce(a.updatedAt, a.createdAt, 0) AS ts
                                    ORDER BY ts DESC
                                    LIMIT 1
                                    """,
                                    q=q,
                                ).single()
                            )
                            return rec["text"] if rec and rec.get("text") else ""
                    except Exception:
                        return ""
                prior = await asyncio.to_thread(_get_answer_from_neo4j, request.query)
                if not _is_empty_answer_text(prior):
                    combined_answer = prior
        except Exception:
            pass

        if _is_empty_answer_text(combined_answer):
            combined_answer = "No answer found."

        result = {
            "answer": combined_answer,
            "confidence": confidence,
            "sources": sources,
            "reasoning": "Hybrid vector+graph retrieval with LlamaIndex.",
            "metadata": {}
        }

        # Best-effort: persist Q&A to Neo4j
        try:
            if neo4j_driver is not None and not _is_empty_answer_text(combined_answer):
                def _write_to_neo4j(q: str, a: str):
                    with neo4j_driver.session(database=config.NEO4J_DATABASE) as session:
                        session.execute_write(
                            lambda tx: tx.run(
                                """
                                MERGE (q:HackRxQuestion {text: $q})
                                ON CREATE SET q.firstSeenAt = timestamp()
                                ON MATCH SET q.lastSeenAt = timestamp()
                                MERGE (ans:HackRxAnswer {text: $a})
                                ON CREATE SET ans.createdAt = timestamp()
                                ON MATCH SET ans.updatedAt = timestamp()
                                MERGE (q)-[r:HAS_ANSWER]->(ans)
                                RETURN 1
                                """,
                                q=q, a=a,
                            )
                        )
                await asyncio.to_thread(_write_to_neo4j, request.query, combined_answer)
        except Exception as e:
            logger.debug(f"Skipping Neo4j Q&A persist due to error: {e}")

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
            index_stats_obj = pc_client.Index(config.PINECONE_INDEX_NAME).describe_index_stats()
            # Normalize Pinecone stats to a dict and extract total_vector_count safely
            if isinstance(index_stats_obj, dict):
                total_vecs = int(index_stats_obj.get("total_vector_count", 0))
            elif hasattr(index_stats_obj, "to_dict"):
                total_vecs = int(index_stats_obj.to_dict().get("total_vector_count", 0))
            elif hasattr(index_stats_obj, "model_dump"):
                total_vecs = int(index_stats_obj.model_dump().get("total_vector_count", 0))
            else:
                total_vecs = 0
            if total_vecs == 0:
                raise HTTPException(status_code=404, detail="No document index available. Please use the 'documents' field to provide a URL for a document to be indexed.")
            logger.info(f"📂 Using existing document index with {total_vecs} vectors.")
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
        response_mode="compact",
        node_postprocessors=[LongContextReorder()],
    )

    async def process_single_question(question: str):
        logger.info(f"  - Starting query for question: '{question}'")

        try:
            # STEP 1: First fetch all context from graph database
            graph_context = ""
            try:
                if neo4j_driver is not None:
                    async with neo4j_driver.session(database=config.NEO4J_DATABASE) as session:
                        # Get the document source from Neo4j
                        source_result = await session.run("MATCH (d:Document) RETURN d.source AS source LIMIT 1").single()
                        source = source_result["source"] if source_result else ""
                        
                        # 1. Fulltext search on pages (most important)
                        try:
                            fulltext_result = await session.run(
                                "CALL db.index.fulltext.queryNodes('pageTextIndex', $q) YIELD node, score RETURN node.text, node.page, score ORDER BY score DESC LIMIT 10",
                                q=question
                            ).data()
                            if fulltext_result:
                                graph_context += f"[GRAPH CONTEXT - Fulltext Search] Found {len(fulltext_result)} relevant pages:\n"
                                for r in fulltext_result:
                                    page_text = r.get('text', '')[:400] + "..." if len(r.get('text', '')) > 400 else r.get('text', '')
                                    graph_context += f"Page {r.get('page')} (score: {r.get('score', 0):.2f}): {page_text}\n\n"
                        except Exception as e:
                            graph_context += f"[GRAPH ERROR - Fulltext]: {e}\n"
                        
                        # 2. Entity search with flexible matching
                        try:
                            # Use query processor for hints
                            processed_query = await query_processor.process_query(question)
                            keywords = processed_query["enhanced_extraction"].get("key_entities", [])
                            
                            if keywords:
                                for kw in keywords:
                                    entity_result = await session.run(
                                        """
                                        MATCH (e:Entity)-[:MENTIONED_IN]->(p:Page)
                                        WHERE toLower(e.name) CONTAINS toLower($kw) 
                                           OR toLower(e.type) CONTAINS toLower($kw)
                                           OR toLower($kw) CONTAINS toLower(e.name)
                                        RETURN e.name, e.type, p.page, p.text
                                        ORDER BY p.page
                                        LIMIT 15
                                        """,
                                        kw=kw
                                    ).data()
                                    if entity_result:
                                        graph_context += f"[GRAPH CONTEXT - Entity Search for '{kw}'] Found {len(entity_result)} matches:\n"
                                        for r in entity_result:
                                            page_text = r.get('text', '')[:300] + "..." if len(r.get('text', '')) > 300 else r.get('text', '')
                                            graph_context += f"- {r.get('name')} ({r.get('type')}) on page {r.get('page')}: {page_text}\n"
                                        graph_context += "\n"
                        except Exception as e:
                            graph_context += f"[GRAPH ERROR - Entity Search]: {e}\n"
                        
                        # 3. Direct page search for key insurance terms
                        try:
                            key_terms = ["Any one Illness", "Moratorium Period", "BMI", "In-patient cash benefit", 
                                       "International coverage", "Portability", "Migration", "Pre-approval", 
                                       "Grace period", "Waiting period", "Exclusions", "Coverage", "Hospital"]
                            
                            for term in key_terms:
                                if term.lower() in question.lower():
                                    term_result = await session.run(
                                        """
                                        MATCH (p:Page)
                                        WHERE toLower(p.text) CONTAINS toLower($term)
                                        RETURN p.page, p.text
                                        ORDER BY p.page
                                        LIMIT 8
                                        """,
                                        term=term
                                    ).data()
                                    if term_result:
                                        graph_context += f"[GRAPH CONTEXT - Key Term '{term}'] Found on {len(term_result)} pages:\n"
                                        for r in term_result:
                                            page_text = r.get('text', '')[:350] + "..." if len(r.get('text', '')) > 350 else r.get('text', '')
                                            graph_context += f"Page {r.get('page')}: {page_text}\n"
                                        graph_context += "\n"
                                    break  # Only search for the first matching term
                        except Exception as e:
                            graph_context += f"[GRAPH ERROR - Key Term Search]: {e}\n"
                        
                        # 4. Get general entity overview if no specific results
                        if len(graph_context.strip()) < 100:  # If we got very little context
                            try:
                                general_entities = await session.run(
                                    """
                                    MATCH (e:Entity)-[:MENTIONED_IN]->(p:Page)
                                    RETURN e.name, e.type, p.page
                                    ORDER BY p.page
                                    LIMIT 30
                                    """
                                ).data()
                                if general_entities:
                                    graph_context += f"[GRAPH CONTEXT - General Entities] Found {len(general_entities)} entities across pages:\n"
                                    for r in general_entities:
                                        graph_context += f"- {r.get('name')} ({r.get('type')}) on page {r.get('page')}\n"
                                    graph_context += "\n"
                            except Exception as e:
                                graph_context += f"[GRAPH ERROR - General Entities]: {e}\n"
            except Exception as e:
                graph_context = f"[GRAPH ERROR - Connection]: {e}\n"
            
            # STEP 2: Use graph context to generate LLM answer
            if graph_context.strip():
                # Create a comprehensive prompt with graph context
                llm_prompt = f"""
You are an expert insurance policy analyst. Answer the following question based on the provided graph context from the policy document.

QUESTION: {question}

GRAPH CONTEXT FROM POLICY DOCUMENT:
{graph_context}

INSTRUCTIONS:
1. Use ONLY the information provided in the graph context above
2. If the graph context contains relevant information, provide a detailed answer
3. If the graph context doesn't contain enough information, say "Based on the available policy information, I cannot provide a complete answer"
4. Be specific and cite page numbers when available
5. Use exact terms and definitions as they appear in the policy

ANSWER:
"""
                try:
                    llm = Settings.llm
                    llm_response = await llm.acomplete(llm_prompt)
                    answer = llm_response.text if hasattr(llm_response, "text") else str(llm_response)
                    
                    # Add graph context source information
                    if answer and not answer.strip().startswith("Based on the available policy information, I cannot provide a complete answer"):
                        answer += f"\n\n[Source: Graph database analysis with {len(graph_context.split('Page'))} relevant page sections]"
                    
                    return answer
                except Exception as e:
                    logger.error(f"LLM generation failed: {e}")
                    return f"Error generating answer: {e}"
            else:
                # STEP 3: Fallback to vector search if no graph context
                try:
                    vector_resp = await query_engine.aquery(question)
                    return str(vector_resp) if vector_resp else "No relevant information found in the policy document."
                except Exception as e:
                    logger.error(f"Vector query failed: {e}")
                    return f"Error retrieving information: {e}"

        except Exception as e:
            logger.error(f"  - Failed to process question '{question}': {e}")
            return "Error processing this question."

    tasks = [process_single_question(q) for q in request.questions]
    answers = await asyncio.gather(*tasks)

    processing_time = time.time() - start_time
    logger.info(f"🎉 Batch query processing completed for {len(request.questions)} questions in {processing_time:.2f}s.")
    
    return HackRxResponse(answers=answers)

@app.get("/debug/neo4j", tags=["Debug"])
async def debug_neo4j_data(token: str = Depends(verify_token)) -> Dict[str, Any]:
    """Debug endpoint to check Neo4j data."""
    if neo4j_driver is None:
        return {"error": "Neo4j driver not initialized"}
    
    try:
        async with neo4j_driver.session(database=config.NEO4J_DATABASE) as session:
            # Get document count
            doc_count = await session.run("MATCH (d:Document) RETURN count(d) as count").single()
            
            # Get page count
            page_count = await session.run("MATCH (p:Page) RETURN count(p) as count").single()
            
            # Get entity count
            entity_count = await session.run("MATCH (e:Entity) RETURN count(e) as count").single()
            
            # Get sample entities
            sample_entities = await session.run("MATCH (e:Entity) RETURN e.name, e.type LIMIT 20").data()
            
            # Get sample pages
            sample_pages = await session.run("MATCH (p:Page) RETURN p.page, substring(p.text, 0, 100) as text_preview LIMIT 5").data()
            
            return {
                "document_count": doc_count["count"] if doc_count else 0,
                "page_count": page_count["count"] if page_count else 0,
                "entity_count": entity_count["count"] if entity_count else 0,
                "sample_entities": sample_entities,
                "sample_pages": sample_pages
            }
    except Exception as e:
        return {"error": str(e)}
    
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