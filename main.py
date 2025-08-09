#!/usr/bin/env python3
"""
LLM-Powered Intelligent Document Query System - FastAPI Server
This server provides an API for document loading, querying, and batch processing.
Refactored for LlamaIndex with a focus on performance optimization.
"""
#main.py
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
import re # Added for _safe_json

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
            with neo4j_driver.session(database=config.NEO4J_DATABASE) as session:
                session.run("MATCH (n) DETACH DELETE n").consume()
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

def _cleanup_problematic_neo4j_data():
    """Clean up any problematic data in Neo4j that might cause issues."""
    if neo4j_driver is None:
        return
    
    try:
        with neo4j_driver.session(database=config.NEO4J_DATABASE) as session:
            # Try to use apoc.map.clean if available
            try:
                session.run("CALL apoc.map.clean($map, $keys, $values) YIELD value RETURN value", 
                          map={}, keys=[], values=[])
                logger.info("✅ APOC available, using apoc.map.clean for data cleanup")
                
                # Clean entities with problematic properties
                session.run("""
                    MATCH (e:Entity)
                    WHERE any(prop in keys(e) WHERE e[prop] IS NULL OR e[prop] = '' OR NOT e[prop] IS STRING)
                    DETACH DELETE e
                """)
                
                # Clean relationships with problematic properties
                session.run("""
                    MATCH ()-[r]-()
                    WHERE any(prop in keys(r) WHERE r[prop] IS NULL OR r[prop] = '' OR NOT r[prop] IS STRING)
                    DELETE r
                """)
                
            except Exception:
                # Fallback: delete entities with null or empty properties
                logger.info("⚠️ APOC not available, using fallback cleanup")
                
                # Delete entities with null or empty string properties
                session.run("""
                    MATCH (e:Entity)
                    WHERE any(prop in keys(e) WHERE e[prop] IS NULL OR e[prop] = '')
                    DETACH DELETE e
                """)
                
                # Delete relationships with null or empty string properties
                session.run("""
                    MATCH ()-[r]-()
                    WHERE any(prop in keys(r) WHERE r[prop] IS NULL OR r[prop] = '')
                    DELETE r
                """)
                
            logger.info("✅ Neo4j data cleanup completed")
            
    except Exception as e:
        logger.warning(f"Failed to cleanup Neo4j data: {e}")

def _clear_all_neo4j_data():
    """Clear all data from Neo4j to ensure a clean slate."""
    if neo4j_driver is None:
        return
    
    try:
        with neo4j_driver.session(database=config.NEO4J_DATABASE) as session:
            session.run("MATCH (n) DETACH DELETE n").consume()
            logger.info("🗑️ Cleared all Neo4j data for clean startup")
    except Exception as e:
        logger.warning(f"Failed to clear Neo4j data: {e}")

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
                
                # Clear all Neo4j data for clean startup
                _clear_all_neo4j_data()
                
                # Clean up any problematic data that might cause StringArray errors
                _cleanup_problematic_neo4j_data()
                
            except Exception as e:
                logger.warning(f"Neo4j connectivity check failed: {e}")

            # Initialize Neo4j property graph store
            neo4j_pg_store = Neo4jPropertyGraphStore(
                username=config.NEO4J_USERNAME,
                password=config.NEO4J_PASSWORD,
                url=config.NEO4J_URI,
                database=config.NEO4J_DATABASE,
            )
            
            # Try to create property graph index, but handle existing data issues gracefully
            try:
                pg_index = PropertyGraphIndex.from_documents(
                    documents=[],
                    property_graph_store=neo4j_pg_store,
                    embed_model=Settings.embed_model,
                    show_progress=False,
                )
                logger.info("✨ Neo4j graph store initialized.")
            except Exception as pg_error:
                logger.warning(f"PropertyGraphIndex initialization failed: {pg_error}")
                logger.info("🔄 Falling back to basic Neo4j store without PropertyGraphIndex")
                pg_index = None
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize Neo4j graph store: {e}")
            neo4j_driver = None
            neo4j_pg_store = None
            pg_index = None

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

def _neo4j_upsert_pages(pages: List[Dict[str, Any]], source: str) -> None:
    """Upsert pages to Neo4j with better constraint handling."""
    if not pages or neo4j_driver is None:
        return
    
    try:
        with neo4j_driver.session(database=config.NEO4J_DATABASE) as session:
            # First, try to create the document node with ON CREATE
            try:
                session.run(
                    "MERGE (d:Document {source: $source}) ON CREATE SET d.createdAt = timestamp() SET d.updatedAt = timestamp()",
                    source=source
                )
            except Exception as e:
                logger.warning(f"Document node creation failed: {e}")
            
            # Then upsert pages with better error handling
            for page_data in pages:
                page_num = page_data.get('page_num', 0)
                text = page_data.get('text', '')
                
                if not text.strip():
                    continue
                
                try:
                    # Use MERGE to avoid constraint violations
                    session.run(
                        """
                        MERGE (p:Page {source: $source, page: $page})
                        ON CREATE SET p.text = $text, p.createdAt = timestamp()
                        SET p.updatedAt = timestamp(), p.text = $text
                        """,
                        source=source, page=page_num, text=text
                    )
                    
                    # Create relationship to document
                    session.run(
                        """
                        MATCH (d:Document {source: $source})
                        MATCH (p:Page {source: $source, page: $page})
                        MERGE (d)-[:HAS_PAGE]->(p)
                        """,
                        source=source, page=page_num
                    )
                except Exception as e:
                    logger.warning(f"Page upsert failed for page {page_num}: {e}")
                    
    except Exception as e:
        logger.error(f"Neo4j pages upsert failed: {e}")


# --------- LLM-driven KG extraction (entities + relations) ---------
def _sanitize_label(raw: str) -> str:
    s = ''.join(ch if ch.isalnum() or ch in ['_'] else '_' for ch in (raw or 'Entity'))
    if not s:
        return 'Entity'
    return s

def _safe_json(text: str) -> Dict[str, Any]:
    """Safely parse JSON from LLM response, with fallback handling."""
    if not text or not text.strip():
        return {}
    
    # Clean the text
    text = text.strip()
    
    # Try to extract JSON from markdown code blocks
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()
    
    # Remove any leading/trailing text that's not JSON
    text = text.strip()
    if text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    
    try:
        # Try direct JSON parsing
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        else:
            logger.warning(f"LLM response is not a dict: {type(result)}")
            return {}
    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode error: {e}")
        
        # Try to fix common JSON issues
        try:
            # Remove any trailing commas
            text = re.sub(r',(\s*[}\]])', r'\1', text)
            # Fix single quotes to double quotes
            text = re.sub(r"'([^']*)'", r'"\1"', text)
            # Fix unquoted keys
            text = re.sub(r'(\w+):', r'"\1":', text)
            
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except:
            pass
        
        # Last resort: try to extract basic structure
        try:
            # Look for nodes and edges patterns
            nodes_match = re.search(r'"nodes":\s*\[(.*?)\]', text, re.DOTALL)
            edges_match = re.search(r'"edges":\s*\[(.*?)\]', text, re.DOTALL)
            
            result = {}
            if nodes_match:
                result['nodes'] = []
            if edges_match:
                result['edges'] = []
            
            return result
        except:
            logger.error(f"Failed to parse LLM response: {text[:200]}...")
            return {}

async def _llm_extract_graph_for_page(page_text: str, source: str, page_num: int) -> Dict[str, Any]:
    """Extract knowledge graph from page text using LLM for ANY type of document."""
    try:
        llm = Settings.llm
        
        prompt = f"""
You are a precise information extraction system for documents of any type (scientific papers, legal documents, technical manuals, academic research, policy documents, etc.). From the page content below, extract a knowledge graph focusing on key concepts, entities, relationships, and important information.

Return ONLY JSON with keys: nodes, edges.

Schema:
- nodes: Array of objects with fields: name (string), type (string), properties (object, optional)
- edges: Array of objects with fields: source (string: node.name), target (string: node.name), relation (string), properties (object, optional)

Focus on extracting:
- Key terms and definitions: Important concepts, technical terms, definitions, formulas
- Entities and objects: People, organizations, locations, products, systems, processes
- Relationships and connections: How entities relate to each other, dependencies, hierarchies
- Conditions and requirements: Rules, criteria, thresholds, specifications
- Procedures and processes: Steps, methods, workflows, algorithms
- Measurements and data: Numbers, units, statistics, parameters
- Categories and classifications: Types, categories, groups, classifications
- Exceptions and limitations: What's excluded, restrictions, caveats

Rules:
- Use exact terms as they appear in the document
- Extract specific numbers, amounts, measurements, and thresholds
- Include relationships like "requires", "depends_on", "contains", "defines", "measures", "classifies"
- Limit to at most 20 nodes and 30 edges per page
- Be precise and avoid hallucination
- Adapt to the document type (scientific, legal, technical, etc.)
- IMPORTANT: All property values must be single strings, not arrays or lists
- If you have multiple values, combine them into a single comma-separated string
- CRITICAL: NEVER create nodes with type "Document", "Doc", "File", or "PDF" - these are reserved
- Only create nodes for actual content entities, concepts, terms, etc.

Page meta: source={source}, page={page_num}
Page text:
{page_text[:4000]}  # Limit to first 4000 chars to avoid token limits
"""
        
        response = await llm.acomplete(prompt)
        response_text = response.text if hasattr(response, "text") else str(response)
        
        # Clean and parse JSON
        json_str = _safe_json(response_text)
        if not json_str:
            return {"nodes": [], "edges": []}
        
        return json_str
        
    except Exception as e:
        logger.error(f"Failed to extract graph for page {page_num}: {e}")
        return {"nodes": [], "edges": []}

def _upsert_graph_json(graph: Dict[str, Any], source: str, page_num: int) -> None:
    if not graph or neo4j_driver is None:
        return
    nodes = graph.get('nodes') or []
    edges = graph.get('edges') or []
    try:
        with neo4j_driver.session(database=config.NEO4J_DATABASE) as session:
            # Upsert nodes with better constraint handling
            for n in nodes:
                name = str(n.get('name', '')).strip()
                ntype = str(n.get('type', 'Entity')).strip() or 'Entity'
                props = n.get('properties') or {}
                
                # Ensure props is a dictionary
                if not isinstance(props, dict):
                    props = {}
                
                # Convert any arrays in properties to strings
                cleaned_props = {}
                for key, value in props.items():
                    if isinstance(value, list):
                        cleaned_props[key] = ', '.join(str(item) for item in value)
                    else:
                        cleaned_props[key] = str(value) if value is not None else ''
                
                if not name:
                    continue
                
                # CRITICAL: Never create Document nodes - only Entity nodes
                # If the LLM extracted a Document node, skip it or convert to Entity
                if ntype.lower() in ['document', 'doc', 'file', 'pdf']:
                    logger.debug(f"Skipping Document node: {name}")
                    continue
                
                # Use a safe label that won't conflict with existing constraints
                label = _sanitize_label(ntype)
                if label.lower() in ['document', 'doc', 'file', 'pdf']:
                    label = 'Entity'  # Fallback to Entity label
                
                try:
                    # Use MERGE with ON CREATE to avoid constraint violations
                    # Only create Entity nodes, never Document nodes
                    cy = (
                        f"MERGE (e:Entity {{source:$source, name:$name, type:$type}}) "
                        f"ON CREATE SET e.firstSeenAt=timestamp(), e += $props "
                        f"SET e.lastSeenAt=timestamp(), e += $props "
                        f"WITH e "
                        f"MATCH (p:Page {{source:$source, page:$page}}) "
                        f"MERGE (e)-[:MENTIONED_IN]->(p)"
                    )
                    session.run(cy, source=source, name=name, type=ntype, props=cleaned_props, page=page_num)
                except Exception as e:
                    logger.warning(f"Node upsert failed for {name}: {e}")
            
            # Upsert edges with better relationship handling
            for e in edges:
                sname = str(e.get('source', '')).strip()
                tname = str(e.get('target', '')).strip()
                rel = _sanitize_label(str(e.get('relation', 'RELATED_TO')) or 'RELATED_TO')
                eprops = e.get('properties') or {}
                
                # Ensure eprops is a dictionary
                if not isinstance(eprops, dict):
                    eprops = {}
                
                # Convert any arrays in edge properties to strings
                cleaned_eprops = {}
                for key, value in eprops.items():
                    if isinstance(value, list):
                        cleaned_eprops[key] = ', '.join(str(item) for item in value)
                    else:
                        cleaned_eprops[key] = str(value) if value is not None else ''
                
                if not sname or not tname or sname == tname:
                    continue
                
                try:
                    # Use a more efficient approach to avoid cartesian product
                    # First check if both nodes exist, then create relationship
                    cy = (
                        f"MATCH (a:Entity {{source:$source, name:$sname}}) "
                        f"MATCH (b:Entity {{source:$source, name:$tname}}) "
                        f"WHERE a <> b "
                        f"WITH a, b "
                        f"MERGE (a)-[r:{rel} {{source:$source}}]->(b) "
                        f"SET r += $props"
                    )
                    session.run(cy, source=source, sname=sname, tname=tname, props=cleaned_eprops)
                except Exception as e:
                    logger.warning(f"Edge upsert failed for {sname}->{tname}: {e}")
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
    """Load and index document from URL with automatic data replacement."""
    logger.info(f"📥 Loading document from URL: {document_url}")
    
    # ALWAYS clear existing data when loading a new document
    logger.info("🗑️ Clearing all existing data to prepare for new document...")
    await _clear_all_existing_data()
    
    try:
        # Download the document
        response = requests.get(document_url, timeout=30)
        response.raise_for_status()
        
        # Save to temporary file
        temp_file = f"temp_document_{int(time.time())}.pdf"
        with open(temp_file, "wb") as f:
            f.write(response.content)
        
        logger.info(f"📄 Document downloaded successfully: {len(response.content)} bytes")
        
        # Extract text from PDF using PyMuPDF
        documents = []
        try:
            import fitz  # PyMuPDF
            pdf_document = fitz.open(temp_file)
            total_pages = len(pdf_document)
            logger.info(f"📖 PDF has {total_pages} pages")
            
            # Process pages in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for page_num in range(total_pages):
                    future = executor.submit(_thread_process_pdf_page_with_pymupdf_first, temp_file, page_num)
                    futures.append((page_num, future))
                
                for page_num, future in futures:
                    try:
                        page_text = future.result(timeout=30)
                        if page_text.strip():
                            doc = Document(text=page_text, metadata={"page": page_num + 1, "source": document_url})
                            documents.append(doc)
                            logger.info(f"✅ Page {page_num + 1} processed successfully")
                        else:
                            logger.warning(f"⚠️ Page {page_num + 1} is empty")
                    except Exception as e:
                        logger.error(f"❌ Failed to process page {page_num + 1}: {e}")
            
            pdf_document.close()
            
        except Exception as e:
            logger.error(f"❌ PDF processing failed: {e}")
            raise
        
        finally:
            # Clean up temporary file
            try:
                os.remove(temp_file)
            except:
                pass
        
        if not documents:
            raise ValueError("No text content extracted from PDF")
        
        logger.info(f"📚 Extracted {len(documents)} pages with content")
        
        # Persist in Neo4j synchronously to guarantee graph availability
        _neo4j_ensure_indexes()
        
        # Convert documents to the expected format
        pages_data = []
        for doc in documents:
            page_no = int(doc.metadata.get("page", 0)) if doc.metadata else 0
            text = doc.text or ""
            pages_data.append({
                'page_num': page_no,
                'text': text
            })
        
        _neo4j_upsert_pages(pages_data, document_url)
        
        # Extract higher-level nodes/relations from each page (LLM) and upsert to Neo4j
        try:
            await _extract_and_upsert_graph(documents, document_url)
            logger.info("🧩 LLM-driven KG extracted and stored in Neo4j.")
        except Exception as e:
            logger.warning(f"KG extraction failed: {e}")
        
        # Create vector store and index
        vector_store = PineconeVectorStore(
            pinecone_index=pc_client.Index(config.PINECONE_INDEX_NAME),
            text_key="text"
        )
        
        # Create document store
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        
        # Create query engine
        global query_engine
        query_engine = index.as_query_engine(
            similarity_top_k=5,
            response_mode="compact"
        )
        
        logger.info("✅ Document indexing completed successfully")
        
        # Send Graphiti episode
        if config.GRAPHITI_ENABLED:
            try:
                await graphiti_client.send_episode(
                    "document_ingested",
                    {
                        "document_url": document_url,
                        "pages_processed": len(documents),
                        "total_pages": total_pages,
                        "timestamp": time.time()
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to send Graphiti episode: {e}")
        
        return {"status": "success", "pages_processed": len(documents)}
        
    except Exception as e:
        logger.error(f"❌ Document loading failed: {e}")
        raise

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

    async def vector_task():
        try:
            return await query_engine.aquery(request.query)
        except Exception as e:
            logger.error(f"Vector query failed: {e}")
            return None

    try:
        # STEP 1: First fetch all context from graph database
        graph_context = ""
        try:
            if neo4j_driver is not None:
                # Use synchronous session instead of async
                with neo4j_driver.session(database=config.NEO4J_DATABASE) as session:
                    # Get the document source from Neo4j
                    source_result = session.run("MATCH (d:Document) RETURN d.source AS source LIMIT 1").single()
                    source = source_result["source"] if source_result else ""
                    
                    # 1. Fulltext search on pages (most important)
                    try:
                        fulltext_result = session.run(
                            "CALL db.index.fulltext.queryNodes('pageTextIndex', $q) YIELD node, score RETURN node.text, node.page, score ORDER BY score DESC LIMIT 15",
                            q=request.query
                        ).data()
                        if fulltext_result:
                            graph_context += f"[GRAPH CONTEXT - Fulltext Search] Found {len(fulltext_result)} relevant pages:\n"
                            for r in fulltext_result:
                                page_text = r.get('text', '')[:800] + "..." if len(r.get('text', '')) > 800 else r.get('text', '')
                                graph_context += f"Page {r.get('page')} (score: {r.get('score', 0):.2f}): {page_text}\n\n"
                    except Exception as e:
                        graph_context += f"[GRAPH ERROR - Fulltext]: {e}\n"
                    
                    # 2. Entity search with flexible matching
                    try:
                        # Use query processor for hints
                        processed_query = await query_processor.process_query(request.query)
                        keywords = processed_query["enhanced_extraction"].get("key_entities", [])
                        
                        if keywords:
                            for kw in keywords:
                                entity_result = session.run(
                                    """
                                    MATCH (e:Entity)-[:MENTIONED_IN]->(p:Page)
                                    WHERE toLower(e.name) CONTAINS toLower($kw) 
                                       OR toLower(e.type) CONTAINS toLower($kw)
                                       OR toLower($kw) CONTAINS toLower(e.name)
                                       OR toLower(e.name) CONTAINS toLower($kw)
                                    RETURN e.name, e.type, p.page, p.text
                                    ORDER BY p.page
                                    LIMIT 20
                                    """,
                                    kw=kw
                                ).data()
                                if entity_result:
                                    graph_context += f"[GRAPH CONTEXT - Entity Search for '{kw}'] Found {len(entity_result)} matches:\n"
                                    for r in entity_result:
                                        page_text = r.get('text', '')[:600] + "..." if len(r.get('text', '')) > 600 else r.get('text', '')
                                        graph_context += f"- {r.get('name')} ({r.get('type')}) on page {r.get('page')}: {page_text}\n"
                                    graph_context += "\n"
                    except Exception as e:
                        graph_context += f"[GRAPH ERROR - Entity Search]: {e}\n"
                    
                    # 3. Direct page search for key terms (universal for any document type)
                    try:
                        # Extract key terms from the question itself
                        question_lower = request.query.lower()
                        key_terms = []
                        
                        # Common terms that might appear in any document
                        universal_terms = ["definition", "formula", "equation", "method", "procedure", "process", "system", 
                                         "analysis", "study", "research", "experiment", "test", "measurement", "data",
                                         "result", "conclusion", "finding", "evidence", "proof", "theory", "model",
                                         "algorithm", "function", "variable", "parameter", "condition", "requirement",
                                         "specification", "standard", "protocol", "guideline", "rule", "law", "policy",
                                         "regulation", "statute", "clause", "section", "article", "chapter", "part"]
                        
                        # Add terms that appear in the question
                        for term in universal_terms:
                            if term in question_lower:
                                key_terms.append(term)
                        
                        # If no specific terms found, search for common document elements
                        if not key_terms:
                            key_terms = ["definition", "formula", "method", "procedure", "analysis"]
                        
                        for term in key_terms:
                            term_result = session.run(
                                """
                                MATCH (p:Page)
                                WHERE toLower(p.text) CONTAINS toLower($term)
                                RETURN p.page, p.text
                                ORDER BY p.page
                                LIMIT 12
                                """,
                                term=term
                            ).data()
                            if term_result:
                                graph_context += f"[GRAPH CONTEXT - Key Term '{term}'] Found on {len(term_result)} pages:\n"
                                for r in term_result:
                                    page_text = r.get('text', '')[:700] + "..." if len(r.get('text', '')) > 700 else r.get('text', '')
                                    graph_context += f"Page {r.get('page')}: {page_text}\n"
                                graph_context += "\n"
                            break  # Only search for the first matching term
                    except Exception as e:
                        graph_context += f"[GRAPH ERROR - Key Term Search]: {e}\n"
                    
                    # 4. Get general entity overview if no specific results
                    if len(graph_context.strip()) < 100:  # If we got very little context
                        try:
                            general_entities = session.run(
                                """
                                MATCH (e:Entity)-[:MENTIONED_IN]->(p:Page)
                                RETURN e.name, e.type, p.page
                                ORDER BY p.page
                                LIMIT 25
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
            # Create a very aggressive prompt that forces information extraction
            llm_prompt = f"""
You are an expert document analyst. Your task is to extract and present ANY relevant information from the provided document context.

QUESTION: {request.query}

DOCUMENT CONTEXT:
{graph_context}

CRITICAL INSTRUCTIONS:
- You MUST extract and present ANY relevant information you can find in the context
- Do NOT say "cannot provide complete answer" unless the context has ZERO relevant information
- If you see ANY text that relates to the question, present it as an answer
- Look for keywords, definitions, numbers, dates, conditions, requirements, etc.
- Even if information is incomplete, present what you find
- Be aggressive in finding and presenting relevant content
- Use exact quotes from the context when available
- Cite page numbers when mentioned in the context
- If you find partial information, present it and note what's missing
- The goal is to provide useful information, not perfect completeness

ANSWER (be aggressive in extracting information):
"""
            try:
                llm = Settings.llm
                llm_response = await llm.acomplete(llm_prompt)
                answer = llm_response.text if hasattr(llm_response, "text") else str(llm_response)
                
                # Check if the answer is too generic and try vector search as backup
                if "cannot provide complete answer" in answer.lower() or "does not contain specific information" in answer.lower() or "does not contain specific" in answer.lower() or len(answer.strip()) < 100:
                    logger.info("🔄 Graph context insufficient, trying vector search as backup...")
                    try:
                        vector_resp = await query_engine.aquery(request.query)
                        vector_answer = str(vector_resp) if vector_resp else ""
                        if vector_answer and len(vector_answer.strip()) > 50:
                            answer = f"{answer}\n\n[Additional information from document search:]\n{vector_answer}"
                            logger.info("✅ Enhanced answer with vector search results")
                    except Exception as e:
                        logger.warning(f"Vector search backup failed: {e}")
                
                # Add graph context source information
                if answer and not answer.strip().startswith("Based on the available document information, I cannot provide a complete answer"):
                    page_count = len([line for line in graph_context.split('\n') if 'Page ' in line])
                    answer += f"\n\n[Source: Graph database analysis with {page_count} relevant page sections]"
                
                response = answer
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                response = f"Error generating answer: {e}"
        else:
            # STEP 3: Fallback to vector search if no graph context
            try:
                vector_resp = await query_engine.aquery(request.query)
                response = str(vector_resp) if vector_resp else "No relevant information found in the document."
            except Exception as e:
                logger.error(f"Vector query failed: {e}")
                response = f"Error retrieving information: {e}"

        end_time = time.time()
        logger.info(f"✅ Query processed successfully in {end_time - start_time:.2f}s.")

        result = {
            "answer": response,
            "confidence": 0.9 if graph_context.strip() else 0.5,
            "sources": ["Graph Database" if graph_context.strip() else "Vector Search"],
            "reasoning": "Graph-first retrieval with LLM generation based on structured context.",
            "metadata": {"graph_context_length": len(graph_context)}
        }

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

    async def process_single_question(question: str) -> str:
        """Process a single question using PARALLEL semantic and graph search for faster response times."""
        try:
            # PARALLEL PROCESSING: Run semantic and graph search simultaneously
            logger.info(f"🚀 Starting PARALLEL search for: {question}")
            
            async def semantic_search_task():
                """Semantic search task running in parallel."""
                try:
                    vector_resp = await query_engine.aquery(question)
                    semantic_answer = str(vector_resp) if vector_resp else ""
                    logger.info(f"✅ Semantic search completed, found {len(semantic_answer)} characters")
                    return semantic_answer
                except Exception as e:
                    logger.warning(f"❌ Semantic search failed: {e}")
                    return ""
            
            async def graph_search_task():
                """Graph search task running in parallel."""
                graph_context = ""
                if neo4j_driver is not None:
                    try:
                        with neo4j_driver.session(database=config.NEO4J_DATABASE) as session:
                            # Get document source
                            doc_result = session.run("MATCH (d:Document) RETURN d.source LIMIT 1").data()
                            if doc_result:
                                source = doc_result[0].get('source', '')
                                logger.info(f"📄 Found document source: {source[:50]}...")
                                
                                # 1. Fulltext search on pages
                                try:
                                    fulltext_result = session.run(
                                        "CALL db.index.fulltext.queryNodes('pageTextIndex', $q) YIELD node, score RETURN node.text, node.page, score ORDER BY score DESC LIMIT 10",
                                        q=question
                                    ).data()
                                    if fulltext_result:
                                        logger.info(f"✅ Fulltext search found {len(fulltext_result)} relevant pages")
                                        graph_context += f"[GRAPH CONTEXT - Fulltext Search] Found {len(fulltext_result)} relevant pages:\n"
                                        for r in fulltext_result:
                                            page_text = r.get('text', '')[:600] + "..." if len(r.get('text', '')) > 600 else r.get('text', '')
                                            graph_context += f"Page {r.get('page')} (score: {r.get('score', 0):.2f}): {page_text}\n\n"
                                except Exception as e:
                                    logger.error(f"❌ Fulltext search error: {e}")
                                
                                # 2. Entity search
                                try:
                                    processed_query = await query_processor.process_query(question)
                                    keywords = processed_query["enhanced_extraction"].get("key_entities", [])
                                    logger.info(f"🔍 Extracted keywords: {keywords}")
                                    
                                    if keywords:
                                        for kw in keywords:
                                            entity_result = session.run(
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
                                                logger.info(f"✅ Entity search for '{kw}' found {len(entity_result)} matches")
                                                graph_context += f"[GRAPH CONTEXT - Entity Search for '{kw}'] Found {len(entity_result)} matches:\n"
                                                for r in entity_result:
                                                    page_text = r.get('text', '')[:400] + "..." if len(r.get('text', '')) > 400 else r.get('text', '')
                                                    graph_context += f"- {r.get('name')} ({r.get('type')}) on page {r.get('page')}: {page_text}\n"
                                                graph_context += "\n"
                                except Exception as e:
                                    logger.error(f"❌ Entity search error: {e}")
                                
                                # 3. Key term search
                                try:
                                    question_lower = question.lower()
                                    key_terms = ["definition", "formula", "method", "procedure", "policy", "condition", "requirement"]
                                    
                                    for term in key_terms:
                                        if term in question_lower:
                                            term_result = session.run(
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
                                                logger.info(f"✅ Key term '{term}' found on {len(term_result)} pages")
                                                graph_context += f"[GRAPH CONTEXT - Key Term '{term}'] Found on {len(term_result)} pages:\n"
                                                for r in term_result:
                                                    page_text = r.get('text', '')[:500] + "..." if len(r.get('text', '')) > 500 else r.get('text', '')
                                                    graph_context += f"Page {r.get('page')}: {page_text}\n"
                                                graph_context += "\n"
                                            break
                                except Exception as e:
                                    logger.error(f"❌ Key term search error: {e}")
                                    
                    except Exception as e:
                        logger.error(f"❌ Graph search error: {e}")
                
                return graph_context
            
            # RUN BOTH TASKS IN PARALLEL
            semantic_task = semantic_search_task()
            graph_task = graph_search_task()
            
            # Wait for both tasks to complete
            semantic_answer, graph_context = await asyncio.gather(semantic_task, graph_task, return_exceptions=True)
            
            # Handle exceptions from parallel tasks
            if isinstance(semantic_answer, Exception):
                logger.error(f"❌ Semantic search task failed: {semantic_answer}")
                semantic_answer = ""
            if isinstance(graph_context, Exception):
                logger.error(f"❌ Graph search task failed: {graph_context}")
                graph_context = ""
            
            # STEP 3: Combine results and generate final answer
            if graph_context.strip():
                # Use graph context to enhance the answer
                llm_prompt = f"""
QUESTION: {question}

SEMANTIC SEARCH RESULT:
{semantic_answer}

GRAPH DATABASE CONTEXT:
{graph_context}

CRITICAL INSTRUCTIONS:
- You MUST extract and present ANY relevant information you can find in the context
- Do NOT say "cannot provide complete answer" unless the context has ZERO relevant information
- If you see ANY text that relates to the question, present it as an answer
- Look for keywords, definitions, numbers, dates, conditions, requirements, etc.
- Even if information is incomplete, present what you find
- Be aggressive in finding and presenting relevant content
- Use exact quotes from the context when available
- Cite page numbers when mentioned in the context
- If you find partial information, present it and note what's missing
- The goal is to provide useful information, not perfect completeness

ANSWER (be aggressive in extracting information):
"""
                try:
                    llm = Settings.llm
                    llm_response = await llm.acomplete(llm_prompt)
                    answer = llm_response.text if hasattr(llm_response, "text") else str(llm_response)
                    
                    # Check if the answer is too generic and try vector search as backup
                    if "cannot provide complete answer" in answer.lower() or "does not contain specific information" in answer.lower() or "does not contain specific" in answer.lower() or len(answer.strip()) < 100:
                        logger.info("🔄 Graph context insufficient, trying vector search as backup...")
                        if semantic_answer and len(semantic_answer.strip()) > 50:
                            answer = f"{answer}\n\n[Additional information from semantic search:]\n{semantic_answer}"
                            logger.info("✅ Enhanced answer with semantic search results")
                    
                    # Add graph context source information
                    if answer and not answer.strip().startswith("Based on the available document information, I cannot provide a complete answer"):
                        page_count = len([line for line in graph_context.split('\n') if 'Page ' in line])
                        answer += f"\n\n[Source: Parallel search - Graph database ({page_count} pages) + Semantic search]"
                    
                    return answer
                except Exception as e:
                    logger.error(f"LLM generation failed: {e}")
                    return f"Error generating answer: {e}"
            else:
                # Fallback to semantic search only
                if semantic_answer:
                    return f"{semantic_answer}\n\n[Source: Semantic search only - Graph search returned no results]"
                else:
                    return "No relevant information found in the document."
                    
        except Exception as e:
            logger.error(f"❌ Error processing question: {e}")
            return f"Error processing question: {e}"

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
        with neo4j_driver.session(database=config.NEO4J_DATABASE) as session:
            # Get document count
            doc_count = session.run("MATCH (d:Document) RETURN count(d) as count").single()
            
            # Get page count
            page_count = session.run("MATCH (p:Page) RETURN count(p) as count").single()
            
            # Get entity count
            entity_count = session.run("MATCH (e:Entity) RETURN count(e) as count").single()
            
            # Get sample entities
            sample_entities = session.run("MATCH (e:Entity) RETURN e.name, e.type LIMIT 20").data()
            
            # Get sample pages
            sample_pages = session.run("MATCH (p:Page) RETURN p.page, substring(p.text, 0, 100) as text_preview LIMIT 5").data()
            
            return {
                "document_count": doc_count["count"] if doc_count else 0,
                "page_count": page_count["count"] if page_count else 0,
                "entity_count": entity_count["count"] if entity_count else 0,
                "sample_entities": sample_entities,
                "sample_pages": sample_pages
            }
    except Exception as e:
        return {"error": str(e)}

@app.get("/test/graph-context", tags=["Debug"])
async def test_graph_context(question: str = "What is BMI?", token: str = Depends(verify_token)) -> Dict[str, Any]:
    """Test endpoint to check graph context extraction for a specific question."""
    if neo4j_driver is None:
        return {"error": "Neo4j driver not initialized"}
    
    try:
        # Use the same logic as the batch query
        graph_context = ""
        logger.info(f"🔍 Testing graph context extraction for: {question}")
        
        with neo4j_driver.session(database=config.NEO4J_DATABASE) as session:
            # Get the document source from Neo4j
            source_result = session.run("MATCH (d:Document) RETURN d.source AS source LIMIT 1").single()
            source = source_result["source"] if source_result else ""
            
            # 1. Fulltext search on pages
            try:
                fulltext_result = session.run(
                    "CALL db.index.fulltext.queryNodes('pageTextIndex', $q) YIELD node, score RETURN node.text, node.page, score ORDER BY score DESC LIMIT 5",
                    q=question
                ).data()
                if fulltext_result:
                    graph_context += f"[Fulltext Search] Found {len(fulltext_result)} relevant pages:\n"
                    for r in fulltext_result:
                        page_text = r.get('text', '')[:200] + "..." if len(r.get('text', '')) > 200 else r.get('text', '')
                        graph_context += f"Page {r.get('page')} (score: {r.get('score', 0):.2f}): {page_text}\n\n"
            except Exception as e:
                graph_context += f"[Fulltext Error]: {e}\n"
            
            # 2. Entity search
            try:
                processed_query = await query_processor.process_query(question)
                keywords = processed_query["enhanced_extraction"].get("key_entities", [])
                
                if keywords:
                    for kw in keywords:
                        entity_result = session.run(
                            """
                            MATCH (e:Entity)-[:MENTIONED_IN]->(p:Page)
                            WHERE toLower(e.name) CONTAINS toLower($kw) 
                               OR toLower(e.type) CONTAINS toLower($kw)
                            RETURN e.name, e.type, p.page
                            ORDER BY p.page
                            LIMIT 10
                            """,
                            kw=kw
                        ).data()
                        if entity_result:
                            graph_context += f"[Entity Search for '{kw}'] Found {len(entity_result)} matches:\n"
                            for r in entity_result:
                                graph_context += f"- {r.get('name')} ({r.get('type')}) on page {r.get('page')}\n"
                            graph_context += "\n"
            except Exception as e:
                graph_context += f"[Entity Search Error]: {e}\n"
            
            # 3. General entities
            try:
                general_entities = session.run(
                    """
                    MATCH (e:Entity)-[:MENTIONED_IN]->(p:Page)
                    RETURN e.name, e.type, p.page
                    ORDER BY p.page
                    LIMIT 20
                    """
                ).data()
                if general_entities:
                    graph_context += f"[General Entities] Found {len(general_entities)} entities:\n"
                    for r in general_entities:
                        graph_context += f"- {r.get('name')} ({r.get('type')}) on page {r.get('page')}\n"
            except Exception as e:
                graph_context += f"[General Entities Error]: {e}\n"
        
        return {
            "question": question,
            "graph_context": graph_context,
            "context_length": len(graph_context),
            "has_context": bool(graph_context.strip()),
            "source": source[:100] + "..." if len(source) > 100 else source
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