import re
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from openai import OpenAI, AsyncOpenAI
from typing import Tuple

from config import config

logger = logging.getLogger(__name__)

# Use async version of LLM client
async_llm_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

@dataclass
class QueryEntities:
    """Structured representation of extracted query entities."""
    key_entities: List[str] = None
    constraints: Dict[str, Any] = None
    query_type: Optional[str] = None
    
    def __post_init__(self):
        if self.key_entities is None:
            self.key_entities = []
        if self.constraints is None:
            self.constraints = {}

class QueryProcessor:
    """Processes natural language queries and extracts structured information."""
    
    def __init__(self):
        # We don't need a separate client here as we'll use the async one globally.
        pass
    
    async def _enhance_query_with_llm(self, query: str) -> Dict[str, Any]:
        """Use LLM to perform a universal query analysis and extraction."""
        try:
            prompt = f"""
You are an enterprise-grade information extraction engine assisting a Retrieval-Augmented Generation (RAG) system. The user may ask about any domain (medicine, law, finance, engineering, science, HR, compliance, contracts, manuals, academic PDFs, etc.). Extract structured intent and entities with high recall and precision, avoiding speculation.

User Query: "{query}"

Return ONLY a compact JSON object with these fields:
- key_entities: Array of the main subjects/concepts/terms of art. Use canonical forms when possible.
- constraints: Object mapping constraint names to normalized values. Include dates (ISO 8601 if present), quantities (with units), locations, demographic attributes, policy/section/article references, statutes, formulas, or any explicit qualifiers.
- query_type: One of [coverage_check, amount_inquiry, eligibility_check, definition_lookup, procedure_guide, comparison, calculation, legal_citation, safety_compliance, general_inquiry]. Choose the closest.
- domain_hint: Short string with likely domain (e.g., "health_insurance", "contract_law", "civil_engineering", "pharmacology", "physics", "finance", "hr_policy", "research_article", "manual").
- confidence: Float between 0 and 1 reflecting extraction confidence.

Notes:
- Be conservative; do not invent facts.
- Normalize numbers and units where explicit.
- Include section/page references if present in the query.
"""
            
            response = await async_llm_client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            
            import json
            llm_response = response.choices[0].message.content.strip()
            if not llm_response:
                logger.warning("LLM returned empty response")
                return self._get_default_query_result()
            
            if llm_response.startswith('```json'):
                llm_response = llm_response[7:]
            if llm_response.startswith('```'):
                llm_response = llm_response[3:]
            if llm_response.endswith('```'):
                llm_response = llm_response[:-3]
            
            try:
                result = json.loads(llm_response)
                if not isinstance(result, dict):
                    logger.warning("LLM response is not a dictionary")
                    return self._get_default_query_result()
                
                required_fields = ['key_entities', 'constraints', 'query_type', 'confidence']
                for field in required_fields:
                    if field not in result:
                        result[field] = [] if field == 'key_entities' else {} if field == 'constraints' else 'general_inquiry' if field == 'query_type' else 0.5
                
                return result
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed: {e}. Response content: {llm_response[:100]}...")
                return self._get_default_query_result()
            
        except Exception as e:
            logger.error(f"Error enhancing query with LLM: {e}")
            return self._get_default_query_result()
            
    def _get_default_query_result(self) -> Dict[str, Any]:
        """Return a default query result structure when LLM enhancement fails."""
        return {
            "key_entities": [],
            "constraints": {},
            "query_type": "general_inquiry",
            "confidence": 0.5
        }
    
    def create_search_queries(self, query: str, enhanced_result: Dict[str, Any]) -> List[str]:
        """Dynamically create search queries based on the LLM's structured output."""
        queries = [query]
        
        entities = enhanced_result.get('key_entities', [])
        constraints = enhanced_result.get('constraints', {})
        query_type = enhanced_result.get('query_type', 'general_inquiry')
        
        base_query_parts = entities + [f"{k}: {v}" for k, v in constraints.items() if v]
        if base_query_parts:
            queries.append(" ".join(base_query_parts))

        if query_type == 'coverage_check' and entities:
            queries.append(f"coverage for {', '.join(entities)}")
        
        if query_type == 'amount_inquiry' and entities:
            queries.append(f"limit for {', '.join(entities)}")
            
        if query_type == 'eligibility_check' and entities:
            queries.append(f"conditions for {', '.join(entities)}")
        
        if not entities and not constraints:
            queries.append(query.lower())

        return list(dict.fromkeys(queries))

    async def process_query(self, query: str) -> Dict[str, Any]:
        """Main method to process a query and return structured information and search queries."""
        logger.info(f"Processing query: {query}")
        
        # Use LLM to extract entities and query type in a general way
        enhanced_result = await self._enhance_query_with_llm(query)
        
        # Dynamically generate search queries from the LLM's output
        search_queries = self.create_search_queries(query, enhanced_result)
        # Also return a light-weight list of entity pairs that can be used for graph hints
        entity_pairs: List[Tuple[str, str]] = []
        entities = enhanced_result.get('key_entities', []) or []
        if len(entities) >= 2:
            for i in range(len(entities) - 1):
                entity_pairs.append((entities[i], entities[i + 1]))

        return {
            "original_query": query,
            "enhanced_extraction": enhanced_result,
            "search_queries": search_queries,
            "graph_hints": entity_pairs,
            "processing_metadata": {
                "method": "llm-driven",
                "confidence": enhanced_result.get("confidence", 0.7)
            }
        }