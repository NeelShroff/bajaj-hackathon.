import re
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from openai import OpenAI, AsyncOpenAI

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
            Analyze the following query and extract structured information. The query could be about any document (e.g., policy, contract, legal text, HR handbook).

            Query: "{query}"

            Please extract and return a JSON object with the following fields:
            - key_entities: list of main subjects or concepts (e.g., 'knee surgery', 'maternity', 'contract termination').
            - constraints: A dictionary of key-value pairs representing conditions or limitations. Identify any dates, durations, amounts, locations, or specific requirements. For a query like "46-year-old male, Pune, 3-month-old policy," constraints would be {{'age': 46, 'gender': 'male', 'location': 'Pune', 'policy_duration': '3 months'}}.
            - query_type: A single label describing the user's intent, such as "coverage_check", "amount_inquiry", "eligibility_check", "definition_lookup", "procedure_guide", or "general_inquiry".
            - confidence: a confidence score (0-1) for the extraction.

            Return only the JSON object, no additional text or conversational phrases.
            """
            
            response = await async_llm_client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
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
        
        return {
            "original_query": query,
            "enhanced_extraction": enhanced_result,
            "search_queries": search_queries,
            "processing_metadata": {
                "method": "llm-driven",
                "confidence": enhanced_result.get("confidence", 0.7)
            }
        }