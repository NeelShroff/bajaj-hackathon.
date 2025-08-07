# src/query_processor.py

import re
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from openai import AsyncOpenAI
import json

from config import config
from llama_index.core.query_engine import BaseQueryEngine

logger = logging.getLogger(__name__)

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
                max_tokens=500,
                # Using the new, robust JSON mode
                response_format={"type": "json_object"}
            )
            
            llm_response = response.choices[0].message.content.strip()
            return json.loads(llm_response)
        
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
    
    async def process_query_for_decision(self, query: str, query_engine: BaseQueryEngine) -> Dict[str, Any]:
        """
        Main method to process a query, retrieve information, and generate a structured decision.
        Now includes multi-step reasoning for interlinked information.
        """
        logger.info(f"Processing query for structured decision: {query}")

        # 1. Initial Retrieval
        initial_response = await query_engine.aquery(query)
        retrieved_text = initial_response.response
        source_nodes = initial_response.source_nodes

        # 2. Check for interlinked references and perform a second search if necessary
        second_query_needed = False
        reference_keywords = ["refer to", "see section", "as per", "in the table", "as shown in the table"]
        for keyword in reference_keywords:
            if keyword in retrieved_text.lower():
                second_query_needed = True
                break

        if second_query_needed:
            logger.info("Detected interlinked reference. Performing a secondary, targeted query.")
            secondary_query = f"Based on the initial context: '{retrieved_text}', provide a full answer to the user's original query: '{query}'. Include information from all relevant sections and tables."
            secondary_response = await query_engine.aquery(secondary_query)
            retrieved_text = f"{retrieved_text}\n\n[Secondary Search Result]:\n{secondary_response.response}"
            source_nodes.extend(secondary_response.source_nodes)

        clauses = [node.get_text() for node in source_nodes]
        sources = list(set([node.metadata.get('source', 'Unknown') for node in source_nodes]))
        
        # 3. Use a new LLM call to synthesize a structured decision
        try:
            decision_prompt = f"""
            Based on the following document context, provide a direct and concise answer to the user query. The answer should be a single, coherent statement and should use information directly from the provided context without adding any external details. If the context does not contain the answer, state that the information is not available.

            Document Context:
            ---
            {retrieved_text}
            ---

            User Query: "{query}"

            Direct Answer: 
            """
            
            decision_response = await async_llm_client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[{"role": "user", "content": decision_prompt}],
                temperature=0.1,
                max_tokens=1000
            )

            answer_content = decision_response.choices[0].message.content.strip()

            return {
                "justification": answer_content,
                "decision": "Approved" if answer_content and "not available" not in answer_content.lower() else "Rejected",
                "amount": None,
                "clauses": clauses
            }
        except Exception as e:
            logger.error(f"Error processing decision with LLM: {e}")
            return {
                "justification": "Error processing this question.",
                "decision": "Error",
                "amount": None,
                "clauses": clauses
            }