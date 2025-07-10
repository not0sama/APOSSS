import logging
from typing import List, Dict, Any, Optional
from .llm_processor import LLMProcessor
import json

logger = logging.getLogger(__name__)

class LLMRanker:
    """Handles re-ranking of search results using LLM"""
    
    def __init__(self, llm_processor: LLMProcessor):
        """Initialize with LLM processor"""
        self.llm_processor = llm_processor
        logger.info("LLM ranker initialized successfully")
    
    def create_reranking_prompt(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Create a prompt for re-ranking results"""
        prompt = f"""
You are an expert academic search assistant. Your task is to re-rank and analyze the relevance of search results for the following query:

QUERY: "{query}"

Here are the search results to analyze (in JSON format):
{self._format_results_for_prompt(results)}

Please analyze these results and provide a JSON response with the following structure:

{{
    "reranked_results": [
        {{
            "id": "result_id",
            "relevance_score": 0.95,  // Score between 0 and 1
            "relevance_explanation": "Brief explanation of why this result is relevant",
            "intent_match": {{
                "primary_intent": "main intent this result matches",
                "secondary_intents": ["additional intents this result matches"],
                "confidence": 0.95
            }},
            "key_aspects": [
                {{
                    "aspect": "specific aspect of the result",
                    "relevance": "how this aspect relates to the query",
                    "importance": 0.95  // Score between 0 and 1
                }}
            ]
        }}
    ],
    "analysis": {{
        "query_intent_fulfillment": {{
            "primary_intent": "how well the results match the primary intent",
            "secondary_intents": "how well the results match secondary intents"
        }},
        "coverage_analysis": {{
            "strengths": ["what aspects of the query are well covered"],
            "gaps": ["what aspects of the query are not well covered"]
        }},
        "quality_assessment": {{
            "overall_quality": 0.95,  // Score between 0 and 1
            "reliability": 0.95,  // Score between 0 and 1
            "completeness": 0.95  // Score between 0 and 1
        }}
    }}
}}

Guidelines:
1. Focus on academic and scientific relevance
2. Consider both explicit and implicit relevance
3. Evaluate the quality and reliability of the results
4. Identify how well each result matches different aspects of the query
5. Provide clear, concise explanations for relevance
6. Consider the academic context and domain-specific factors
7. Evaluate the completeness and depth of the information
8. Assess the reliability and authority of the sources

Return only the JSON response, no additional text.
"""
        return prompt
    
    def _format_results_for_prompt(self, results: List[Dict[str, Any]]) -> str:
        """Format results for the prompt"""
        formatted = []
        for result in results:
            formatted.append({
                "id": result.get("id", ""),
                "title": result.get("title", ""),
                "content": result.get("content", "")[:500],  # Limit content length
                "metadata": {
                    "type": result.get("type", ""),
                    "score": result.get("score", 0),
                    "score_breakdown": result.get("score_breakdown", {})
                }
            })
        return str(formatted)
    
    def rerank_results(self, query: str, results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Re-rank search results using LLM"""
        try:
            # Create prompt
            prompt = self.create_reranking_prompt(query, results)
            
            # Get LLM response
            response = self.llm_processor.model.generate_content(prompt)
            
            if response and response.text:
                # Parse LLM response
                reranked_data = json.loads(response.text)
                
                # Update original results with LLM analysis
                self._update_results_with_llm_analysis(results, reranked_data)
                
                return reranked_data
            else:
                logger.warning("No response from LLM for re-ranking")
                return None
                
        except Exception as e:
            logger.error(f"Error in LLM re-ranking: {str(e)}")
            return None
    
    def _update_results_with_llm_analysis(self, results: List[Dict[str, Any]], 
                                        reranked_data: Dict[str, Any]) -> None:
        """Update original results with LLM analysis"""
        # Create lookup dictionary for reranked results
        reranked_lookup = {
            item["id"]: item 
            for item in reranked_data.get("reranked_results", [])
        }
        
        # Update original results
        for result in results:
            result_id = result.get("id")
            if result_id in reranked_lookup:
                reranked = reranked_lookup[result_id]
                
                # Add LLM analysis to score breakdown
                if "score_breakdown" not in result:
                    result["score_breakdown"] = {}
                
                result["score_breakdown"]["llm_relevance"] = reranked["relevance_score"]
                result["score_breakdown"]["llm_explanation"] = reranked["relevance_explanation"]
                result["score_breakdown"]["llm_intent_match"] = reranked["intent_match"]
                result["score_breakdown"]["llm_key_aspects"] = reranked["key_aspects"]
                
                # Update overall score to include LLM relevance
                if "score" in result:
                    result["score"] = (result["score"] + reranked["relevance_score"]) / 2 