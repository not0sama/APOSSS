import os
import json
import logging
from typing import Dict, Any, Optional, List
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

logger = logging.getLogger(__name__)

class LLMProcessor:
    """Handles LLM processing using Gemini-2.0-flash for query understanding"""
    
    def __init__(self):
        """Initialize Gemini API"""
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Initialize the model
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config={
                "temperature": 0.3,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 2048,
            },
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
        )
        
        logger.info("Gemini-2.0-flash LLM processor initialized successfully")
    
    def create_query_analysis_prompt(self, user_query: str) -> str:
        """Create a structured prompt for query analysis"""
        prompt = f"""
You are an AI assistant specialized in analyzing academic and scientific research queries. Your task is to analyze the following user query and extract structured information that will be used to search across multiple databases containing academic resources.

User Query: "{user_query}"

Please analyze this query and provide a JSON response with the following structure:

{{
    "corrected_query": "spelling-corrected version of the query",
    "intent": {{
        "primary_intent": "main purpose of the query (e.g., 'find_research', 'find_expert', 'find_equipment', 'find_materials', 'find_literature')",
        "secondary_intents": ["list of additional intents"],
        "confidence": 0.95
    }},
    "entities": {{
        "technologies": ["list of technologies mentioned"],
        "concepts": ["list of scientific/academic concepts"],
        "people": ["list of people mentioned"],
        "organizations": ["list of organizations/institutions"],
        "locations": ["list of locations"],
        "time_periods": ["list of time references"]
    }},
    "keywords": {{
        "primary": ["most important 3-5 keywords"],
        "secondary": ["additional relevant keywords"],
        "technical_terms": ["specialized technical terms"]
    }},
    "synonyms_and_related": {{
        "synonyms": ["alternative terms with same meaning"],
        "related_terms": ["conceptually related terms"],
        "broader_terms": ["more general terms"],
        "narrower_terms": ["more specific terms"]
    }},
    "academic_fields": {{
        "primary_field": "main academic discipline",
        "related_fields": ["list of related academic fields"],
        "specializations": ["specific areas of specialization"]
    }},
    "search_context": {{
        "resource_preferences": ["types of resources most relevant (books, papers, experts, equipment, etc.)"],
        "urgency": "low/medium/high",
        "scope": "narrow/broad/comprehensive"
    }}
}}

Guidelines:
1. Correct any spelling errors in the original query
2. Be comprehensive but precise in keyword extraction
3. Consider academic and scientific terminology
4. Think about what types of resources would be most helpful
5. Consider related fields that might have relevant information
6. Provide confidence scores where applicable

Return only the JSON response, no additional text.
"""
        return prompt
    
    def process_query(self, user_query: str) -> Optional[Dict[str, Any]]:
        """Process user query with Gemini and return structured analysis"""
        try:
            # Create the analysis prompt
            prompt = self.create_query_analysis_prompt(user_query)
            
            # Generate response from Gemini
            response = self.model.generate_content(prompt)
            
            # Extract text from response
            if not response.text:
                logger.error("Empty response from Gemini")
                return None
            
            # Parse JSON response
            try:
                # Clean up the response text (remove any markdown formatting)
                response_text = response.text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                
                parsed_response = json.loads(response_text)
                
                # Add metadata
                parsed_response['_metadata'] = {
                    'original_query': user_query,
                    'processing_timestamp': self._get_timestamp(),
                    'model_used': 'gemini-2.0-flash-exp',
                    'success': True
                }
                
                logger.info(f"Successfully processed query: {user_query[:50]}...")
                return parsed_response
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {str(e)}")
                logger.error(f"Raw response: {response.text}")
                
                # Return fallback response
                return self._create_fallback_response(user_query, f"JSON parsing error: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error processing query with Gemini: {str(e)}")
            return self._create_fallback_response(user_query, str(e))
    
    def _create_fallback_response(self, user_query: str, error_msg: str) -> Dict[str, Any]:
        """Create a fallback response when LLM processing fails"""
        # Basic keyword extraction (simple approach)
        words = user_query.lower().split()
        keywords = [word.strip('.,!?;:"()[]{}') for word in words if len(word) > 2]
        
        return {
            "corrected_query": user_query,
            "intent": {
                "primary_intent": "general_search",
                "secondary_intents": [],
                "confidence": 0.5
            },
            "entities": {
                "technologies": [],
                "concepts": keywords[:3],
                "people": [],
                "organizations": [],
                "locations": [],
                "time_periods": []
            },
            "keywords": {
                "primary": keywords[:5],
                "secondary": keywords[5:10] if len(keywords) > 5 else [],
                "technical_terms": []
            },
            "synonyms_and_related": {
                "synonyms": [],
                "related_terms": [],
                "broader_terms": [],
                "narrower_terms": []
            },
            "academic_fields": {
                "primary_field": "general",
                "related_fields": [],
                "specializations": []
            },
            "search_context": {
                "resource_preferences": ["all"],
                "urgency": "medium",
                "scope": "broad"
            },
            "_metadata": {
                "original_query": user_query,
                "processing_timestamp": self._get_timestamp(),
                "model_used": "fallback",
                "success": False,
                "error": error_msg
            }
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def test_connection(self) -> Dict[str, Any]:
        """Test the connection to Gemini API"""
        try:
            test_response = self.model.generate_content("Hello, this is a connection test.")
            
            return {
                "connected": True,
                "model": "gemini-2.0-flash-exp",
                "test_response": test_response.text[:100] + "..." if test_response.text else "No response"
            }
        except Exception as e:
            return {
                "connected": False,
                "error": str(e)
            } 