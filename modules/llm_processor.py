import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
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
                "temperature": 0.2,  # Lower temperature for more consistent structured output
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 3072,  # Increased for more comprehensive analysis
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
        """Create a comprehensive structured prompt for multilingual query analysis"""
        prompt = f"""
You are an expert AI assistant specialized in analyzing academic, scientific, and research queries across multiple languages. Your task is to perform a comprehensive analysis of the user query through multiple stages.

USER QUERY: "{user_query}"

ANALYSIS REQUIREMENTS:
1. Language Detection: Identify the language with high accuracy
2. Query Correction: Fix any spelling errors, typos, or grammatical issues
3. Translation: Translate to English if the original is not English
4. Comprehensive Information Extraction: Extract all relevant information for academic search

Please provide a JSON response with the following EXACT structure:

{{
    "language_analysis": {{
        "detected_language": "ISO 639-1 code (en, fr, es, ar, zh, de, it, pt, ru, ja, ko, hi, etc.)",
        "language_name": "Full language name in English",
        "confidence_score": 0.95,
        "is_english": true/false,
        "script_type": "latin/arabic/cyrillic/chinese/japanese/korean/devanagari/etc."
    }},
    "query_processing": {{
        "original_query": "{user_query}",
        "corrected_original": "corrected version of original query (fix typos/grammar)",
        "english_translation": "English translation (same as corrected_original if already English)",
        "translation_needed": true/false,
        "correction_made": true/false,
        "processing_notes": "brief note about corrections or translation approach"
    }},
    "intent_analysis": {{
        "primary_intent": "find_research_papers/find_experts/find_equipment/find_materials/find_books/find_funding/comparative_analysis/methodology_search/literature_review/general_search",
        "secondary_intents": ["array of additional intents"],
        "search_scope": "narrow/medium/broad/comprehensive",
        "urgency_level": "low/medium/high",
        "academic_level": "undergraduate/graduate/postgraduate/professional/general",
        "confidence": 0.95
    }},
    "entity_extraction": {{
        "people": ["researchers, authors, scientists, inventors"],
        "organizations": ["universities, companies, research institutions, labs"],
        "locations": ["countries, cities, regions relevant to research"],
        "technologies": ["specific technologies, methods, techniques, tools"],
        "concepts": ["scientific concepts, theories, phenomena"],
        "chemicals_materials": ["chemical compounds, materials, substances"],
        "medical_terms": ["diseases, treatments, medical procedures, anatomy"],
        "mathematical_terms": ["formulas, mathematical concepts, statistical methods"],
        "time_periods": ["years, decades, historical periods"],
        "publications": ["journals, books, conferences, specific papers"],
        "fields_of_study": ["academic disciplines and subdisciplines"]
    }},
    "keyword_analysis": {{
        "primary_keywords": ["3-5 most important keywords from English version"],
        "secondary_keywords": ["5-8 additional relevant keywords"],
        "technical_terms": ["specialized technical/scientific terminology"],
        "original_language_keywords": ["important keywords from original language if not English"],
        "long_tail_keywords": ["specific phrase-based keywords"],
        "alternative_spellings": ["different spellings of key terms"]
    }},
    "semantic_expansion": {{
        "synonyms": ["direct synonyms of key terms"],
        "related_terms": ["conceptually related terms"],
        "broader_terms": ["more general/umbrella terms"],
        "narrower_terms": ["more specific/detailed terms"],
        "domain_specific_terms": ["field-specific jargon and terminology"],
        "cross_linguistic_terms": ["equivalent terms in other languages if relevant"],
        "acronyms_abbreviations": ["relevant acronyms and their expansions"]
    }},
    "academic_classification": {{
        "primary_field": "main academic discipline",
        "secondary_fields": ["related academic fields"],
        "specializations": ["specific research areas or subspecialties"],
        "interdisciplinary_connections": ["fields that might have relevant cross-over"],
        "research_methodologies": ["applicable research methods"],
        "publication_types": ["most relevant types of publications for this query"]
    }},
    "search_strategy": {{
        "database_priorities": ["academic_library/research_papers/experts_system/laboratories/funding"],
        "resource_types": ["books/journals/articles/theses/equipment/materials/experts/projects"],
        "temporal_focus": "historical/current/cutting_edge/all_periods",
        "geographical_scope": "local/national/international/global",
        "quality_indicators": ["peer_reviewed/high_impact/recent/authoritative"],
        "search_complexity": "simple/moderate/complex/expert_level"
    }},
    "multilingual_considerations": {{
        "preserve_original_terms": ["terms that should be kept in original language"],
        "cultural_context": ["cultural or regional aspects to consider"],
        "translation_challenges": ["terms that may not translate directly"],
        "alternative_romanizations": ["different ways to romanize non-Latin scripts"]
    }},
    "metadata": {{
        "processing_timestamp": "{self._get_timestamp()}",
        "model_version": "gemini-2.0-flash-exp",
        "analysis_confidence": 0.95,
        "processing_time_estimate": "estimated processing time in seconds",
        "query_complexity": "simple/moderate/complex/highly_complex",
        "success": true
    }}
}}

CRITICAL INSTRUCTIONS:
1. ALWAYS detect language first - be highly accurate
2. Correct spelling/grammar errors in the original language before translation
3. Provide high-quality translation to English preserving academic meaning
4. Extract information from BOTH original and English versions when beneficial
5. Be comprehensive but precise - academic search requires thoroughness
6. Consider cross-cultural academic terminology differences
7. Include domain-specific terminology and jargon
8. Think about what researchers actually need to find
9. Consider different academic traditions and naming conventions
10. Provide confidence scores for uncertain elements

RESPONSE FORMAT: Return ONLY the JSON response, no additional text or formatting.
"""
        return prompt
    
    def process_query(self, user_query: str) -> Optional[Dict[str, Any]]:
        """Process query with LLM and return raw response for QueryProcessor to validate"""
        try:
            # Create comprehensive prompt
            prompt = self.create_query_analysis_prompt(user_query)
            
            # Get LLM response
            logger.info("Sending query to Gemini for comprehensive analysis...")
            response = self.model.generate_content(prompt)
            
            if response and response.text:
                # Clean and parse LLM response
                response_text = response.text.strip()
                
                # Remove any markdown formatting that might be present
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                
                # Parse JSON response
                processed_query = json.loads(response_text)
                
                # Basic validation only - let QueryProcessor handle detailed validation
                processed_query['metadata']['success'] = True
                processed_query['metadata']['original_query'] = user_query
                
                logger.info("Query processed successfully by LLM")
                return processed_query
                
            else:
                logger.warning("No response from LLM")
                return None
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            logger.error(f"Raw response: {response.text if response else 'No response'}")
            return None
        except Exception as e:
            logger.error(f"Error processing query with LLM: {str(e)}")
            return None
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        return datetime.now().isoformat()
    
    def test_connection(self) -> Dict[str, Any]:
        """Test basic connection to Gemini API"""
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
    
    def test_enhanced_connection(self) -> Dict[str, Any]:
        """Test the enhanced LLM processing with multilingual capability"""
        try:
            # Test with a multilingual query
            test_queries = [
                "machine learning for medical diagnosis",  # English
                "apprentissage automatique pour diagnostic médical",  # French
                "تعلم الآلة للتشخيص الطبي",  # Arabic
                "医療診断のための機械学習"  # Japanese
            ]
            
            test_results = []
            
            for query in test_queries:
                try:
                    result = self.process_query(query)
                    if result:
                        lang_analysis = result.get('language_analysis', {})
                        query_processing = result.get('query_processing', {})
                        test_results.append({
                            "query": query,
                            "detected_language": lang_analysis.get('detected_language'),
                            "translation_needed": query_processing.get('translation_needed'),
                            "success": result.get('metadata', {}).get('success', False)
                        })
                    else:
                        test_results.append({
                            "query": query,
                            "success": False,
                            "error": "No result returned"
                        })
                except Exception as e:
                    test_results.append({
                        "query": query,
                        "success": False,
                        "error": str(e)
                    })
            
            return {
                "connected": True,
                "model": "gemini-2.0-flash-exp",
                "enhanced_processing": True,
                "multilingual_support": True,
                "test_results": test_results,
                "overall_success": all(r.get('success', False) for r in test_results)
            }
            
        except Exception as e:
            return {
                "connected": False,
                "enhanced_processing": False,
                "multilingual_support": False,
                "error": str(e)
            } 