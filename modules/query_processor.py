import logging
from typing import Dict, Any, Optional
from datetime import datetime
from .llm_processor import LLMProcessor

logger = logging.getLogger(__name__)

class QueryProcessor:
    """Orchestrates the query processing workflow with comprehensive validation and enhancement"""
    
    def __init__(self, llm_processor: LLMProcessor = None):
        """Initialize with LLM processor"""
        self.llm_processor = llm_processor or LLMProcessor()
        logger.info("Query processor initialized successfully")
    
    def process_query(self, user_query: str) -> Optional[Dict[str, Any]]:
        """
        Process a user query through the complete pipeline
        
        Args:
            user_query: The raw user query string
            
        Returns:
            Processed query data or None if processing fails
        """
        try:
            # Step 1: Input validation
            if not user_query or not user_query.strip():
                logger.warning("Empty or whitespace-only query received")
                return None
            
            # Step 2: Clean the query
            cleaned_query = self._clean_query(user_query)
            logger.info(f"Processing query: {cleaned_query[:50]}...")
            
            # Step 3: Process with LLM
            processed_data = self.llm_processor.process_query(cleaned_query)
            
            if not processed_data:
                logger.error("LLM processing returned no data")
                return self._create_enhanced_fallback_response(cleaned_query, "LLM processing failed")
            
            # Step 4: Validate and enhance the LLM response
            processed_data = self._validate_and_enhance_response(processed_data, cleaned_query)
            
            # Step 5: Add backward compatibility fields
            processed_data = self._add_backward_compatibility(processed_data)
            
            # Step 6: Final validation and enhancement
            validated_data = self._validate_and_enhance_results(processed_data)
            
            logger.info("Query processing completed successfully")
            return validated_data
            
        except Exception as e:
            logger.error(f"Error in query processing pipeline: {str(e)}")
            return self._create_enhanced_fallback_response(user_query, str(e))
    
    def _clean_query(self, query: str) -> str:
        """Clean and prepare the query"""
        # Remove excessive whitespace
        cleaned = ' '.join(query.split())
        
        # Remove common noise characters while preserving meaningful punctuation
        noise_chars = ['<', '>', '|', '\\', '/', '{', '}', '[', ']', '`', '~']
        for char in noise_chars:
            cleaned = cleaned.replace(char, ' ')
        
        # Remove excessive whitespace again
        cleaned = ' '.join(cleaned.split())
        
        return cleaned.strip()
    
    def _validate_and_enhance_response(self, processed_query: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """Validate and enhance the LLM response"""
        try:
            # Ensure all required sections exist
            required_sections = [
                'language_analysis', 'query_processing', 'intent_analysis',
                'entity_extraction', 'keyword_analysis', 'semantic_expansion',
                'academic_classification', 'search_strategy', 'metadata'
            ]
            
            for section in required_sections:
                if section not in processed_query:
                    logger.warning(f"Missing section {section}, adding default")
                    processed_query[section] = self._get_default_section(section, original_query)
            
            # Validate language analysis
            lang_analysis = processed_query['language_analysis']
            if not lang_analysis.get('detected_language'):
                lang_analysis['detected_language'] = 'en'
                lang_analysis['language_name'] = 'English'
                lang_analysis['is_english'] = True
            
            # Ensure query processing has the English version
            query_proc = processed_query['query_processing']
            if not query_proc.get('english_translation'):
                query_proc['english_translation'] = original_query
            
            # Set metadata success flag
            processed_query['metadata']['success'] = True
            processed_query['metadata']['original_query'] = original_query
            
            return processed_query
            
        except Exception as e:
            logger.error(f"Error validating response: {str(e)}")
            return processed_query
    
    def _get_default_section(self, section: str, original_query: str) -> Dict[str, Any]:
        """Get default values for missing sections"""
        defaults = {
            'language_analysis': {
                'detected_language': 'en',
                'language_name': 'English',
                'confidence_score': 0.5,
                'is_english': True,
                'script_type': 'latin'
            },
            'query_processing': {
                'original_query': original_query,
                'corrected_original': original_query,
                'english_translation': original_query,
                'translation_needed': False,
                'correction_made': False,
                'processing_notes': 'Default processing - no analysis performed'
            },
            'intent_analysis': {
                'primary_intent': 'general_search',
                'secondary_intents': [],
                'search_scope': 'broad',
                'urgency_level': 'medium',
                'academic_level': 'general',
                'confidence': 0.5
            },
            'entity_extraction': {
                'people': [], 'organizations': [], 'locations': [],
                'technologies': [], 'concepts': [], 'chemicals_materials': [],
                'medical_terms': [], 'mathematical_terms': [], 'time_periods': [],
                'publications': [], 'fields_of_study': []
            },
            'keyword_analysis': {
                'primary_keywords': original_query.split()[:5],
                'secondary_keywords': [],
                'technical_terms': [],
                'original_language_keywords': [],
                'long_tail_keywords': [],
                'alternative_spellings': []
            },
            'semantic_expansion': {
                'synonyms': [], 'related_terms': [], 'broader_terms': [],
                'narrower_terms': [], 'domain_specific_terms': [],
                'cross_linguistic_terms': [], 'acronyms_abbreviations': []
            },
            'academic_classification': {
                'primary_field': 'general',
                'secondary_fields': [],
                'specializations': [],
                'interdisciplinary_connections': [],
                'research_methodologies': [],
                'publication_types': []
            },
            'search_strategy': {
                'database_priorities': ['academic_library', 'research_papers'],
                'resource_types': ['all'],
                'temporal_focus': 'all_periods',
                'geographical_scope': 'global',
                'quality_indicators': ['authoritative'],
                'search_complexity': 'simple'
            },
            'metadata': {
                'processing_timestamp': self._get_timestamp(),
                'model_version': 'gemini-2.0-flash-exp',
                'analysis_confidence': 0.5,
                'query_complexity': 'simple',
                'success': False
            }
        }
        
        return defaults.get(section, {})
    
    def _add_backward_compatibility(self, processed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Add backward compatibility fields for existing code"""
        try:
            # Add old format fields for compatibility
            query_proc = processed_query.get('query_processing', {})
            keyword_analysis = processed_query.get('keyword_analysis', {})
            entity_extraction = processed_query.get('entity_extraction', {})
            academic_class = processed_query.get('academic_classification', {})
            intent_analysis = processed_query.get('intent_analysis', {})
            search_strategy = processed_query.get('search_strategy', {})
            
            # Old format compatibility
            processed_query['corrected_query'] = query_proc.get('english_translation', '')
            
            processed_query['intent'] = {
                'primary_intent': intent_analysis.get('primary_intent', 'general_search'),
                'secondary_intents': intent_analysis.get('secondary_intents', []),
                'confidence': intent_analysis.get('confidence', 0.5)
            }
            
            processed_query['entities'] = {
                'technologies': entity_extraction.get('technologies', []),
                'concepts': entity_extraction.get('concepts', []),
                'people': entity_extraction.get('people', []),
                'organizations': entity_extraction.get('organizations', []),
                'locations': entity_extraction.get('locations', []),
                'time_periods': entity_extraction.get('time_periods', [])
            }
            
            processed_query['keywords'] = {
                'primary': keyword_analysis.get('primary_keywords', []),
                'secondary': keyword_analysis.get('secondary_keywords', []),
                'technical_terms': keyword_analysis.get('technical_terms', [])
            }
            
            semantic_exp = processed_query.get('semantic_expansion', {})
            processed_query['synonyms_and_related'] = {
                'synonyms': semantic_exp.get('synonyms', []),
                'related_terms': semantic_exp.get('related_terms', []),
                'broader_terms': semantic_exp.get('broader_terms', []),
                'narrower_terms': semantic_exp.get('narrower_terms', [])
            }
            
            processed_query['academic_fields'] = {
                'primary_field': academic_class.get('primary_field', 'general'),
                'related_fields': academic_class.get('secondary_fields', []),
                'specializations': academic_class.get('specializations', [])
            }
            
            processed_query['search_context'] = {
                'resource_preferences': search_strategy.get('resource_types', ['all']),
                'urgency': intent_analysis.get('urgency_level', 'medium'),
                'scope': intent_analysis.get('search_scope', 'broad')
            }
            
            # Add metadata for new features
            processed_query['_metadata'] = {
                'original_query': query_proc.get('original_query', ''),
                'processing_timestamp': processed_query.get('metadata', {}).get('processing_timestamp', self._get_timestamp()),
                'model_used': 'gemini-2.0-flash-exp',
                'success': processed_query.get('metadata', {}).get('success', True),
                'enhanced_analysis': True,
                'language_detected': processed_query.get('language_analysis', {}).get('detected_language', 'en'),
                'translation_performed': query_proc.get('translation_needed', False)
            }
            
            return processed_query
            
        except Exception as e:
            logger.error(f"Error adding backward compatibility: {str(e)}")
            return processed_query
    
    def _create_enhanced_fallback_response(self, user_query: str, error_msg: str) -> Dict[str, Any]:
        """Create an enhanced fallback response when LLM processing fails"""
        # Basic keyword extraction
        words = user_query.lower().split()
        keywords = [word.strip('.,!?;:"()[]{}') for word in words if len(word) > 2]
        
        # Basic language detection (simple heuristic)
        is_likely_english = all(ord(char) < 128 for char in user_query)
        
        fallback_response = {
            "language_analysis": {
                "detected_language": "en" if is_likely_english else "unknown",
                "language_name": "English" if is_likely_english else "Unknown",
                "confidence_score": 0.3,
                "is_english": is_likely_english,
                "script_type": "latin" if is_likely_english else "unknown"
            },
            "query_processing": {
                "original_query": user_query,
                "corrected_original": user_query,
                "english_translation": user_query,
                "translation_needed": False,
                "correction_made": False,
                "processing_notes": "Fallback processing - limited analysis"
            },
            "intent_analysis": {
                "primary_intent": "general_search",
                "secondary_intents": [],
                "search_scope": "broad",
                "urgency_level": "medium",
                "academic_level": "general",
                "confidence": 0.3
            },
            "entity_extraction": {
                "people": [], "organizations": [], "locations": [],
                "technologies": [], "concepts": keywords[:3],
                "chemicals_materials": [], "medical_terms": [],
                "mathematical_terms": [], "time_periods": [],
                "publications": [], "fields_of_study": []
            },
            "keyword_analysis": {
                "primary_keywords": keywords[:5],
                "secondary_keywords": keywords[5:10] if len(keywords) > 5 else [],
                "technical_terms": [],
                "original_language_keywords": [],
                "long_tail_keywords": [],
                "alternative_spellings": []
            },
            "semantic_expansion": {
                "synonyms": [], "related_terms": [], "broader_terms": [],
                "narrower_terms": [], "domain_specific_terms": [],
                "cross_linguistic_terms": [], "acronyms_abbreviations": []
            },
            "academic_classification": {
                "primary_field": "general",
                "secondary_fields": [],
                "specializations": [],
                "interdisciplinary_connections": [],
                "research_methodologies": [],
                "publication_types": []
            },
            "search_strategy": {
                "database_priorities": ["academic_library", "research_papers"],
                "resource_types": ["all"],
                "temporal_focus": "all_periods",
                "geographical_scope": "global",
                "quality_indicators": ["authoritative"],
                "search_complexity": "simple"
            },
            "metadata": {
                "processing_timestamp": self._get_timestamp(),
                "model_version": "fallback",
                "analysis_confidence": 0.3,
                "query_complexity": "unknown",
                "success": False,
                "error": error_msg
            }
        }
        
        # Add backward compatibility
        return self._add_backward_compatibility(fallback_response)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        return datetime.now().isoformat()
    
    def _validate_and_enhance_results(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and enhance the processed query results"""
        try:
            # Ensure all required fields exist
            required_fields = [
                'corrected_query', 'intent', 'entities', 'keywords',
                'synonyms_and_related', 'academic_fields', 'search_context'
            ]
            
            for field in required_fields:
                if field not in data:
                    logger.warning(f"Missing required field: {field}")
                    data[field] = self._get_default_field_value(field)
            
            # Validate and clean keywords
            data['keywords'] = self._validate_keywords(data.get('keywords', {}))
            
            # Validate intent structure
            data['intent'] = self._validate_intent(data.get('intent', {}))
            
            # Add processing statistics
            if '_metadata' not in data:
                data['_metadata'] = {}
            
            data['_metadata']['processing_stats'] = self._calculate_processing_stats(data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error validating results: {str(e)}")
            return data
    
    def _get_default_field_value(self, field: str) -> Any:
        """Get default value for missing fields"""
        defaults = {
            'corrected_query': '',
            'intent': {
                'primary_intent': 'general_search',
                'secondary_intents': [],
                'confidence': 0.5
            },
            'entities': {
                'technologies': [],
                'concepts': [],
                'people': [],
                'organizations': [],
                'locations': [],
                'time_periods': []
            },
            'keywords': {
                'primary': [],
                'secondary': [],
                'technical_terms': []
            },
            'synonyms_and_related': {
                'synonyms': [],
                'related_terms': [],
                'broader_terms': [],
                'narrower_terms': []
            },
            'academic_fields': {
                'primary_field': 'general',
                'related_fields': [],
                'specializations': []
            },
            'search_context': {
                'resource_preferences': ['all'],
                'urgency': 'medium',
                'scope': 'broad'
            }
        }
        
        return defaults.get(field, {})
    
    def _validate_keywords(self, keywords: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean keywords structure"""
        validated = {
            'primary': [],
            'secondary': [],
            'technical_terms': []
        }
        
        for key in validated.keys():
            if key in keywords and isinstance(keywords[key], list):
                # Clean and filter keywords
                clean_keywords = []
                for keyword in keywords[key]:
                    if isinstance(keyword, str) and keyword.strip():
                        clean_keyword = keyword.strip().lower()
                        if len(clean_keyword) > 1 and clean_keyword not in clean_keywords:
                            clean_keywords.append(clean_keyword)
                
                validated[key] = clean_keywords[:10]  # Limit to 10 keywords per category
        
        return validated
    
    def _validate_intent(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Validate intent structure"""
        valid_intents = [
            'find_research', 'find_expert', 'find_equipment', 'find_materials',
            'find_literature', 'general_search', 'comparative_analysis'
        ]
        
        validated = {
            'primary_intent': 'general_search',
            'secondary_intents': [],
            'confidence': 0.5
        }
        
        # Validate primary intent
        if 'primary_intent' in intent and intent['primary_intent'] in valid_intents:
            validated['primary_intent'] = intent['primary_intent']
        
        # Validate secondary intents
        if 'secondary_intents' in intent and isinstance(intent['secondary_intents'], list):
            validated['secondary_intents'] = [
                i for i in intent['secondary_intents'] 
                if isinstance(i, str) and i in valid_intents
            ][:3]  # Limit to 3 secondary intents
        
        # Validate confidence
        if 'confidence' in intent and isinstance(intent['confidence'], (int, float)):
            confidence = float(intent['confidence'])
            validated['confidence'] = max(0.0, min(1.0, confidence))
        
        return validated
    
    def _calculate_processing_stats(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate processing statistics"""
        stats = {
            'total_keywords': 0,
            'total_entities': 0,
            'total_synonyms': 0,
            'complexity_score': 0.0
        }
        
        # Count keywords
        if 'keywords' in data:
            for category in ['primary', 'secondary', 'technical_terms']:
                if category in data['keywords']:
                    stats['total_keywords'] += len(data['keywords'][category])
        
        # Count entities
        if 'entities' in data:
            for category in data['entities'].values():
                if isinstance(category, list):
                    stats['total_entities'] += len(category)
        
        # Count synonyms and related terms
        if 'synonyms_and_related' in data:
            for category in data['synonyms_and_related'].values():
                if isinstance(category, list):
                    stats['total_synonyms'] += len(category)
        
        # Calculate complexity score (0-1)
        complexity_factors = [
            min(stats['total_keywords'] / 10, 1),
            min(stats['total_entities'] / 5, 1),
            min(stats['total_synonyms'] / 15, 1),
            data.get('intent', {}).get('confidence', 0.5)
        ]
        
        stats['complexity_score'] = sum(complexity_factors) / len(complexity_factors)
        
        return stats
    
    def get_search_terms(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract search terms in a format suitable for database queries"""
        if not processed_data:
            return {}
        
        search_terms = {
            'primary_terms': [],
            'secondary_terms': [],
            'entity_terms': [],
            'field_terms': [],
            'intent': processed_data.get('intent', {}).get('primary_intent', 'general_search')
        }
        
        # Extract primary search terms
        keywords = processed_data.get('keywords', {})
        search_terms['primary_terms'] = keywords.get('primary', [])
        search_terms['secondary_terms'] = keywords.get('secondary', []) + keywords.get('technical_terms', [])
        
        # Extract entity terms
        entities = processed_data.get('entities', {})
        for entity_type in ['technologies', 'concepts', 'people', 'organizations']:
            search_terms['entity_terms'].extend(entities.get(entity_type, []))
        
        # Extract field terms
        academic_fields = processed_data.get('academic_fields', {})
        search_terms['field_terms'] = (
            [academic_fields.get('primary_field', '')] +
            academic_fields.get('related_fields', []) +
            academic_fields.get('specializations', [])
        )
        
        # Remove empty strings and duplicates
        for key in search_terms:
            if isinstance(search_terms[key], list):
                search_terms[key] = list(set(filter(None, search_terms[key])))
        
        return search_terms
    
    def get_comprehensive_language_info(self, processed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive language information from processed query"""
        if not processed_query:
            return {"language": "unknown", "is_multilingual": False}
        
        language_analysis = processed_query.get('language_analysis', {})
        query_processing = processed_query.get('query_processing', {})
        multilingual = processed_query.get('multilingual_considerations', {})
        
        return {
            "detected_language": language_analysis.get('detected_language', 'en'),
            "language_name": language_analysis.get('language_name', 'English'),
            "script_type": language_analysis.get('script_type', 'latin'),
            "confidence_score": language_analysis.get('confidence_score', 0.5),
            "is_english": language_analysis.get('is_english', True),
            "translation_needed": query_processing.get('translation_needed', False),
            "correction_made": query_processing.get('correction_made', False),
            "original_query": query_processing.get('original_query', ''),
            "corrected_original": query_processing.get('corrected_original', ''),
            "english_translation": query_processing.get('english_translation', ''),
            "processing_notes": query_processing.get('processing_notes', ''),
            "cultural_context": multilingual.get('cultural_context', []),
            "preserve_original_terms": multilingual.get('preserve_original_terms', []),
            "translation_challenges": multilingual.get('translation_challenges', [])
        }
    
    def test_enhanced_connection(self) -> Dict[str, Any]:
        """Test the enhanced LLM processing with multilingual capability"""
        return self.llm_processor.test_enhanced_connection() 