import logging
from typing import Dict, Any, Optional
from .llm_processor import LLMProcessor

logger = logging.getLogger(__name__)

class QueryProcessor:
    """Orchestrates the query processing workflow"""
    
    def __init__(self, llm_processor: LLMProcessor):
        """Initialize with LLM processor"""
        self.llm_processor = llm_processor
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
            # Validate input
            if not user_query or not user_query.strip():
                logger.warning("Empty or whitespace-only query received")
                return None
            
            # Clean the query
            cleaned_query = self._clean_query(user_query)
            
            # Process with LLM
            logger.info(f"Processing query: {cleaned_query[:50]}...")
            processed_data = self.llm_processor.process_query(cleaned_query)
            
            if not processed_data:
                logger.error("LLM processing returned no data")
                return None
            
            # Post-process and validate the results
            validated_data = self._validate_and_enhance_results(processed_data)
            
            logger.info("Query processing completed successfully")
            return validated_data
            
        except Exception as e:
            logger.error(f"Error in query processing pipeline: {str(e)}")
            return None
    
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