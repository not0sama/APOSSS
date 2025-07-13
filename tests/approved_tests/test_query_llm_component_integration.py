#!/usr/bin/env python3
"""
Component Integration Test: LLMProcessor + QueryProcessor
=========================================================

This test demonstrates the better testing approach for interdependent modules.
Instead of testing LLMProcessor and QueryProcessor in isolation with mocks,
we test them together as they naturally depend on each other.

Why this approach is better:
1. Tests real interactions between components
2. Catches integration issues that unit tests miss
3. Validates the actual data flow between components
4. Tests the complete query processing pipeline
"""

import unittest
import os
import sys
import json
import logging
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.llm_processor import LLMProcessor
from modules.query_processor import QueryProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestQueryLLMComponentIntegration(unittest.TestCase):
    """Test LLMProcessor and QueryProcessor working together as a component"""
    
    def setUp(self):
        """Set up the component integration test"""
        # Mock the Gemini API to avoid external dependencies
        self.mock_gemini_response = {
            "language_analysis": {
                "detected_language": "en",
                "language_name": "English",
                "confidence_score": 0.95,
                "is_english": True,
                "script_type": "latin"
            },
            "query_processing": {
                "original_query": "machine learning algorithms",
                "corrected_original": "machine learning algorithms",
                "english_translation": "machine learning algorithms",
                "translation_needed": False,
                "correction_made": False,
                "processing_notes": "No corrections needed"
            },
            "intent_analysis": {
                "primary_intent": "find_research_papers",
                "secondary_intents": ["methodology_search"],
                "search_scope": "broad",
                "urgency_level": "medium",
                "academic_level": "graduate",
                "confidence": 0.9
            },
            "entity_extraction": {
                "people": [],
                "organizations": [],
                "locations": [],
                "technologies": ["machine learning", "algorithms"],
                "concepts": ["artificial intelligence", "data science"],
                "chemicals_materials": [],
                "medical_terms": [],
                "mathematical_terms": ["optimization", "statistics"],
                "time_periods": [],
                "publications": [],
                "fields_of_study": ["computer science", "artificial intelligence"]
            },
            "keyword_analysis": {
                "primary_keywords": ["machine learning", "algorithms", "ML"],
                "secondary_keywords": ["artificial intelligence", "neural networks", "deep learning"],
                "technical_terms": ["supervised learning", "unsupervised learning"],
                "original_language_keywords": ["machine learning", "algorithms"],
                "long_tail_keywords": ["machine learning algorithms research"],
                "alternative_spellings": ["ML", "AI"]
            },
            "semantic_expansion": {
                "synonyms": ["ML", "artificial intelligence", "AI"],
                "related_terms": ["neural networks", "deep learning", "data mining"],
                "broader_terms": ["artificial intelligence", "computer science"],
                "narrower_terms": ["supervised learning", "reinforcement learning"],
                "domain_specific_terms": ["gradient descent", "backpropagation"],
                "cross_linguistic_terms": [],
                "acronyms_abbreviations": ["ML", "AI", "DL"]
            },
            "academic_classification": {
                "primary_field": "computer science",
                "secondary_fields": ["mathematics", "statistics"],
                "specializations": ["artificial intelligence", "machine learning"],
                "interdisciplinary_connections": ["cognitive science", "neuroscience"],
                "research_methodologies": ["experimental", "theoretical"],
                "publication_types": ["journal_article", "conference_paper"]
            },
            "search_strategy": {
                "database_priorities": ["research_papers", "academic_library"],
                "resource_types": ["articles", "papers", "books"],
                "temporal_focus": "current",
                "geographical_scope": "global",
                "quality_indicators": ["peer_reviewed", "high_impact"],
                "search_complexity": "moderate"
            },
            "multilingual_considerations": {
                "preserve_original_terms": ["machine learning"],
                "cultural_context": [],
                "translation_challenges": [],
                "alternative_romanizations": []
            },
            "metadata": {
                "processing_timestamp": "2024-01-15T10:30:00Z",
                "model_version": "gemini-2.0-flash-exp",
                "analysis_confidence": 0.9,
                "processing_time_estimate": "2.5",
                "query_complexity": "moderate",
                "success": True
            }
        }
    
    @patch('modules.llm_processor.genai.configure')
    @patch('modules.llm_processor.genai.GenerativeModel')
    def test_complete_query_processing_pipeline(self, mock_model_class, mock_configure):
        """Test the complete query processing pipeline with LLM integration"""
        
        # Set up mock LLM response
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(self.mock_gemini_response)
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        # Set up environment
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            # Initialize the component (LLMProcessor + QueryProcessor)
            llm_processor = LLMProcessor()
            query_processor = QueryProcessor(llm_processor)
            
            # Test the complete pipeline
            test_query = "machine learning algorithms"
            result = query_processor.process_query(test_query)
            
            # Verify the result is not None
            self.assertIsNotNone(result)
            
            # Verify all required sections are present
            required_sections = [
                'language_analysis', 'query_processing', 'intent_analysis',
                'entity_extraction', 'keyword_analysis', 'semantic_expansion',
                'academic_classification', 'search_strategy', 'metadata'
            ]
            
            for section in required_sections:
                self.assertIn(section, result)
            
            # Verify the processed query contains expected data
            self.assertEqual(result['query_processing']['original_query'], test_query)
            self.assertEqual(result['intent_analysis']['primary_intent'], 'find_research_papers')
            self.assertTrue(result['metadata']['success'])
            
            # Verify backward compatibility fields are added
            self.assertIn('corrected_query', result)
            self.assertIn('intent', result)
            self.assertIn('keywords', result)
            self.assertIn('entities', result)
            
            # Verify LLM was called with correct parameters
            mock_model.generate_content.assert_called_once()
            call_args = mock_model.generate_content.call_args[0][0]
            self.assertIn(test_query, call_args)
    
    @patch('modules.llm_processor.genai.configure')
    @patch('modules.llm_processor.genai.GenerativeModel')
    def test_llm_failure_fallback_handling(self, mock_model_class, mock_configure):
        """Test how the component handles LLM failures gracefully"""
        
        # Set up mock LLM to fail
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("API Error")
        mock_model_class.return_value = mock_model
        
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            llm_processor = LLMProcessor()
            query_processor = QueryProcessor(llm_processor)
            
            # Test fallback behavior
            test_query = "test query"
            result = query_processor.process_query(test_query)
            
            # Should still return a result with fallback data
            self.assertIsNotNone(result)
            self.assertIn('metadata', result)
            self.assertIn('query_processing', result)
            self.assertEqual(result['query_processing']['original_query'], test_query)
            
            # Verify fallback processing
            self.assertIn('processing_notes', result['query_processing'])
            self.assertIn('fallback', result['query_processing']['processing_notes'].lower())
    
    @patch('modules.llm_processor.genai.configure')
    @patch('modules.llm_processor.genai.GenerativeModel')
    def test_multilingual_query_processing(self, mock_model_class, mock_configure):
        """Test multilingual query processing end-to-end"""
        
        # Create Arabic query response
        arabic_response = self.mock_gemini_response.copy()
        arabic_response['language_analysis'] = {
            "detected_language": "ar",
            "language_name": "Arabic",
            "confidence_score": 0.95,
            "is_english": False,
            "script_type": "arabic"
        }
        arabic_response['query_processing'] = {
            "original_query": "خوارزميات التعلم الآلي",
            "corrected_original": "خوارزميات التعلم الآلي",
            "english_translation": "machine learning algorithms",
            "translation_needed": True,
            "correction_made": False,
            "processing_notes": "Translated from Arabic to English"
        }
        
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(arabic_response)
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            llm_processor = LLMProcessor()
            query_processor = QueryProcessor(llm_processor)
            
            # Test Arabic query
            arabic_query = "خوارزميات التعلم الآلي"
            result = query_processor.process_query(arabic_query)
            
            # Verify multilingual processing
            self.assertIsNotNone(result)
            self.assertEqual(result['language_analysis']['detected_language'], 'ar')
            self.assertFalse(result['language_analysis']['is_english'])
            self.assertTrue(result['query_processing']['translation_needed'])
            self.assertEqual(result['query_processing']['english_translation'], 'machine learning algorithms')
    
    @patch('modules.llm_processor.genai.configure')
    @patch('modules.llm_processor.genai.GenerativeModel')
    def test_invalid_json_response_handling(self, mock_model_class, mock_configure):
        """Test handling of invalid JSON responses from LLM"""
        
        # Set up mock LLM to return invalid JSON
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This is not valid JSON"
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            llm_processor = LLMProcessor()
            query_processor = QueryProcessor(llm_processor)
            
            # Test invalid JSON handling
            test_query = "test query"
            result = query_processor.process_query(test_query)
            
            # Should still return a result with fallback data
            self.assertIsNotNone(result)
            self.assertIn('metadata', result)
            self.assertIn('query_processing', result)
            self.assertEqual(result['query_processing']['original_query'], test_query)
    
    @patch('modules.llm_processor.genai.configure')
    @patch('modules.llm_processor.genai.GenerativeModel')
    def test_component_data_flow_validation(self, mock_model_class, mock_configure):
        """Test that data flows correctly between LLMProcessor and QueryProcessor"""
        
        # Set up mock with specific data we want to track
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(self.mock_gemini_response)
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            llm_processor = LLMProcessor()
            query_processor = QueryProcessor(llm_processor)
            
            test_query = "AI research papers"
            result = query_processor.process_query(test_query)
            
            # Verify data flows correctly through both components
            # 1. LLMProcessor processes the query
            # 2. QueryProcessor validates and enhances the result
            # 3. QueryProcessor adds backward compatibility fields
            
            # Test LLMProcessor output is correctly processed
            self.assertIn('language_analysis', result)
            self.assertIn('intent_analysis', result)
            self.assertIn('semantic_expansion', result)
            
            # Test QueryProcessor enhancements
            self.assertIn('corrected_query', result)  # Added by QueryProcessor
            self.assertIn('intent', result)           # Added by QueryProcessor
            self.assertIn('keywords', result)         # Added by QueryProcessor
            self.assertIn('entities', result)         # Added by QueryProcessor
            
            # Test validation worked
            self.assertTrue(result['metadata']['success'])
            self.assertEqual(result['metadata']['original_query'], test_query)
    
    @patch('modules.llm_processor.genai.configure')
    @patch('modules.llm_processor.genai.GenerativeModel')
    def test_component_performance_characteristics(self, mock_model_class, mock_configure):
        """Test performance characteristics of the component"""
        
        def create_dynamic_response(query):
            """Create a dynamic response based on the query"""
            response = self.mock_gemini_response.copy()
            response['query_processing']['original_query'] = query
            response['query_processing']['corrected_original'] = query
            response['query_processing']['english_translation'] = query
            return json.dumps(response)
        
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            llm_processor = LLMProcessor()
            query_processor = QueryProcessor(llm_processor)
            
            # Test multiple queries to ensure consistent behavior
            test_queries = [
                "machine learning",
                "deep learning neural networks",
                "natural language processing",
                "computer vision algorithms",
                "reinforcement learning"
            ]
            
            results = []
            for query in test_queries:
                # Set up dynamic response for each query
                mock_response = MagicMock()
                mock_response.text = create_dynamic_response(query)
                mock_model.generate_content.return_value = mock_response
                
                result = query_processor.process_query(query)
                results.append(result)
                
                # Each result should be valid
                self.assertIsNotNone(result)
                self.assertTrue(result['metadata']['success'])
                self.assertEqual(result['query_processing']['original_query'], query)
            
            # Verify all queries were processed successfully
            self.assertEqual(len(results), len(test_queries))
            
            # Verify LLM was called for each query
            self.assertEqual(mock_model.generate_content.call_count, len(test_queries))

if __name__ == '__main__':
    unittest.main() 