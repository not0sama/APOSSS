#!/usr/bin/env python3
"""
Unit tests for APOSSS LLMProcessor class
Tests query analysis, keyword extraction, semantic expansion, and academic classification
"""
import sys
import os
import unittest
import logging
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestLLMProcessor(unittest.TestCase):
    """Unit tests for LLMProcessor class"""

    def setUp(self):
        """Set up test fixtures"""
        # Sample test data
        self.test_query = "machine learning for medical diagnosis"
        self.test_queries = [
            "machine learning applications in healthcare",
            "deep learning for medical imaging",
            "artificial intelligence in drug discovery",
            "neural networks for disease prediction",
            "AI ethics in healthcare systems"
        ]
        
        # Mock LLM response
        self.mock_llm_response = {
            "language_detection": {
                "detected_language": "en",
                "confidence": 0.99,
                "original_query": "machine learning for medical diagnosis",
                "english_translation": "machine learning for medical diagnosis"
            },
            "query_intent": {
                "primary_intent": "find_research",
                "secondary_intents": ["find_papers", "find_experts"],
                "confidence": 0.85,
                "intent_explanation": "User is seeking research materials about machine learning applications in medical diagnosis"
            },
            "entity_extraction": {
                "people": [],
                "organizations": [],
                "locations": [],
                "technologies": ["machine learning", "artificial intelligence"],
                "concepts": ["medical diagnosis", "healthcare applications"],
                "academic_fields": ["Computer Science", "Medicine", "Healthcare Technology"]
            },
            "keyword_extraction": {
                "primary_keywords": ["machine learning", "medical", "diagnosis", "healthcare"],
                "secondary_keywords": ["AI", "artificial intelligence", "ML", "diagnostic", "clinical"],
                "technical_terms": ["supervised learning", "neural networks", "classification", "prediction"],
                "original_language_keywords": [],
                "long_tail_keywords": ["machine learning in medical diagnosis", "AI for healthcare"],
                "alternative_spellings": ["ML", "AI", "diagnostics"]
            },
            "semantic_expansion": {
                "synonyms": ["artificial intelligence", "automated diagnosis", "computer-aided diagnosis"],
                "related_terms": ["deep learning", "neural networks", "pattern recognition"],
                "broader_terms": ["artificial intelligence", "healthcare technology", "medical informatics"],
                "narrower_terms": ["medical image analysis", "clinical decision support", "diagnostic algorithms"],
                "domain_specific_terms": ["radiology AI", "pathology automation", "clinical ML"],
                "cross_linguistic_terms": ["apprentissage automatique", "機械学習", "aprendizaje automático"],
                "acronyms_abbreviations": ["ML", "AI", "CAD", "CDX"]
            },
            "academic_classification": {
                "primary_field": "Computer Science",
                "secondary_fields": ["Medicine", "Healthcare Technology", "Biomedical Engineering"],
                "specializations": ["Machine Learning", "Medical Informatics", "Healthcare AI"],
                "interdisciplinary_connections": ["Biostatistics", "Medical Physics", "Health Informatics"],
                "research_methodologies": ["supervised learning", "cross-validation", "clinical trials"],
                "publication_types": ["journal articles", "conference papers", "clinical studies"]
            },
            "search_strategy": {
                "database_priorities": ["research_papers", "academic_library", "experts_system"],
                "resource_types": ["articles", "journals", "conferences", "experts"],
                "temporal_focus": "recent",
                "geographical_scope": "international",
                "quality_indicators": ["peer_reviewed", "high_impact", "recent"],
                "search_complexity": "moderate"
            }
        }

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_api_key'})
    @patch('modules.llm_processor.genai')
    def test_llm_processor_initialization(self, mock_genai):
        """Test LLMProcessor initialization"""
        from modules.llm_processor import LLMProcessor
        
        # Mock Gemini configuration
        mock_model = Mock()
        mock_genai.GenerativeModel.return_value = mock_model
        
        # Test successful initialization
        processor = LLMProcessor()
        
        self.assertIsNotNone(processor)
        self.assertEqual(processor.api_key, 'test_api_key')
        self.assertEqual(processor.model, mock_model)
        
        # Verify Gemini configuration
        mock_genai.configure.assert_called_once_with(api_key='test_api_key')
        mock_genai.GenerativeModel.assert_called_once()

    def test_llm_processor_initialization_no_api_key(self):
        """Test LLMProcessor initialization without API key"""
        from modules.llm_processor import LLMProcessor
        
        # Test initialization without API key
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as context:
                LLMProcessor()
            
            self.assertIn("GEMINI_API_KEY", str(context.exception))

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_api_key'})
    @patch('modules.llm_processor.genai')
    def test_create_query_analysis_prompt(self, mock_genai):
        """Test query analysis prompt creation"""
        from modules.llm_processor import LLMProcessor
        
        # Mock Gemini configuration
        mock_model = Mock()
        mock_genai.GenerativeModel.return_value = mock_model
        
        processor = LLMProcessor()
        
        # Test prompt creation
        prompt = processor.create_query_analysis_prompt(self.test_query)
        
        self.assertIsInstance(prompt, str)
        self.assertIn(self.test_query, prompt)
        self.assertIn("language_detection", prompt)
        self.assertIn("query_intent", prompt)
        self.assertIn("entity_extraction", prompt)
        self.assertIn("keyword_extraction", prompt)
        self.assertIn("semantic_expansion", prompt)
        self.assertIn("academic_classification", prompt)
        self.assertIn("search_strategy", prompt)

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_api_key'})
    @patch('modules.llm_processor.genai')
    def test_process_query_success(self, mock_genai):
        """Test successful query processing"""
        from modules.llm_processor import LLMProcessor
        
        # Mock Gemini model and response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps(self.mock_llm_response)
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        processor = LLMProcessor()
        
        # Test query processing
        result = processor.process_query(self.test_query)
        
        self.assertIsInstance(result, dict)
        self.assertIn('language_detection', result)
        self.assertIn('query_intent', result)
        self.assertIn('entity_extraction', result)
        self.assertIn('keyword_extraction', result)
        self.assertIn('semantic_expansion', result)
        self.assertIn('academic_classification', result)
        self.assertIn('search_strategy', result)
        self.assertIn('_metadata', result)
        
        # Check metadata
        self.assertEqual(result['_metadata']['original_query'], self.test_query)
        self.assertIn('processing_time', result['_metadata'])

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_api_key'})
    @patch('modules.llm_processor.genai')
    def test_process_query_invalid_json(self, mock_genai):
        """Test query processing with invalid JSON response"""
        from modules.llm_processor import LLMProcessor
        
        # Mock Gemini model with invalid JSON response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Invalid JSON response"
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        processor = LLMProcessor()
        
        # Test query processing with invalid JSON
        result = processor.process_query(self.test_query)
        
        self.assertIsInstance(result, dict)
        self.assertIn('error', result)
        self.assertIn('_metadata', result)

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_api_key'})
    @patch('modules.llm_processor.genai')
    def test_process_query_api_error(self, mock_genai):
        """Test query processing with API error"""
        from modules.llm_processor import LLMProcessor
        
        # Mock Gemini model with API error
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("API Error")
        mock_genai.GenerativeModel.return_value = mock_model
        
        processor = LLMProcessor()
        
        # Test query processing with API error
        result = processor.process_query(self.test_query)
        
        self.assertIsInstance(result, dict)
        self.assertIn('error', result)
        self.assertIn('_metadata', result)

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_api_key'})
    @patch('modules.llm_processor.genai')
    def test_process_multiple_queries(self, mock_genai):
        """Test processing multiple queries"""
        from modules.llm_processor import LLMProcessor
        
        # Mock Gemini model and response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps(self.mock_llm_response)
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        processor = LLMProcessor()
        
        # Test multiple query processing
        for query in self.test_queries:
            result = processor.process_query(query)
            
            self.assertIsInstance(result, dict)
            self.assertIn('_metadata', result)
            self.assertEqual(result['_metadata']['original_query'], query)

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_api_key'})
    @patch('modules.llm_processor.genai')
    def test_language_detection(self, mock_genai):
        """Test language detection functionality"""
        from modules.llm_processor import LLMProcessor
        
        # Mock Gemini model and response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps(self.mock_llm_response)
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        processor = LLMProcessor()
        
        # Test language detection
        result = processor.process_query("machine learning")
        
        self.assertIn('language_detection', result)
        self.assertIn('detected_language', result['language_detection'])
        self.assertIn('confidence', result['language_detection'])
        self.assertIn('original_query', result['language_detection'])

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_api_key'})
    @patch('modules.llm_processor.genai')
    def test_intent_classification(self, mock_genai):
        """Test intent classification functionality"""
        from modules.llm_processor import LLMProcessor
        
        # Mock Gemini model and response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps(self.mock_llm_response)
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        processor = LLMProcessor()
        
        # Test intent classification
        result = processor.process_query("find research papers about machine learning")
        
        self.assertIn('query_intent', result)
        self.assertIn('primary_intent', result['query_intent'])
        self.assertIn('confidence', result['query_intent'])
        self.assertIn('intent_explanation', result['query_intent'])

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_api_key'})
    @patch('modules.llm_processor.genai')
    def test_entity_extraction(self, mock_genai):
        """Test entity extraction functionality"""
        from modules.llm_processor import LLMProcessor
        
        # Mock Gemini model and response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps(self.mock_llm_response)
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        processor = LLMProcessor()
        
        # Test entity extraction
        result = processor.process_query("machine learning research at MIT")
        
        self.assertIn('entity_extraction', result)
        self.assertIn('people', result['entity_extraction'])
        self.assertIn('organizations', result['entity_extraction'])
        self.assertIn('locations', result['entity_extraction'])
        self.assertIn('technologies', result['entity_extraction'])
        self.assertIn('concepts', result['entity_extraction'])

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_api_key'})
    @patch('modules.llm_processor.genai')
    def test_keyword_extraction(self, mock_genai):
        """Test keyword extraction functionality"""
        from modules.llm_processor import LLMProcessor
        
        # Mock Gemini model and response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps(self.mock_llm_response)
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        processor = LLMProcessor()
        
        # Test keyword extraction
        result = processor.process_query("machine learning for medical diagnosis")
        
        self.assertIn('keyword_extraction', result)
        self.assertIn('primary_keywords', result['keyword_extraction'])
        self.assertIn('secondary_keywords', result['keyword_extraction'])
        self.assertIn('technical_terms', result['keyword_extraction'])
        self.assertIn('long_tail_keywords', result['keyword_extraction'])

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_api_key'})
    @patch('modules.llm_processor.genai')
    def test_semantic_expansion(self, mock_genai):
        """Test semantic expansion functionality"""
        from modules.llm_processor import LLMProcessor
        
        # Mock Gemini model and response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps(self.mock_llm_response)
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        processor = LLMProcessor()
        
        # Test semantic expansion
        result = processor.process_query("machine learning")
        
        self.assertIn('semantic_expansion', result)
        self.assertIn('synonyms', result['semantic_expansion'])
        self.assertIn('related_terms', result['semantic_expansion'])
        self.assertIn('broader_terms', result['semantic_expansion'])
        self.assertIn('narrower_terms', result['semantic_expansion'])
        self.assertIn('domain_specific_terms', result['semantic_expansion'])

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_api_key'})
    @patch('modules.llm_processor.genai')
    def test_academic_classification(self, mock_genai):
        """Test academic classification functionality"""
        from modules.llm_processor import LLMProcessor
        
        # Mock Gemini model and response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps(self.mock_llm_response)
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        processor = LLMProcessor()
        
        # Test academic classification
        result = processor.process_query("machine learning research")
        
        self.assertIn('academic_classification', result)
        self.assertIn('primary_field', result['academic_classification'])
        self.assertIn('secondary_fields', result['academic_classification'])
        self.assertIn('specializations', result['academic_classification'])
        self.assertIn('interdisciplinary_connections', result['academic_classification'])
        self.assertIn('research_methodologies', result['academic_classification'])

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_api_key'})
    @patch('modules.llm_processor.genai')
    def test_search_strategy(self, mock_genai):
        """Test search strategy functionality"""
        from modules.llm_processor import LLMProcessor
        
        # Mock Gemini model and response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps(self.mock_llm_response)
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        processor = LLMProcessor()
        
        # Test search strategy
        result = processor.process_query("find recent papers on machine learning")
        
        self.assertIn('search_strategy', result)
        self.assertIn('database_priorities', result['search_strategy'])
        self.assertIn('resource_types', result['search_strategy'])
        self.assertIn('temporal_focus', result['search_strategy'])
        self.assertIn('geographical_scope', result['search_strategy'])
        self.assertIn('quality_indicators', result['search_strategy'])

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_api_key'})
    @patch('modules.llm_processor.genai')
    def test_empty_query_handling(self, mock_genai):
        """Test handling of empty queries"""
        from modules.llm_processor import LLMProcessor
        
        # Mock Gemini model and response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps(self.mock_llm_response)
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        processor = LLMProcessor()
        
        # Test with empty query
        result = processor.process_query("")
        
        self.assertIsInstance(result, dict)
        self.assertIn('_metadata', result)

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_api_key'})
    @patch('modules.llm_processor.genai')
    def test_very_long_query_handling(self, mock_genai):
        """Test handling of very long queries"""
        from modules.llm_processor import LLMProcessor
        
        # Mock Gemini model and response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps(self.mock_llm_response)
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        processor = LLMProcessor()
        
        # Test with very long query
        long_query = "machine learning " * 1000  # Very long query
        result = processor.process_query(long_query)
        
        self.assertIsInstance(result, dict)
        self.assertIn('_metadata', result)

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_api_key'})
    @patch('modules.llm_processor.genai')
    def test_special_characters_in_query(self, mock_genai):
        """Test handling of special characters in queries"""
        from modules.llm_processor import LLMProcessor
        
        # Mock Gemini model and response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps(self.mock_llm_response)
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        processor = LLMProcessor()
        
        # Test with special characters
        special_query = "machine learning & AI: deep learning @2023 #research"
        result = processor.process_query(special_query)
        
        self.assertIsInstance(result, dict)
        self.assertIn('_metadata', result)

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_api_key'})
    @patch('modules.llm_processor.genai')
    def test_multilingual_query_processing(self, mock_genai):
        """Test processing of multilingual queries"""
        from modules.llm_processor import LLMProcessor
        
        # Mock Gemini model and response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps(self.mock_llm_response)
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        processor = LLMProcessor()
        
        # Test multilingual queries
        multilingual_queries = [
            "machine learning",  # English
            "aprendizaje automático",  # Spanish
            "apprentissage automatique",  # French
            "機械学習",  # Japanese
            "机器学习"  # Chinese
        ]
        
        for query in multilingual_queries:
            result = processor.process_query(query)
            
            self.assertIsInstance(result, dict)
            self.assertIn('language_detection', result)
            self.assertIn('_metadata', result)

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_api_key'})
    @patch('modules.llm_processor.genai')
    def test_query_refinement(self, mock_genai):
        """Test query refinement functionality"""
        from modules.llm_processor import LLMProcessor
        
        # Mock Gemini model and response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps(self.mock_llm_response)
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        processor = LLMProcessor()
        
        # Test query refinement if available
        if hasattr(processor, 'refine_query'):
            refined_query = processor.refine_query("ML in healthcare")
            
            self.assertIsInstance(refined_query, str)
            self.assertNotEqual(refined_query, "ML in healthcare")

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_api_key'})
    @patch('modules.llm_processor.genai')
    def test_query_similarity(self, mock_genai):
        """Test query similarity functionality"""
        from modules.llm_processor import LLMProcessor
        
        # Mock Gemini model and response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps(self.mock_llm_response)
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        processor = LLMProcessor()
        
        # Test query similarity if available
        if hasattr(processor, 'calculate_query_similarity'):
            similarity = processor.calculate_query_similarity(
                "machine learning", "artificial intelligence"
            )
            
            self.assertIsInstance(similarity, (int, float))
            self.assertGreaterEqual(similarity, 0)
            self.assertLessEqual(similarity, 1)

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_api_key'})
    @patch('modules.llm_processor.genai')
    def test_caching_functionality(self, mock_genai):
        """Test caching functionality"""
        from modules.llm_processor import LLMProcessor
        
        # Mock Gemini model and response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps(self.mock_llm_response)
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        processor = LLMProcessor()
        
        # Test caching if available
        if hasattr(processor, 'cache_query_result'):
            processor.cache_query_result(self.test_query, self.mock_llm_response)
        
        if hasattr(processor, 'get_cached_result'):
            cached_result = processor.get_cached_result(self.test_query)
            if cached_result:
                self.assertIsInstance(cached_result, dict)


class TestLLMProcessorIntegration(unittest.TestCase):
    """Integration tests for LLMProcessor with real components"""

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_api_key'})
    def test_llm_processor_with_real_gemini(self):
        """Test LLMProcessor with actual Gemini API (if available)"""
        try:
            import google.generativeai as genai
            from modules.llm_processor import LLMProcessor
            
            # This test only runs if Google Generative AI is available
            # Note: This would require a real API key for full integration testing
            processor = LLMProcessor()
            
            # Test that Gemini is properly initialized
            self.assertIsNotNone(processor.model)
            
        except ImportError:
            self.skipTest("Google Generative AI not available")

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_api_key'})
    def test_llm_processor_performance(self):
        """Test LLMProcessor performance characteristics"""
        from modules.llm_processor import LLMProcessor
        
        # Test performance metrics if available
        if hasattr(LLMProcessor, 'get_performance_metrics'):
            processor = LLMProcessor()
            metrics = processor.get_performance_metrics()
            
            self.assertIsInstance(metrics, dict)


if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2) 