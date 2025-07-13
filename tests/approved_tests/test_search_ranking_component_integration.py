#!/usr/bin/env python3
"""
Component Integration Test: SearchEngine + QueryProcessor + RankingEngine
========================================================================

This test demonstrates component integration testing for the core search pipeline.
These three modules work together as a natural component:
- QueryProcessor: Processes user queries with LLM analysis
- SearchEngine: Retrieves documents from multiple databases
- RankingEngine: Ranks and orders search results

Testing them together validates the complete search workflow.
"""

import unittest
import os
import sys
import json
import logging
from unittest.mock import patch, MagicMock, Mock

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.search_engine import SearchEngine
from modules.query_processor import QueryProcessor
from modules.ranking_engine import RankingEngine
from modules.llm_processor import LLMProcessor
from modules.database_manager import DatabaseManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestSearchRankingComponentIntegration(unittest.TestCase):
    """Test SearchEngine + QueryProcessor + RankingEngine working together"""
    
    def setUp(self):
        """Set up the search and ranking component integration test"""
        
        # Mock processed query from LLM
        self.mock_processed_query = {
            "language_analysis": {
                "detected_language": "en",
                "language_name": "English",
                "is_english": True
            },
            "query_processing": {
                "original_query": "machine learning algorithms",
                "corrected_original": "machine learning algorithms",
                "english_translation": "machine learning algorithms"
            },
            "intent_analysis": {
                "primary_intent": "find_research_papers",
                "secondary_intents": ["methodology_search"],
                "search_scope": "broad",
                "academic_level": "graduate",
                "confidence": 0.9
            },
            "entity_extraction": {
                "technologies": ["machine learning", "algorithms"],
                "concepts": ["artificial intelligence", "data science"],
                "fields_of_study": ["computer science", "artificial intelligence"]
            },
            "keyword_analysis": {
                "primary_keywords": ["machine learning", "algorithms", "ML"],
                "secondary_keywords": ["artificial intelligence", "neural networks"],
                "technical_terms": ["supervised learning", "unsupervised learning"]
            },
            "semantic_expansion": {
                "synonyms": ["ML", "AI"],
                "related_terms": ["neural networks", "deep learning"],
                "broader_terms": ["artificial intelligence"]
            },
            "academic_classification": {
                "primary_field": "computer science",
                "secondary_fields": ["mathematics", "statistics"]
            },
            "search_strategy": {
                "database_priorities": ["research_papers", "academic_library"],
                "resource_types": ["articles", "papers"],
                "quality_indicators": ["peer_reviewed", "high_impact"]
            },
            "metadata": {
                "success": True,
                "original_query": "machine learning algorithms"
            },
            # Backward compatibility fields
            "corrected_query": "machine learning algorithms",
            "intent": {
                "primary_intent": "find_research_papers",
                "confidence": 0.9
            },
            "keywords": {
                "primary": ["machine learning", "algorithms"],
                "secondary": ["artificial intelligence", "neural networks"]
            },
            "entities": {
                "technologies": ["machine learning", "algorithms"],
                "concepts": ["artificial intelligence", "data science"]
            }
        }
        
        # Mock search results
        self.mock_search_results = {
            "results": [
                {
                    "id": "paper1",
                    "title": "Deep Learning for Computer Vision",
                    "description": "A comprehensive survey of deep learning applications in computer vision",
                    "authors": ["John Smith", "Jane Doe"],
                    "type": "paper",
                    "database": "research_papers",
                    "collection": "papers",
                    "year": 2023,
                    "citations": 150
                },
                {
                    "id": "paper2", 
                    "title": "Machine Learning Algorithms Overview",
                    "description": "An overview of various machine learning algorithms and their applications",
                    "authors": ["Alice Johnson"],
                    "type": "paper",
                    "database": "academic_library",
                    "collection": "books",
                    "year": 2022,
                    "citations": 89
                },
                {
                    "id": "expert1",
                    "title": "Dr. Michael Chen - AI Research Expert",
                    "description": "Professor of Computer Science specializing in artificial intelligence",
                    "type": "expert",
                    "database": "experts_system",
                    "collection": "experts",
                    "institution": "MIT",
                    "research_areas": ["machine learning", "neural networks"]
                }
            ],
            "total_results": 3,
            "result_counts": {
                "research_papers": 2,
                "experts_system": 1,
                "academic_library": 1
            }
        }
    
    @patch('modules.llm_processor.genai.configure')
    @patch('modules.llm_processor.genai.GenerativeModel')
    @patch('modules.database_manager.DatabaseManager.get_database')
    def test_complete_search_and_ranking_pipeline(self, mock_get_db, mock_model_class, mock_configure):
        """Test the complete search and ranking pipeline"""
        
        # Set up LLM mocks
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "language_analysis": self.mock_processed_query["language_analysis"],
            "query_processing": self.mock_processed_query["query_processing"],
            "intent_analysis": self.mock_processed_query["intent_analysis"],
            "entity_extraction": self.mock_processed_query["entity_extraction"],
            "keyword_analysis": self.mock_processed_query["keyword_analysis"],
            "semantic_expansion": self.mock_processed_query["semantic_expansion"],
            "academic_classification": self.mock_processed_query["academic_classification"],
            "search_strategy": self.mock_processed_query["search_strategy"],
            "multilingual_considerations": {
                "preserve_original_terms": [],
                "cultural_context": [],
                "translation_challenges": [],
                "alternative_romanizations": []
            },
            "metadata": self.mock_processed_query["metadata"]
        })
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        # Set up database mocks with proper MongoDB cursor simulation
        mock_db = MagicMock()
        mock_collection = MagicMock()
        
        # Create mock cursor that properly implements .limit() and iteration
        mock_cursor = MagicMock()
        mock_cursor.limit.return_value = [
            {
                "_id": "paper1",
                "title": "Deep Learning for Computer Vision",
                "description": "A comprehensive survey of deep learning applications",
                "authors": ["John Smith", "Jane Doe"],
                "year": 2023,
                "citations": 150
            },
            {
                "_id": "paper2",
                "title": "Machine Learning Algorithms Overview", 
                "description": "An overview of various machine learning algorithms",
                "authors": ["Alice Johnson"],
                "year": 2022,
                "citations": 89
            }
        ]
        
        # Make find() return the mock cursor
        mock_collection.find.return_value = mock_cursor
        mock_db.__getitem__.return_value = mock_collection
        mock_get_db.return_value = mock_db
        
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            # Initialize the complete search component
            llm_processor = LLMProcessor()
            query_processor = QueryProcessor(llm_processor)
            db_manager = DatabaseManager()
            search_engine = SearchEngine(db_manager, use_preindex=False)
            ranking_engine = RankingEngine(llm_processor, use_embedding=False, use_ltr=False)
            
            # Test the complete pipeline
            user_query = "machine learning algorithms"
            
            # Step 1: Process query
            processed_query = query_processor.process_query(user_query)
            self.assertIsNotNone(processed_query)
            self.assertTrue(processed_query['metadata']['success'])
            
            # Step 2: Search for results
            search_results = search_engine.search_all_databases(processed_query, hybrid_search=False)
            self.assertIsNotNone(search_results)
            self.assertIn('results', search_results)
            
            # Step 3: Rank results
            ranked_results = ranking_engine.rank_search_results(
                search_results, 
                processed_query,
                ranking_mode="traditional"
            )
            
            # Verify the complete pipeline works
            self.assertIsNotNone(ranked_results)
            self.assertIn('results', ranked_results)
            self.assertGreater(len(ranked_results['results']), 0)
            
            # Verify ranking scores are added (using 'ranking_score' field)
            for result in ranked_results['results']:
                self.assertIn('ranking_score', result)
                self.assertIsInstance(result['ranking_score'], (int, float))
                
            # Verify results are properly ranked (higher scores first)
            scores = [r['ranking_score'] for r in ranked_results['results']]
            self.assertEqual(scores, sorted(scores, reverse=True))
    
    @patch('modules.llm_processor.genai.configure')
    @patch('modules.llm_processor.genai.GenerativeModel')
    @patch('modules.database_manager.DatabaseManager.get_database')
    def test_search_component_with_different_intents(self, mock_get_db, mock_model_class, mock_configure):
        """Test search component with different query intents"""
        
        test_cases = [
            {
                "intent": "find_experts",
                "query": "AI researchers at MIT",
                "expected_db_priority": "experts_system"
            },
            {
                "intent": "find_equipment",
                "query": "GPU clusters for machine learning",
                "expected_db_priority": "laboratories"
            },
            {
                "intent": "find_funding",
                "query": "AI research grants",
                "expected_db_priority": "funding"
            }
        ]
        
        for case in test_cases:
            with self.subTest(intent=case["intent"]):
                # Update mock response for this intent
                mock_response_data = self.mock_processed_query.copy()
                mock_response_data["intent_analysis"]["primary_intent"] = case["intent"]
                mock_response_data["query_processing"]["original_query"] = case["query"]
                mock_response_data["corrected_query"] = case["query"]
                
                mock_model = MagicMock()
                mock_response = MagicMock()
                mock_response.text = json.dumps({
                    "language_analysis": mock_response_data["language_analysis"],
                    "query_processing": mock_response_data["query_processing"],
                    "intent_analysis": mock_response_data["intent_analysis"],
                    "entity_extraction": mock_response_data["entity_extraction"],
                    "keyword_analysis": mock_response_data["keyword_analysis"],
                    "semantic_expansion": mock_response_data["semantic_expansion"],
                    "academic_classification": mock_response_data["academic_classification"],
                    "search_strategy": mock_response_data["search_strategy"],
                    "multilingual_considerations": {
                        "preserve_original_terms": [],
                        "cultural_context": [],
                        "translation_challenges": [],
                        "alternative_romanizations": []
                    },
                    "metadata": mock_response_data["metadata"]
                })
                mock_model.generate_content.return_value = mock_response
                mock_model_class.return_value = mock_model
                
                # Set up database mock with proper cursor simulation
                mock_db = MagicMock()
                mock_collection = MagicMock()
                
                # Create mock cursor
                mock_cursor = MagicMock()
                mock_cursor.limit.return_value = [
                    {
                        "_id": f"result_{case['intent']}_1",
                        "title": f"Sample {case['intent']} result",
                        "description": f"Description for {case['intent']}",
                        "type": case["intent"].replace("find_", "")
                    }
                ]
                
                mock_collection.find.return_value = mock_cursor
                mock_db.__getitem__.return_value = mock_collection
                mock_get_db.return_value = mock_db
                
                with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
                    # Test the component with different intents
                    llm_processor = LLMProcessor()
                    query_processor = QueryProcessor(llm_processor)
                    db_manager = DatabaseManager()
                    search_engine = SearchEngine(db_manager, use_preindex=False)
                    ranking_engine = RankingEngine(llm_processor, use_embedding=False, use_ltr=False)
                    
                    # Process query
                    processed_query = query_processor.process_query(case["query"])
                    self.assertEqual(processed_query["intent_analysis"]["primary_intent"], case["intent"])
                    
                    # Search and rank
                    search_results = search_engine.search_all_databases(processed_query, hybrid_search=False)
                    ranked_results = ranking_engine.rank_search_results(
                        search_results, 
                        processed_query,
                        ranking_mode="traditional"
                    )
                    
                    # Verify intent-specific processing
                    self.assertIsNotNone(ranked_results)
                    self.assertIn('results', ranked_results)
    
    @patch('modules.llm_processor.genai.configure')
    @patch('modules.llm_processor.genai.GenerativeModel')
    @patch('modules.database_manager.DatabaseManager.get_database')
    def test_search_component_error_handling(self, mock_get_db, mock_model_class, mock_configure):
        """Test search component error handling"""
        
        # Test LLM failure
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("LLM API Error")
        mock_model_class.return_value = mock_model
        
        # Set up database mock with proper cursor simulation
        mock_db = MagicMock()
        mock_collection = MagicMock()
        
        # Create mock cursor for empty results
        mock_cursor = MagicMock()
        mock_cursor.limit.return_value = []
        
        mock_collection.find.return_value = mock_cursor
        mock_db.__getitem__.return_value = mock_collection
        mock_get_db.return_value = mock_db
        
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            # Initialize components
            llm_processor = LLMProcessor()
            query_processor = QueryProcessor(llm_processor)
            db_manager = DatabaseManager()
            search_engine = SearchEngine(db_manager, use_preindex=False)
            ranking_engine = RankingEngine(llm_processor, use_embedding=False, use_ltr=False)
            
            # Test with LLM failure
            processed_query = query_processor.process_query("test query")
            
            # Should still work with fallback processing
            self.assertIsNotNone(processed_query)
            self.assertIn('query_processing', processed_query)
            
            # Search should still work
            search_results = search_engine.search_all_databases(processed_query, hybrid_search=False)
            self.assertIsNotNone(search_results)
            
            # Ranking should still work
            ranked_results = ranking_engine.rank_search_results(
                search_results, 
                processed_query,
                ranking_mode="traditional"
            )
            self.assertIsNotNone(ranked_results)
    
    @patch('modules.llm_processor.genai.configure')
    @patch('modules.llm_processor.genai.GenerativeModel')
    @patch('modules.database_manager.DatabaseManager.get_database')
    def test_search_component_empty_results_handling(self, mock_get_db, mock_model_class, mock_configure):
        """Test search component with empty results"""
        
        # Set up LLM mock
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "language_analysis": self.mock_processed_query["language_analysis"],
            "query_processing": self.mock_processed_query["query_processing"],
            "intent_analysis": self.mock_processed_query["intent_analysis"],
            "entity_extraction": self.mock_processed_query["entity_extraction"],
            "keyword_analysis": self.mock_processed_query["keyword_analysis"],
            "semantic_expansion": self.mock_processed_query["semantic_expansion"],
            "academic_classification": self.mock_processed_query["academic_classification"],
            "search_strategy": self.mock_processed_query["search_strategy"],
            "multilingual_considerations": {
                "preserve_original_terms": [],
                "cultural_context": [],
                "translation_challenges": [],
                "alternative_romanizations": []
            },
            "metadata": self.mock_processed_query["metadata"]
        })
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        # Set up database mock to return empty results with proper cursor simulation
        mock_db = MagicMock()
        mock_collection = MagicMock()
        
        # Create mock cursor for empty results
        mock_cursor = MagicMock()
        mock_cursor.limit.return_value = []
        
        mock_collection.find.return_value = mock_cursor
        mock_db.__getitem__.return_value = mock_collection
        mock_get_db.return_value = mock_db
        
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            # Initialize components
            llm_processor = LLMProcessor()
            query_processor = QueryProcessor(llm_processor)
            db_manager = DatabaseManager()
            search_engine = SearchEngine(db_manager, use_preindex=False)
            ranking_engine = RankingEngine(llm_processor, use_embedding=False, use_ltr=False)
            
            # Test with empty results
            processed_query = query_processor.process_query("very rare query")
            search_results = search_engine.search_all_databases(processed_query, hybrid_search=False)
            ranked_results = ranking_engine.rank_search_results(
                search_results, 
                processed_query,
                ranking_mode="traditional"
            )
            
            # Should handle empty results gracefully
            self.assertIsNotNone(ranked_results)
            self.assertIn('results', ranked_results)
            self.assertEqual(len(ranked_results['results']), 0)
            self.assertEqual(ranked_results['total_results'], 0)
    
    @patch('modules.llm_processor.genai.configure')
    @patch('modules.llm_processor.genai.GenerativeModel')
    @patch('modules.database_manager.DatabaseManager.get_database')
    def test_search_component_data_flow_validation(self, mock_get_db, mock_model_class, mock_configure):
        """Test that data flows correctly through the search component"""
        
        # Set up LLM mock
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "language_analysis": self.mock_processed_query["language_analysis"],
            "query_processing": self.mock_processed_query["query_processing"],
            "intent_analysis": self.mock_processed_query["intent_analysis"],
            "entity_extraction": self.mock_processed_query["entity_extraction"],
            "keyword_analysis": self.mock_processed_query["keyword_analysis"],
            "semantic_expansion": self.mock_processed_query["semantic_expansion"],
            "academic_classification": self.mock_processed_query["academic_classification"],
            "search_strategy": self.mock_processed_query["search_strategy"],
            "multilingual_considerations": {
                "preserve_original_terms": [],
                "cultural_context": [],
                "translation_challenges": [],
                "alternative_romanizations": []
            },
            "metadata": self.mock_processed_query["metadata"]
        })
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        # Set up database mock with proper cursor simulation
        mock_db = MagicMock()
        mock_collection = MagicMock()
        
        # Create mock cursor
        mock_cursor = MagicMock()
        mock_cursor.limit.return_value = [
            {
                "_id": "test_paper",
                "title": "Test Paper",
                "description": "A test paper for validation",
                "authors": ["Test Author"],
                "year": 2023
            }
        ]
        
        mock_collection.find.return_value = mock_cursor
        mock_db.__getitem__.return_value = mock_collection
        mock_get_db.return_value = mock_db
        
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            # Initialize components
            llm_processor = LLMProcessor()
            query_processor = QueryProcessor(llm_processor)
            db_manager = DatabaseManager()
            search_engine = SearchEngine(db_manager, use_preindex=False)
            ranking_engine = RankingEngine(llm_processor, use_embedding=False, use_ltr=False)
            
            # Test data flow through component
            user_query = "test query"
            
            # Step 1: Query processing
            processed_query = query_processor.process_query(user_query)
            self.assertIn('corrected_query', processed_query)
            self.assertIn('intent', processed_query)
            self.assertIn('keywords', processed_query)
            
            # Step 2: Search using processed query
            search_results = search_engine.search_all_databases(processed_query, hybrid_search=False)
            self.assertIn('results', search_results)
            
            # Step 3: Ranking using both search results and processed query
            ranked_results = ranking_engine.rank_search_results(
                search_results, 
                processed_query,
                ranking_mode="traditional"
            )
            
            # Verify data flow
            self.assertIsNotNone(ranked_results)
            self.assertIn('results', ranked_results)
            
            # Verify each result has proper structure
            for result in ranked_results['results']:
                self.assertIn('id', result)
                self.assertIn('title', result)
                self.assertIn('ranking_score', result)
                self.assertIn('score_breakdown', result)

if __name__ == '__main__':
    unittest.main() 