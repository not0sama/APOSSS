#!/usr/bin/env python3
"""
Component Integration Test: ML Pipeline Component
================================================

This test demonstrates component integration testing for the ML-powered ranking pipeline.
These modules work together as a natural ML component:
- RankingEngine: Orchestrates multiple ranking algorithms
- EmbeddingRanker: Provides semantic similarity ranking
- LTRRanker: Provides learning-to-rank capabilities
- KnowledgeGraph: Provides graph-based authority features

Testing them together validates the complete ML ranking workflow.
"""

import unittest
import os
import sys
import json
import logging
import numpy as np
from unittest.mock import patch, MagicMock, Mock

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.ranking_engine import RankingEngine
from modules.llm_processor import LLMProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestMLPipelineComponentIntegration(unittest.TestCase):
    """Test ML Pipeline components working together"""
    
    def setUp(self):
        """Set up the ML pipeline component integration test"""
        
        # Mock processed query for ML pipeline
        self.mock_processed_query = {
            "query_processing": {
                "original_query": "deep learning neural networks",
                "english_translation": "deep learning neural networks"
            },
            "intent_analysis": {
                "primary_intent": "find_research_papers",
                "confidence": 0.9
            },
            "keyword_analysis": {
                "primary_keywords": ["deep learning", "neural networks"],
                "secondary_keywords": ["machine learning", "AI"]
            },
            "entity_extraction": {
                "technologies": ["deep learning", "neural networks"],
                "concepts": ["artificial intelligence", "machine learning"]
            },
            "semantic_expansion": {
                "synonyms": ["deep learning", "neural networks"],
                "related_terms": ["CNN", "RNN", "transformer"]
            },
            "academic_classification": {
                "primary_field": "computer science",
                "secondary_fields": ["artificial intelligence"]
            },
            "metadata": {
                "success": True,
                "original_query": "deep learning neural networks"
            },
            # Backward compatibility
            "corrected_query": "deep learning neural networks",
            "intent": {"primary_intent": "find_research_papers"},
            "keywords": {"primary": ["deep learning", "neural networks"]},
            "entities": {"technologies": ["deep learning", "neural networks"]}
        }
        
        # Mock search results for ML pipeline
        self.mock_search_results = {
            "results": [
                {
                    "id": "paper1",
                    "title": "Deep Learning for Computer Vision",
                    "description": "A comprehensive survey of deep learning applications in computer vision tasks including image classification and object detection",
                    "authors": ["John Smith", "Jane Doe"],
                    "type": "paper",
                    "year": 2023,
                    "citations": 150,
                    "keywords": ["deep learning", "computer vision", "neural networks"]
                },
                {
                    "id": "paper2",
                    "title": "Neural Network Architectures",
                    "description": "An overview of various neural network architectures including CNNs, RNNs, and Transformers",
                    "authors": ["Alice Johnson", "Bob Chen"],
                    "type": "paper",
                    "year": 2022,
                    "citations": 89,
                    "keywords": ["neural networks", "deep learning", "architecture"]
                },
                {
                    "id": "paper3",
                    "title": "Machine Learning Fundamentals",
                    "description": "Basic concepts and algorithms in machine learning",
                    "authors": ["Carol Williams"],
                    "type": "paper",
                    "year": 2021,
                    "citations": 45,
                    "keywords": ["machine learning", "algorithms", "fundamentals"]
                }
            ],
            "total_results": 3
        }
        
        # Mock user feedback data
        self.mock_user_feedback = {
            "user_id": "test_user",
            "feedback_data": [
                {
                    "query": "deep learning",
                    "clicked_results": ["paper1", "paper2"],
                    "rating": 4,
                    "timestamp": "2023-01-15T10:00:00Z"
                }
            ]
        }
    
    @patch('modules.llm_processor.genai.configure')
    @patch('modules.llm_processor.genai.GenerativeModel')
    @patch('modules.ranking_engine.SKLEARN_AVAILABLE', True)
    @patch('modules.ranking_engine.EMBEDDING_AVAILABLE', True)
    @patch('modules.ranking_engine.LTR_AVAILABLE', True)
    @patch('modules.ranking_engine.KNOWLEDGE_GRAPH_AVAILABLE', True)
    def test_ml_pipeline_traditional_ranking(self, mock_model_class, mock_configure):
        """Test ML pipeline with traditional ranking algorithms"""
        
        # Set up LLM mock
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "language_analysis": {"detected_language": "en"},
            "query_processing": {"original_query": "test"},
            "intent_analysis": {"primary_intent": "find_research_papers"},
            "entity_extraction": {"technologies": []},
            "keyword_analysis": {"primary_keywords": []},
            "semantic_expansion": {"synonyms": []},
            "academic_classification": {"primary_field": "computer science"},
            "search_strategy": {"database_priorities": []},
            "multilingual_considerations": {"preserve_original_terms": []},
            "metadata": {"success": True}
        })
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            # Initialize ML pipeline components
            llm_processor = LLMProcessor()
            
            # Mock the ML components to avoid complex dependencies
            with patch('modules.ranking_engine.EmbeddingRanker') as mock_embedding_ranker, \
                 patch('modules.ranking_engine.LTRRanker') as mock_ltr_ranker, \
                 patch('modules.ranking_engine.KnowledgeGraph') as mock_knowledge_graph:
                
                # Set up mocks
                mock_embedding_ranker.return_value.calculate_realtime_similarity.return_value = [0.8, 0.6, 0.4]
                mock_ltr_ranker.return_value.rank_results.return_value = [
                    {"id": "paper1", "ltr_score": 0.9},
                    {"id": "paper2", "ltr_score": 0.7},
                    {"id": "paper3", "ltr_score": 0.5}
                ]
                mock_knowledge_graph.return_value.get_authority_score.return_value = 0.8
                
                # Initialize ranking engine with all ML components
                ranking_engine = RankingEngine(
                    llm_processor,
                    use_embedding=True,
                    use_ltr=True
                )
                
                # Test traditional ranking mode
                ranked_results = ranking_engine.rank_search_results(
                    self.mock_search_results,
                    self.mock_processed_query,
                    ranking_mode="traditional"
                )
                
                # Verify ML pipeline produces ranked results
                self.assertIsNotNone(ranked_results)
                self.assertIn('results', ranked_results)
                self.assertEqual(len(ranked_results['results']), 3)
                
                # Verify each result has ranking scores
                for result in ranked_results['results']:
                    self.assertIn('ranking_score', result)
                    self.assertIn('score_breakdown', result)
                    self.assertIsInstance(result['ranking_score'], (int, float))
                
                # Verify results are sorted by score
                scores = [r['ranking_score'] for r in ranked_results['results']]
                self.assertEqual(scores, sorted(scores, reverse=True))
    
    @patch('modules.llm_processor.genai.configure')
    @patch('modules.llm_processor.genai.GenerativeModel')
    @patch('modules.ranking_engine.SKLEARN_AVAILABLE', True)
    @patch('modules.ranking_engine.EMBEDDING_AVAILABLE', True)
    @patch('modules.ranking_engine.LTR_AVAILABLE', True)
    @patch('modules.ranking_engine.KNOWLEDGE_GRAPH_AVAILABLE', True)
    def test_ml_pipeline_hybrid_ranking(self, mock_model_class, mock_configure):
        """Test ML pipeline with hybrid ranking combining multiple algorithms"""
        
        # Set up LLM mock
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "language_analysis": {"detected_language": "en"},
            "query_processing": {"original_query": "test"},
            "intent_analysis": {"primary_intent": "find_research_papers"},
            "entity_extraction": {"technologies": []},
            "keyword_analysis": {"primary_keywords": []},
            "semantic_expansion": {"synonyms": []},
            "academic_classification": {"primary_field": "computer science"},
            "search_strategy": {"database_priorities": []},
            "multilingual_considerations": {"preserve_original_terms": []},
            "metadata": {"success": True}
        })
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            # Initialize LLM processor
            llm_processor = LLMProcessor()
            
            # Mock the ML components
            with patch('modules.ranking_engine.EmbeddingRanker') as mock_embedding_ranker, \
                 patch('modules.ranking_engine.LTRRanker') as mock_ltr_ranker, \
                 patch('modules.ranking_engine.KnowledgeGraph') as mock_knowledge_graph:
                
                # Set up embedding ranker mock
                mock_embedding_ranker.return_value.calculate_realtime_similarity.return_value = [0.9, 0.7, 0.5]
                
                # Set up LTR ranker mock
                mock_ltr_ranker.return_value.rank_results.return_value = [
                    {"id": "paper1", "ltr_score": 0.8, "final_score": 0.8},
                    {"id": "paper2", "ltr_score": 0.6, "final_score": 0.6},
                    {"id": "paper3", "ltr_score": 0.4, "final_score": 0.4}
                ]
                
                # Set up knowledge graph mock
                mock_kg = mock_knowledge_graph.return_value
                mock_kg.get_authority_score.return_value = 0.75
                mock_kg.calculate_pagerank.return_value = None
                
                # Initialize ranking engine
                ranking_engine = RankingEngine(
                    llm_processor,
                    use_embedding=True,
                    use_ltr=True
                )
                
                # Test hybrid ranking mode
                ranked_results = ranking_engine.rank_search_results(
                    self.mock_search_results,
                    self.mock_processed_query,
                    ranking_mode="hybrid"
                )
                
                # Verify hybrid ranking combines multiple signals
                self.assertIsNotNone(ranked_results)
                self.assertIn('results', ranked_results)
                self.assertEqual(len(ranked_results['results']), 3)
                
                # Verify ranking metadata includes information about algorithms used
                self.assertIn('ranking_metadata', ranked_results)
                ranking_metadata = ranked_results['ranking_metadata']
                self.assertIn('score_components', ranking_metadata)
                self.assertIn('ranking_mode', ranking_metadata)
                self.assertEqual(ranking_metadata['ranking_mode'], 'hybrid')
                
                # Verify each result has comprehensive scoring
                for result in ranked_results['results']:
                    self.assertIn('ranking_score', result)
                    self.assertIn('score_breakdown', result)
    
    @patch('modules.llm_processor.genai.configure')
    @patch('modules.llm_processor.genai.GenerativeModel')
    @patch('modules.ranking_engine.SKLEARN_AVAILABLE', True)
    @patch('modules.ranking_engine.EMBEDDING_AVAILABLE', True)
    @patch('modules.ranking_engine.LTR_AVAILABLE', True)
    @patch('modules.ranking_engine.KNOWLEDGE_GRAPH_AVAILABLE', True)
    def test_ml_pipeline_ltr_only_ranking(self, mock_model_class, mock_configure):
        """Test ML pipeline with LTR-only ranking"""
        
        # Set up LLM mock
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "language_analysis": {"detected_language": "en"},
            "query_processing": {"original_query": "test"},
            "intent_analysis": {"primary_intent": "find_research_papers"},
            "entity_extraction": {"technologies": []},
            "keyword_analysis": {"primary_keywords": []},
            "semantic_expansion": {"synonyms": []},
            "academic_classification": {"primary_field": "computer science"},
            "search_strategy": {"database_priorities": []},
            "multilingual_considerations": {"preserve_original_terms": []},
            "metadata": {"success": True}
        })
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            # Initialize LLM processor
            llm_processor = LLMProcessor()
            
            # Mock the ML components
            with patch('modules.ranking_engine.EmbeddingRanker') as mock_embedding_ranker, \
                 patch('modules.ranking_engine.LTRRanker') as mock_ltr_ranker, \
                 patch('modules.ranking_engine.KnowledgeGraph') as mock_knowledge_graph:
                
                # Set up LTR ranker mock to return ranked results
                mock_ltr_ranker.return_value.rank_results.return_value = [
                    {
                        "id": "paper1",
                        "title": "Deep Learning for Computer Vision",
                        "ranking_score": 0.95,
                        "ltr_score": 0.95,
                        "score_breakdown": {"ltr_score": 0.95}
                    },
                    {
                        "id": "paper2", 
                        "title": "Neural Network Architectures",
                        "ranking_score": 0.75,
                        "ltr_score": 0.75,
                        "score_breakdown": {"ltr_score": 0.75}
                    },
                    {
                        "id": "paper3",
                        "title": "Machine Learning Fundamentals", 
                        "ranking_score": 0.55,
                        "ltr_score": 0.55,
                        "score_breakdown": {"ltr_score": 0.55}
                    }
                ]
                
                # Initialize ranking engine
                ranking_engine = RankingEngine(
                    llm_processor,
                    use_embedding=True,
                    use_ltr=True
                )
                
                # Test LTR-only ranking mode
                ranked_results = ranking_engine.rank_search_results(
                    self.mock_search_results,
                    self.mock_processed_query,
                    user_feedback_data=self.mock_user_feedback,
                    ranking_mode="ltr_only"
                )
                
                # Verify LTR ranking works
                self.assertIsNotNone(ranked_results)
                self.assertIn('results', ranked_results)
                self.assertEqual(len(ranked_results['results']), 3)
                
                # Verify LTR-specific metadata
                self.assertIn('ranking_metadata', ranked_results)
                ranking_metadata = ranked_results['ranking_metadata']
                self.assertEqual(ranking_metadata['ranking_mode'], 'ltr_only')
                
                # Verify results are ranked by LTR scores
                scores = [r['ranking_score'] for r in ranked_results['results']]
                self.assertEqual(scores, sorted(scores, reverse=True))
                
                # Verify LTR ranker was called with correct parameters
                mock_ltr_ranker.return_value.rank_results.assert_called_once()
    
    @patch('modules.llm_processor.genai.configure')
    @patch('modules.llm_processor.genai.GenerativeModel')
    @patch('modules.ranking_engine.SKLEARN_AVAILABLE', True)
    @patch('modules.ranking_engine.EMBEDDING_AVAILABLE', True)
    @patch('modules.ranking_engine.LTR_AVAILABLE', True)
    @patch('modules.ranking_engine.KNOWLEDGE_GRAPH_AVAILABLE', True)
    def test_ml_pipeline_knowledge_graph_integration(self, mock_model_class, mock_configure):
        """Test ML pipeline with knowledge graph integration"""
        
        # Set up LLM mock
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "language_analysis": {"detected_language": "en"},
            "query_processing": {"original_query": "test"},
            "intent_analysis": {"primary_intent": "find_research_papers"},
            "entity_extraction": {"technologies": []},
            "keyword_analysis": {"primary_keywords": []},
            "semantic_expansion": {"synonyms": []},
            "academic_classification": {"primary_field": "computer science"},
            "search_strategy": {"database_priorities": []},
            "multilingual_considerations": {"preserve_original_terms": []},
            "metadata": {"success": True}
        })
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            # Initialize LLM processor
            llm_processor = LLMProcessor()
            
            # Mock the ML components
            with patch('modules.ranking_engine.EmbeddingRanker') as mock_embedding_ranker, \
                 patch('modules.ranking_engine.LTRRanker') as mock_ltr_ranker, \
                 patch('modules.ranking_engine.KnowledgeGraph') as mock_knowledge_graph:
                
                # Set up knowledge graph mock
                mock_kg = mock_knowledge_graph.return_value
                mock_kg.calculate_pagerank.return_value = None
                mock_kg.get_authority_score.return_value = 0.85
                mock_kg.add_paper.return_value = None
                mock_kg.add_expert.return_value = None
                mock_kg.add_equipment.return_value = None
                
                # Set up other mocks
                mock_embedding_ranker.return_value.calculate_realtime_similarity.return_value = [0.7, 0.6, 0.5]
                mock_ltr_ranker.return_value.rank_results.return_value = [
                    {"id": "paper1", "ltr_score": 0.8},
                    {"id": "paper2", "ltr_score": 0.6},
                    {"id": "paper3", "ltr_score": 0.4}
                ]
                
                # Initialize ranking engine
                ranking_engine = RankingEngine(
                    llm_processor,
                    use_embedding=True,
                    use_ltr=True
                )
                
                # Test ranking with knowledge graph
                ranked_results = ranking_engine.rank_search_results(
                    self.mock_search_results,
                    self.mock_processed_query,
                    ranking_mode="hybrid"
                )
                
                # Verify knowledge graph integration
                self.assertIsNotNone(ranked_results)
                self.assertIn('results', ranked_results)
                
                # Verify knowledge graph methods were called
                mock_kg.add_paper.assert_called()
                mock_kg.calculate_pagerank.assert_called_once()
                
                # Verify ranking includes graph-based features
                for result in ranked_results['results']:
                    self.assertIn('ranking_score', result)
                    self.assertIn('score_breakdown', result)
    
    @patch('modules.llm_processor.genai.configure')
    @patch('modules.llm_processor.genai.GenerativeModel')
    @patch('modules.ranking_engine.SKLEARN_AVAILABLE', True)
    @patch('modules.ranking_engine.EMBEDDING_AVAILABLE', True)
    @patch('modules.ranking_engine.LTR_AVAILABLE', True)
    @patch('modules.ranking_engine.KNOWLEDGE_GRAPH_AVAILABLE', True)
    def test_ml_pipeline_error_handling(self, mock_model_class, mock_configure):
        """Test ML pipeline error handling when components fail"""
        
        # Set up LLM mock
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "language_analysis": {"detected_language": "en"},
            "query_processing": {"original_query": "test"},
            "intent_analysis": {"primary_intent": "find_research_papers"},
            "entity_extraction": {"technologies": []},
            "keyword_analysis": {"primary_keywords": []},
            "semantic_expansion": {"synonyms": []},
            "academic_classification": {"primary_field": "computer science"},
            "search_strategy": {"database_priorities": []},
            "multilingual_considerations": {"preserve_original_terms": []},
            "metadata": {"success": True}
        })
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            # Initialize LLM processor
            llm_processor = LLMProcessor()
            
            # Mock the ML components with failures
            with patch('modules.ranking_engine.EmbeddingRanker') as mock_embedding_ranker, \
                 patch('modules.ranking_engine.LTRRanker') as mock_ltr_ranker, \
                 patch('modules.ranking_engine.KnowledgeGraph') as mock_knowledge_graph:
                
                # Set up embedding ranker to fail
                mock_embedding_ranker.return_value.calculate_realtime_similarity.side_effect = Exception("Embedding error")
                
                # Set up LTR ranker to fail
                mock_ltr_ranker.return_value.rank_results.side_effect = Exception("LTR error")
                
                # Set up knowledge graph to fail
                mock_knowledge_graph.side_effect = Exception("Knowledge graph error")
                
                # Initialize ranking engine
                ranking_engine = RankingEngine(
                    llm_processor,
                    use_embedding=True,
                    use_ltr=True
                )
                
                # Test that ranking still works with component failures
                ranked_results = ranking_engine.rank_search_results(
                    self.mock_search_results,
                    self.mock_processed_query,
                    ranking_mode="hybrid"
                )
                
                # Should still return results (original search results when ranking fails)
                self.assertIsNotNone(ranked_results)
                self.assertIn('results', ranked_results)
                self.assertEqual(len(ranked_results['results']), 3)
                
                # When ranking fails completely, the system returns original search results
                # Verify the results structure is preserved
                for result in ranked_results['results']:
                    self.assertIn('id', result)
                    self.assertIn('title', result)
                    # Note: ranking_score may not be present when ranking fails completely
    
    @patch('modules.llm_processor.genai.configure')
    @patch('modules.llm_processor.genai.GenerativeModel')
    @patch('modules.ranking_engine.SKLEARN_AVAILABLE', True)
    @patch('modules.ranking_engine.EMBEDDING_AVAILABLE', True)
    @patch('modules.ranking_engine.LTR_AVAILABLE', True)
    @patch('modules.ranking_engine.KNOWLEDGE_GRAPH_AVAILABLE', True)
    def test_ml_pipeline_personalization_integration(self, mock_model_class, mock_configure):
        """Test ML pipeline with personalization features"""
        
        # Set up LLM mock
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "language_analysis": {"detected_language": "en"},
            "query_processing": {"original_query": "test"},
            "intent_analysis": {"primary_intent": "find_research_papers"},
            "entity_extraction": {"technologies": []},
            "keyword_analysis": {"primary_keywords": []},
            "semantic_expansion": {"synonyms": []},
            "academic_classification": {"primary_field": "computer science"},
            "search_strategy": {"database_priorities": []},
            "multilingual_considerations": {"preserve_original_terms": []},
            "metadata": {"success": True}
        })
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        # Mock personalization data
        personalization_data = {
            "user_id": "test_user",
            "user_profile": {
                "research_interests": ["deep learning", "computer vision"],
                "academic_level": "graduate",
                "preferred_authors": ["John Smith"],
                "preferred_fields": ["computer science", "AI"]
            },
            "interaction_history": [
                {
                    "query": "deep learning",
                    "clicked_results": ["paper1"],
                    "rating": 5,
                    "timestamp": "2023-01-15T10:00:00Z"
                }
            ]
        }
        
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            # Initialize LLM processor
            llm_processor = LLMProcessor()
            
            # Mock the ML components
            with patch('modules.ranking_engine.EmbeddingRanker') as mock_embedding_ranker, \
                 patch('modules.ranking_engine.LTRRanker') as mock_ltr_ranker, \
                 patch('modules.ranking_engine.KnowledgeGraph') as mock_knowledge_graph:
                
                # Set up mocks
                mock_embedding_ranker.return_value.calculate_realtime_similarity.return_value = [0.8, 0.6, 0.4]
                mock_ltr_ranker.return_value.rank_results.return_value = [
                    {"id": "paper1", "ltr_score": 0.9},
                    {"id": "paper2", "ltr_score": 0.7},
                    {"id": "paper3", "ltr_score": 0.5}
                ]
                mock_knowledge_graph.return_value.get_authority_score.return_value = 0.8
                
                # Initialize ranking engine
                ranking_engine = RankingEngine(
                    llm_processor,
                    use_embedding=True,
                    use_ltr=True
                )
                
                # Test ranking with personalization
                ranked_results = ranking_engine.rank_search_results(
                    self.mock_search_results,
                    self.mock_processed_query,
                    user_feedback_data=self.mock_user_feedback,
                    ranking_mode="hybrid",
                    user_personalization_data=personalization_data
                )
                
                # Verify personalization is applied
                self.assertIsNotNone(ranked_results)
                self.assertIn('results', ranked_results)
                
                # Verify personalization metadata
                self.assertIn('ranking_metadata', ranked_results)
                ranking_metadata = ranked_results['ranking_metadata']
                self.assertIn('personalization_enabled', ranking_metadata)
                self.assertTrue(ranking_metadata['personalization_enabled'])
                
                # Verify results include personalization scores
                for result in ranked_results['results']:
                    self.assertIn('ranking_score', result)
                    self.assertIn('score_breakdown', result)
                    self.assertIn('personalization_score', result['score_breakdown'])

if __name__ == '__main__':
    unittest.main() 