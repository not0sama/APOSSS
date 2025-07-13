#!/usr/bin/env python3
"""
Unit tests for APOSSS EmbeddingRanker class
Tests embeddings, FAISS indexing, and semantic similarity search
"""
import sys
import os
import unittest
import logging
import numpy as np
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestEmbeddingRanker(unittest.TestCase):
    """Unit tests for EmbeddingRanker class"""

    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directory for test cache
        self.temp_dir = tempfile.mkdtemp()
        
        # Sample test data
        self.test_query = "machine learning for medical diagnosis"
        self.test_documents = [
            {
                'id': 'doc1',
                'title': 'Machine Learning in Healthcare',
                'description': 'Comprehensive overview of ML applications in medical diagnosis and treatment',
                'content': 'Machine learning has revolutionized healthcare by enabling automated diagnosis and personalized treatment plans.'
            },
            {
                'id': 'doc2',
                'title': 'Deep Learning for Medical Imaging',
                'description': 'Advanced deep learning techniques for analyzing medical images and scans',
                'content': 'Deep learning models can accurately detect anomalies in medical images with high precision.'
            },
            {
                'id': 'doc3',
                'title': 'Artificial Intelligence in Drug Discovery',
                'description': 'AI applications in pharmaceutical research and drug development',
                'content': 'AI accelerates drug discovery by predicting molecular interactions and identifying potential compounds.'
            }
        ]
        
        self.processed_query = {
            'keywords': {
                'primary': ['machine learning', 'medical', 'diagnosis'],
                'secondary': ['healthcare', 'AI', 'algorithm']
            },
            'entities': {
                'technologies': ['machine learning', 'artificial intelligence'],
                'concepts': ['medical diagnosis', 'healthcare']
            },
            'intent': {
                'primary_intent': 'find_research',
                'confidence': 0.85
            }
        }

    def tearDown(self):
        """Clean up test fixtures"""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('modules.embedding_ranker.SentenceTransformer')
    @patch('modules.embedding_ranker.faiss')
    def test_embedding_ranker_initialization(self, mock_faiss, mock_sentence_transformer):
        """Test EmbeddingRanker initialization"""
        from modules.embedding_ranker import EmbeddingRanker
        
        # Mock sentence transformer
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        # Test successful initialization
        ranker = EmbeddingRanker(cache_dir=self.temp_dir)
        
        self.assertIsNotNone(ranker)
        self.assertEqual(ranker.cache_dir, self.temp_dir)
        self.assertEqual(ranker.model_name, 'paraphrase-multilingual-MiniLM-L12-v2')
        self.assertEqual(ranker.embedding_dimension, 384)

    @patch('modules.embedding_ranker.SentenceTransformer')
    @patch('modules.embedding_ranker.faiss')
    def test_model_initialization(self, mock_faiss, mock_sentence_transformer):
        """Test sentence transformer model initialization"""
        from modules.embedding_ranker import EmbeddingRanker
        
        # Mock sentence transformer
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        ranker = EmbeddingRanker(cache_dir=self.temp_dir)
        
        # Test model initialization
        mock_sentence_transformer.assert_called_with('paraphrase-multilingual-MiniLM-L12-v2')
        self.assertEqual(ranker.model, mock_model)

    @patch('modules.embedding_ranker.SentenceTransformer')
    @patch('modules.embedding_ranker.faiss')
    def test_text_embedding_generation(self, mock_faiss, mock_sentence_transformer):
        """Test text embedding generation"""
        from modules.embedding_ranker import EmbeddingRanker
        
        # Mock sentence transformer
        mock_model = Mock()
        mock_embedding = np.random.rand(384).astype(np.float32)
        mock_model.encode.return_value = mock_embedding
        mock_sentence_transformer.return_value = mock_model
        
        ranker = EmbeddingRanker(cache_dir=self.temp_dir)
        
        # Test embedding generation
        embedding = ranker.get_embedding(self.test_query)
        
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape, (384,))
        self.assertEqual(embedding.dtype, np.float32)
        mock_model.encode.assert_called_once()

    @patch('modules.embedding_ranker.SentenceTransformer')
    @patch('modules.embedding_ranker.faiss')
    def test_batch_embedding_generation(self, mock_faiss, mock_sentence_transformer):
        """Test batch embedding generation"""
        from modules.embedding_ranker import EmbeddingRanker
        
        # Mock sentence transformer
        mock_model = Mock()
        mock_embeddings = np.random.rand(3, 384).astype(np.float32)
        mock_model.encode.return_value = mock_embeddings
        mock_sentence_transformer.return_value = mock_model
        
        ranker = EmbeddingRanker(cache_dir=self.temp_dir)
        
        # Test batch embedding generation
        texts = [doc['content'] for doc in self.test_documents]
        embeddings = ranker.get_embeddings(texts)
        
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape, (3, 384))
        self.assertEqual(embeddings.dtype, np.float32)
        mock_model.encode.assert_called_once()

    @patch('modules.embedding_ranker.SentenceTransformer')
    @patch('modules.embedding_ranker.faiss')
    def test_faiss_index_creation(self, mock_faiss, mock_sentence_transformer):
        """Test FAISS index creation"""
        from modules.embedding_ranker import EmbeddingRanker
        
        # Mock FAISS
        mock_index = Mock()
        mock_faiss.IndexFlatIP.return_value = mock_index
        
        # Mock sentence transformer
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        ranker = EmbeddingRanker(cache_dir=self.temp_dir)
        
        # Test index creation
        ranker._create_faiss_index()
        
        mock_faiss.IndexFlatIP.assert_called_with(384)
        self.assertEqual(ranker.faiss_index, mock_index)

    @patch('modules.embedding_ranker.SentenceTransformer')
    @patch('modules.embedding_ranker.faiss')
    def test_document_indexing(self, mock_faiss, mock_sentence_transformer):
        """Test document indexing"""
        from modules.embedding_ranker import EmbeddingRanker
        
        # Mock FAISS
        mock_index = Mock()
        mock_faiss.IndexFlatIP.return_value = mock_index
        
        # Mock sentence transformer
        mock_model = Mock()
        mock_embeddings = np.random.rand(3, 384).astype(np.float32)
        mock_model.encode.return_value = mock_embeddings
        mock_sentence_transformer.return_value = mock_model
        
        ranker = EmbeddingRanker(cache_dir=self.temp_dir)
        
        # Test document indexing
        ranker.index_documents(self.test_documents)
        
        # Verify that documents were added to index
        mock_index.add.assert_called_once()
        self.assertEqual(len(ranker.document_cache), 3)
        
        # Check document cache
        for doc in self.test_documents:
            self.assertIn(doc['id'], ranker.document_cache)

    @patch('modules.embedding_ranker.SentenceTransformer')
    @patch('modules.embedding_ranker.faiss')
    def test_similarity_search(self, mock_faiss, mock_sentence_transformer):
        """Test similarity search"""
        from modules.embedding_ranker import EmbeddingRanker
        
        # Mock FAISS
        mock_index = Mock()
        mock_scores = np.array([[0.9, 0.8, 0.7]], dtype=np.float32)
        mock_indices = np.array([[0, 1, 2]], dtype=np.int64)
        mock_index.search.return_value = (mock_scores, mock_indices)
        mock_faiss.IndexFlatIP.return_value = mock_index
        
        # Mock sentence transformer
        mock_model = Mock()
        mock_query_embedding = np.random.rand(384).astype(np.float32)
        mock_doc_embeddings = np.random.rand(3, 384).astype(np.float32)
        mock_model.encode.side_effect = [mock_doc_embeddings, mock_query_embedding]
        mock_sentence_transformer.return_value = mock_model
        
        ranker = EmbeddingRanker(cache_dir=self.temp_dir)
        
        # Index documents first
        ranker.index_documents(self.test_documents)
        
        # Test similarity search
        results = ranker.search_similar_documents(self.test_query, k=3)
        
        self.assertIsInstance(results, dict)
        self.assertIn('results', results)
        self.assertIn('total_results', results)
        self.assertIn('search_metadata', results)
        
        # Check results
        self.assertEqual(len(results['results']), 3)
        mock_index.search.assert_called_once()

    @patch('modules.embedding_ranker.SentenceTransformer')
    @patch('modules.embedding_ranker.faiss')
    def test_similarity_search_with_processed_query(self, mock_faiss, mock_sentence_transformer):
        """Test similarity search with processed query"""
        from modules.embedding_ranker import EmbeddingRanker
        
        # Mock FAISS
        mock_index = Mock()
        mock_scores = np.array([[0.9, 0.8]], dtype=np.float32)
        mock_indices = np.array([[0, 1]], dtype=np.int64)
        mock_index.search.return_value = (mock_scores, mock_indices)
        mock_faiss.IndexFlatIP.return_value = mock_index
        
        # Mock sentence transformer
        mock_model = Mock()
        mock_query_embedding = np.random.rand(384).astype(np.float32)
        mock_doc_embeddings = np.random.rand(3, 384).astype(np.float32)
        mock_model.encode.side_effect = [mock_doc_embeddings, mock_query_embedding]
        mock_sentence_transformer.return_value = mock_model
        
        ranker = EmbeddingRanker(cache_dir=self.temp_dir)
        
        # Index documents first
        ranker.index_documents(self.test_documents)
        
        # Test similarity search with processed query
        results = ranker.search_similar_documents(
            self.test_query, k=2, processed_query=self.processed_query
        )
        
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results['results']), 2)
        
        # Check that processed query metadata is included
        self.assertIn('processed_query', results['search_metadata'])

    @patch('modules.embedding_ranker.SentenceTransformer')
    @patch('modules.embedding_ranker.faiss')
    def test_cosine_similarity_calculation(self, mock_faiss, mock_sentence_transformer):
        """Test cosine similarity calculation"""
        from modules.embedding_ranker import EmbeddingRanker
        
        # Mock sentence transformer
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        ranker = EmbeddingRanker(cache_dir=self.temp_dir)
        
        # Test cosine similarity calculation
        embedding1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        embedding2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        similarity = ranker.calculate_similarity(embedding1, embedding2)
        
        self.assertIsInstance(similarity, (int, float))
        self.assertGreaterEqual(similarity, -1.0)
        self.assertLessEqual(similarity, 1.0)
        
        # Test with identical embeddings
        identity_similarity = ranker.calculate_similarity(embedding1, embedding1)
        self.assertAlmostEqual(identity_similarity, 1.0, places=5)

    @patch('modules.embedding_ranker.SentenceTransformer')
    @patch('modules.embedding_ranker.faiss')
    def test_cache_management(self, mock_faiss, mock_sentence_transformer):
        """Test cache management"""
        from modules.embedding_ranker import EmbeddingRanker
        
        # Mock sentence transformer
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        ranker = EmbeddingRanker(cache_dir=self.temp_dir)
        
        # Test cache statistics
        stats = ranker.get_cache_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_vectors', stats)
        self.assertIn('total_documents', stats)
        self.assertIn('cache_size_mb', stats)
        self.assertIn('last_updated', stats)

    @patch('modules.embedding_ranker.SentenceTransformer')
    @patch('modules.embedding_ranker.faiss')
    def test_embedding_caching(self, mock_faiss, mock_sentence_transformer):
        """Test embedding caching"""
        from modules.embedding_ranker import EmbeddingRanker
        
        # Mock sentence transformer
        mock_model = Mock()
        mock_embedding = np.random.rand(384).astype(np.float32)
        mock_model.encode.return_value = mock_embedding
        mock_sentence_transformer.return_value = mock_model
        
        ranker = EmbeddingRanker(cache_dir=self.temp_dir)
        
        # Test embedding caching
        text = "test text"
        
        # First call should generate embedding
        embedding1 = ranker.get_embedding(text)
        
        # Second call should use cached embedding
        embedding2 = ranker.get_embedding(text)
        
        np.testing.assert_array_equal(embedding1, embedding2)
        # Model should only be called once due to caching
        self.assertEqual(mock_model.encode.call_count, 1)

    @patch('modules.embedding_ranker.SentenceTransformer')
    @patch('modules.embedding_ranker.faiss')
    def test_empty_query_handling(self, mock_faiss, mock_sentence_transformer):
        """Test handling of empty queries"""
        from modules.embedding_ranker import EmbeddingRanker
        
        # Mock sentence transformer
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        ranker = EmbeddingRanker(cache_dir=self.temp_dir)
        
        # Test with empty query
        results = ranker.search_similar_documents("", k=5)
        
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results['results']), 0)

    @patch('modules.embedding_ranker.SentenceTransformer')
    @patch('modules.embedding_ranker.faiss')
    def test_no_documents_handling(self, mock_faiss, mock_sentence_transformer):
        """Test handling when no documents are indexed"""
        from modules.embedding_ranker import EmbeddingRanker
        
        # Mock sentence transformer
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        ranker = EmbeddingRanker(cache_dir=self.temp_dir)
        
        # Test search with no indexed documents
        results = ranker.search_similar_documents(self.test_query, k=5)
        
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results['results']), 0)

    @patch('modules.embedding_ranker.SentenceTransformer')
    @patch('modules.embedding_ranker.faiss')
    def test_large_batch_processing(self, mock_faiss, mock_sentence_transformer):
        """Test processing large batches of documents"""
        from modules.embedding_ranker import EmbeddingRanker
        
        # Mock sentence transformer
        mock_model = Mock()
        mock_embeddings = np.random.rand(100, 384).astype(np.float32)
        mock_model.encode.return_value = mock_embeddings
        mock_sentence_transformer.return_value = mock_model
        
        # Mock FAISS
        mock_index = Mock()
        mock_faiss.IndexFlatIP.return_value = mock_index
        
        ranker = EmbeddingRanker(cache_dir=self.temp_dir)
        
        # Create large batch of documents
        large_documents = [
            {
                'id': f'doc{i}',
                'title': f'Document {i}',
                'content': f'Content of document {i}'
            }
            for i in range(100)
        ]
        
        # Test large batch processing
        ranker.index_documents(large_documents)
        
        # Verify processing completed
        self.assertEqual(len(ranker.document_cache), 100)
        mock_index.add.assert_called_once()

    @patch('modules.embedding_ranker.SentenceTransformer')
    @patch('modules.embedding_ranker.faiss')
    def test_multilingual_support(self, mock_faiss, mock_sentence_transformer):
        """Test multilingual text processing"""
        from modules.embedding_ranker import EmbeddingRanker
        
        # Mock sentence transformer
        mock_model = Mock()
        mock_embedding = np.random.rand(384).astype(np.float32)
        mock_model.encode.return_value = mock_embedding
        mock_sentence_transformer.return_value = mock_model
        
        ranker = EmbeddingRanker(cache_dir=self.temp_dir)
        
        # Test multilingual queries
        multilingual_queries = [
            "machine learning",  # English
            "aprendizaje automático",  # Spanish
            "apprentissage automatique",  # French
            "机器学习",  # Chinese
            "機械学習"  # Japanese
        ]
        
        for query in multilingual_queries:
            embedding = ranker.get_embedding(query)
            self.assertIsInstance(embedding, np.ndarray)
            self.assertEqual(embedding.shape, (384,))

    @patch('modules.embedding_ranker.SentenceTransformer')
    @patch('modules.embedding_ranker.faiss')
    def test_error_handling(self, mock_faiss, mock_sentence_transformer):
        """Test error handling in embedding operations"""
        from modules.embedding_ranker import EmbeddingRanker
        
        # Mock sentence transformer to raise exception
        mock_model = Mock()
        mock_model.encode.side_effect = Exception("Model error")
        mock_sentence_transformer.return_value = mock_model
        
        ranker = EmbeddingRanker(cache_dir=self.temp_dir)
        
        # Test error handling
        with self.assertRaises(Exception):
            ranker.get_embedding("test query")

    @patch('modules.embedding_ranker.SentenceTransformer')
    @patch('modules.embedding_ranker.faiss')
    def test_index_persistence(self, mock_faiss, mock_sentence_transformer):
        """Test index persistence and loading"""
        from modules.embedding_ranker import EmbeddingRanker
        
        # Mock sentence transformer
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        ranker = EmbeddingRanker(cache_dir=self.temp_dir)
        
        # Test index saving
        if hasattr(ranker, 'save_index'):
            ranker.save_index()
        
        # Test index loading
        if hasattr(ranker, 'load_index'):
            ranker.load_index()

    @patch('modules.embedding_ranker.SentenceTransformer')
    @patch('modules.embedding_ranker.faiss')
    def test_memory_efficiency(self, mock_faiss, mock_sentence_transformer):
        """Test memory efficiency with large embeddings"""
        from modules.embedding_ranker import EmbeddingRanker
        
        # Mock sentence transformer
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        ranker = EmbeddingRanker(cache_dir=self.temp_dir)
        
        # Test memory usage optimization
        if hasattr(ranker, 'clear_cache'):
            ranker.clear_cache()
        
        if hasattr(ranker, 'optimize_memory'):
            ranker.optimize_memory()


class TestEmbeddingRankerIntegration(unittest.TestCase):
    """Integration tests for EmbeddingRanker with real components"""

    def setUp(self):
        """Set up integration test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up integration test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_embedding_ranker_with_real_sentence_transformers(self):
        """Test EmbeddingRanker with actual sentence transformers (if available)"""
        try:
            from sentence_transformers import SentenceTransformer
            from modules.embedding_ranker import EmbeddingRanker
            
            # This test only runs if sentence transformers is available
            ranker = EmbeddingRanker(cache_dir=self.temp_dir)
            
            # Test that sentence transformer is properly initialized
            if hasattr(ranker, 'model'):
                self.assertIsNotNone(ranker.model)
            
        except ImportError:
            self.skipTest("sentence-transformers not available")

    def test_embedding_ranker_with_real_faiss(self):
        """Test EmbeddingRanker with actual FAISS (if available)"""
        try:
            import faiss
            from modules.embedding_ranker import EmbeddingRanker
            
            # This test only runs if FAISS is available
            ranker = EmbeddingRanker(cache_dir=self.temp_dir)
            
            # Test that FAISS functionality is available
            self.assertIsNotNone(faiss)
            
        except ImportError:
            self.skipTest("FAISS not available")

    def test_embedding_ranker_performance(self):
        """Test EmbeddingRanker performance characteristics"""
        from modules.embedding_ranker import EmbeddingRanker
        
        # Test with minimal setup
        ranker = EmbeddingRanker(cache_dir=self.temp_dir)
        
        # Test performance metrics if available
        if hasattr(ranker, 'get_performance_metrics'):
            metrics = ranker.get_performance_metrics()
            self.assertIsInstance(metrics, dict)


if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2) 