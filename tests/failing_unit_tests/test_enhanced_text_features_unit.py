#!/usr/bin/env python3
"""
Unit tests for APOSSS EnhancedTextFeatures class
Tests BM25 scoring, n-gram features, proximity features, and text complexity analysis
"""
import sys
import os
import unittest
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestEnhancedTextFeatures(unittest.TestCase):
    """Unit tests for EnhancedTextFeatures class"""

    def setUp(self):
        """Set up test fixtures"""
        # Sample test data
        self.test_query = "machine learning for medical diagnosis"
        self.test_result = {
            'id': 'result1',
            'title': 'Machine Learning in Healthcare',
            'description': 'Comprehensive overview of ML applications in medical diagnosis and treatment. This paper explores various machine learning techniques used in healthcare systems.',
            'author': 'Dr. Sarah Johnson',
            'type': 'article',
            'metadata': {
                'year': 2023,
                'category': 'Healthcare Technology',
                'citations': 15,
                'status': 'published'
            }
        }
        
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

    @patch('modules.enhanced_text_features.nltk')
    def test_enhanced_text_features_initialization(self, mock_nltk):
        """Test EnhancedTextFeatures initialization"""
        from modules.enhanced_text_features import EnhancedTextFeatures
        
        # Mock NLTK data check
        mock_nltk.data.find.return_value = True
        
        # Test successful initialization
        features = EnhancedTextFeatures()
        
        self.assertIsNotNone(features)
        self.assertEqual(features.corpus, [])
        self.assertEqual(features.corpus_tokens, [])
        self.assertIsNone(features.bm25)

    @patch('modules.enhanced_text_features.nltk')
    def test_extract_all_features(self, mock_nltk):
        """Test extracting all enhanced text features"""
        from modules.enhanced_text_features import EnhancedTextFeatures
        
        # Mock NLTK components
        mock_nltk.data.find.return_value = True
        mock_nltk.word_tokenize.return_value = ['machine', 'learning', 'for', 'medical', 'diagnosis']
        
        features = EnhancedTextFeatures()
        
        # Test feature extraction
        extracted_features = features.extract_all_features(
            self.test_query, self.test_result, self.processed_query
        )
        
        self.assertIsInstance(extracted_features, dict)
        self.assertGreater(len(extracted_features), 0)
        
        # Check for expected feature categories
        feature_keys = extracted_features.keys()
        
        # Should contain BM25 features
        bm25_features = [key for key in feature_keys if 'bm25' in key.lower()]
        self.assertGreater(len(bm25_features), 0)
        
        # Should contain n-gram features
        ngram_features = [key for key in feature_keys if 'gram' in key.lower()]
        self.assertGreater(len(ngram_features), 0)

    @patch('modules.enhanced_text_features.nltk')
    @patch('modules.enhanced_text_features.BM25Okapi')
    def test_bm25_scores_extraction(self, mock_bm25, mock_nltk):
        """Test BM25 scores extraction"""
        from modules.enhanced_text_features import EnhancedTextFeatures
        
        # Mock NLTK components
        mock_nltk.data.find.return_value = True
        mock_nltk.word_tokenize.return_value = ['machine', 'learning', 'for', 'medical', 'diagnosis']
        
        # Mock BM25
        mock_bm25_instance = Mock()
        mock_bm25_instance.get_scores.return_value = [0.5, 0.8, 0.3]
        mock_bm25.return_value = mock_bm25_instance
        
        features = EnhancedTextFeatures()
        
        # Test BM25 extraction
        bm25_features = features._extract_bm25_scores(
            self.test_query, 
            self.test_result['title'], 
            self.test_result['description']
        )
        
        self.assertIsInstance(bm25_features, dict)
        self.assertGreater(len(bm25_features), 0)
        
        # Check for expected BM25 features
        expected_bm25_features = ['bm25_title', 'bm25_description', 'bm25_combined']
        for feature in expected_bm25_features:
            self.assertIn(feature, bm25_features)

    @patch('modules.enhanced_text_features.nltk')
    def test_ngram_features_extraction(self, mock_nltk):
        """Test n-gram features extraction"""
        from modules.enhanced_text_features import EnhancedTextFeatures
        
        # Mock NLTK components
        mock_nltk.data.find.return_value = True
        mock_nltk.word_tokenize.return_value = ['machine', 'learning', 'for', 'medical', 'diagnosis']
        
        features = EnhancedTextFeatures()
        
        # Test n-gram extraction
        ngram_features = features._extract_ngram_features(
            self.test_query, 
            self.test_result['title'] + ' ' + self.test_result['description']
        )
        
        self.assertIsInstance(ngram_features, dict)
        self.assertGreater(len(ngram_features), 0)
        
        # Check for expected n-gram features
        expected_ngram_features = ['unigram_overlap', 'bigram_overlap', 'trigram_overlap']
        for feature in expected_ngram_features:
            self.assertIn(feature, ngram_features)
        
        # Check that overlap values are between 0 and 1
        for feature in expected_ngram_features:
            self.assertGreaterEqual(ngram_features[feature], 0)
            self.assertLessEqual(ngram_features[feature], 1)

    @patch('modules.enhanced_text_features.nltk')
    def test_proximity_features_extraction(self, mock_nltk):
        """Test proximity features extraction"""
        from modules.enhanced_text_features import EnhancedTextFeatures
        
        # Mock NLTK components
        mock_nltk.data.find.return_value = True
        mock_nltk.word_tokenize.return_value = ['machine', 'learning', 'for', 'medical', 'diagnosis']
        
        features = EnhancedTextFeatures()
        
        # Test proximity extraction
        proximity_features = features._extract_proximity_features(
            self.test_query, 
            self.test_result['title'] + ' ' + self.test_result['description']
        )
        
        self.assertIsInstance(proximity_features, dict)
        self.assertGreater(len(proximity_features), 0)
        
        # Check for expected proximity features
        expected_proximity_features = ['query_term_proximity', 'term_distance_avg', 'term_distance_min']
        for feature in expected_proximity_features:
            self.assertIn(feature, proximity_features)

    @patch('modules.enhanced_text_features.nltk')
    @patch('modules.enhanced_text_features.textstat')
    def test_complexity_features_extraction(self, mock_textstat, mock_nltk):
        """Test complexity features extraction"""
        from modules.enhanced_text_features import EnhancedTextFeatures
        
        # Mock NLTK components
        mock_nltk.data.find.return_value = True
        mock_nltk.word_tokenize.return_value = ['machine', 'learning', 'for', 'medical', 'diagnosis']
        
        # Mock textstat
        mock_textstat.flesch_reading_ease.return_value = 50.0
        mock_textstat.flesch_kincaid_grade.return_value = 12.0
        mock_textstat.gunning_fog.return_value = 15.0
        mock_textstat.automated_readability_index.return_value = 10.0
        
        features = EnhancedTextFeatures()
        
        # Test complexity extraction
        complexity_features = features._extract_complexity_features(
            self.test_result['title'] + ' ' + self.test_result['description'], 
            self.processed_query
        )
        
        self.assertIsInstance(complexity_features, dict)
        self.assertGreater(len(complexity_features), 0)
        
        # Check for expected complexity features
        expected_complexity_features = [
            'flesch_reading_ease', 'flesch_kincaid_grade', 
            'gunning_fog', 'automated_readability_index'
        ]
        for feature in expected_complexity_features:
            self.assertIn(feature, complexity_features)

    @patch('modules.enhanced_text_features.nltk')
    def test_term_frequency_calculation(self, mock_nltk):
        """Test term frequency calculation"""
        from modules.enhanced_text_features import EnhancedTextFeatures
        
        # Mock NLTK components
        mock_nltk.data.find.return_value = True
        mock_nltk.word_tokenize.return_value = ['machine', 'learning', 'machine', 'learning', 'diagnosis']
        
        features = EnhancedTextFeatures()
        
        # Test term frequency calculation
        if hasattr(features, '_calculate_term_frequency'):
            term_freq = features._calculate_term_frequency(
                ['machine', 'learning', 'machine', 'learning', 'diagnosis']
            )
            
            self.assertIsInstance(term_freq, dict)
            self.assertEqual(term_freq['machine'], 2)
            self.assertEqual(term_freq['learning'], 2)
            self.assertEqual(term_freq['diagnosis'], 1)

    @patch('modules.enhanced_text_features.nltk')
    def test_document_frequency_calculation(self, mock_nltk):
        """Test document frequency calculation"""
        from modules.enhanced_text_features import EnhancedTextFeatures
        
        # Mock NLTK components
        mock_nltk.data.find.return_value = True
        mock_nltk.word_tokenize.return_value = ['machine', 'learning', 'for', 'medical', 'diagnosis']
        
        features = EnhancedTextFeatures()
        
        # Test document frequency calculation
        if hasattr(features, '_calculate_document_frequency'):
            corpus = [
                ['machine', 'learning', 'healthcare'],
                ['medical', 'diagnosis', 'treatment'],
                ['machine', 'learning', 'diagnosis']
            ]
            
            doc_freq = features._calculate_document_frequency(corpus)
            
            self.assertIsInstance(doc_freq, dict)
            self.assertEqual(doc_freq['machine'], 2)
            self.assertEqual(doc_freq['learning'], 2)
            self.assertEqual(doc_freq['diagnosis'], 2)

    @patch('modules.enhanced_text_features.nltk')
    def test_query_term_coverage(self, mock_nltk):
        """Test query term coverage calculation"""
        from modules.enhanced_text_features import EnhancedTextFeatures
        
        # Mock NLTK components
        mock_nltk.data.find.return_value = True
        mock_nltk.word_tokenize.return_value = ['machine', 'learning', 'for', 'medical', 'diagnosis']
        
        features = EnhancedTextFeatures()
        
        # Test query term coverage
        if hasattr(features, '_calculate_query_coverage'):
            query_terms = ['machine', 'learning', 'medical', 'diagnosis']
            document_terms = ['machine', 'learning', 'healthcare', 'applications']
            
            coverage = features._calculate_query_coverage(query_terms, document_terms)
            
            self.assertIsInstance(coverage, (int, float))
            self.assertGreaterEqual(coverage, 0)
            self.assertLessEqual(coverage, 1)

    @patch('modules.enhanced_text_features.nltk')
    def test_semantic_similarity_features(self, mock_nltk):
        """Test semantic similarity features"""
        from modules.enhanced_text_features import EnhancedTextFeatures
        
        # Mock NLTK components
        mock_nltk.data.find.return_value = True
        mock_nltk.word_tokenize.return_value = ['machine', 'learning', 'for', 'medical', 'diagnosis']
        
        features = EnhancedTextFeatures()
        
        # Test semantic similarity features
        if hasattr(features, '_extract_semantic_features'):
            semantic_features = features._extract_semantic_features(
                self.test_query, 
                self.test_result['title'] + ' ' + self.test_result['description']
            )
            
            self.assertIsInstance(semantic_features, dict)

    @patch('modules.enhanced_text_features.nltk')
    def test_statistical_features(self, mock_nltk):
        """Test statistical text features"""
        from modules.enhanced_text_features import EnhancedTextFeatures
        
        # Mock NLTK components
        mock_nltk.data.find.return_value = True
        mock_nltk.word_tokenize.return_value = ['machine', 'learning', 'for', 'medical', 'diagnosis']
        
        features = EnhancedTextFeatures()
        
        # Test statistical features
        if hasattr(features, '_extract_statistical_features'):
            statistical_features = features._extract_statistical_features(
                self.test_result['title'] + ' ' + self.test_result['description']
            )
            
            self.assertIsInstance(statistical_features, dict)
            
            # Check for expected statistical features
            expected_stats = ['word_count', 'char_count', 'avg_word_length', 'sentence_count']
            for feature in expected_stats:
                if feature in statistical_features:
                    self.assertGreaterEqual(statistical_features[feature], 0)

    @patch('modules.enhanced_text_features.nltk')
    def test_pos_tagging_features(self, mock_nltk):
        """Test part-of-speech tagging features"""
        from modules.enhanced_text_features import EnhancedTextFeatures
        
        # Mock NLTK components
        mock_nltk.data.find.return_value = True
        mock_nltk.word_tokenize.return_value = ['machine', 'learning', 'for', 'medical', 'diagnosis']
        mock_nltk.pos_tag.return_value = [
            ('machine', 'NN'), ('learning', 'VBG'), ('for', 'IN'), 
            ('medical', 'JJ'), ('diagnosis', 'NN')
        ]
        
        features = EnhancedTextFeatures()
        
        # Test POS tagging features
        if hasattr(features, '_extract_pos_features'):
            pos_features = features._extract_pos_features(
                self.test_result['title'] + ' ' + self.test_result['description']
            )
            
            self.assertIsInstance(pos_features, dict)
            
            # Check for expected POS features
            expected_pos = ['noun_count', 'verb_count', 'adj_count', 'adv_count']
            for feature in expected_pos:
                if feature in pos_features:
                    self.assertGreaterEqual(pos_features[feature], 0)

    @patch('modules.enhanced_text_features.nltk')
    def test_empty_text_handling(self, mock_nltk):
        """Test handling of empty text"""
        from modules.enhanced_text_features import EnhancedTextFeatures
        
        # Mock NLTK components
        mock_nltk.data.find.return_value = True
        mock_nltk.word_tokenize.return_value = []
        
        features = EnhancedTextFeatures()
        
        # Test with empty text
        empty_result = {
            'id': 'empty',
            'title': '',
            'description': '',
            'author': '',
            'type': 'article'
        }
        
        extracted_features = features.extract_all_features(
            "", empty_result, self.processed_query
        )
        
        self.assertIsInstance(extracted_features, dict)
        
        # Features should still be extractable (with zero/default values)
        for feature_value in extracted_features.values():
            self.assertIsInstance(feature_value, (int, float))

    @patch('modules.enhanced_text_features.nltk')
    def test_special_characters_handling(self, mock_nltk):
        """Test handling of special characters"""
        from modules.enhanced_text_features import EnhancedTextFeatures
        
        # Mock NLTK components
        mock_nltk.data.find.return_value = True
        mock_nltk.word_tokenize.return_value = ['machine', 'learning', 'medical', 'diagnosis']
        
        features = EnhancedTextFeatures()
        
        # Test with special characters
        special_result = {
            'id': 'special',
            'title': 'Machine Learning: AI & ML!',
            'description': 'Text with @#$%^&*() special characters, numbers 123, and punctuation...',
            'author': 'Dr. Test',
            'type': 'article'
        }
        
        extracted_features = features.extract_all_features(
            "machine learning @#$", special_result, self.processed_query
        )
        
        self.assertIsInstance(extracted_features, dict)
        self.assertGreater(len(extracted_features), 0)

    @patch('modules.enhanced_text_features.nltk')
    def test_multilingual_text_handling(self, mock_nltk):
        """Test handling of multilingual text"""
        from modules.enhanced_text_features import EnhancedTextFeatures
        
        # Mock NLTK components
        mock_nltk.data.find.return_value = True
        mock_nltk.word_tokenize.return_value = ['machine', 'learning', 'médical', 'diagnostic']
        
        features = EnhancedTextFeatures()
        
        # Test with multilingual text
        multilingual_result = {
            'id': 'multilingual',
            'title': 'Machine Learning en Médecine',
            'description': 'Apprentissage automatique pour le diagnostic médical',
            'author': 'Dr. Dupont',
            'type': 'article'
        }
        
        extracted_features = features.extract_all_features(
            "machine learning médical", multilingual_result, self.processed_query
        )
        
        self.assertIsInstance(extracted_features, dict)
        self.assertGreater(len(extracted_features), 0)

    @patch('modules.enhanced_text_features.nltk')
    def test_feature_normalization(self, mock_nltk):
        """Test feature normalization"""
        from modules.enhanced_text_features import EnhancedTextFeatures
        
        # Mock NLTK components
        mock_nltk.data.find.return_value = True
        mock_nltk.word_tokenize.return_value = ['machine', 'learning', 'for', 'medical', 'diagnosis']
        
        features = EnhancedTextFeatures()
        
        # Test feature normalization
        if hasattr(features, '_normalize_features'):
            raw_features = {
                'feature1': 100,
                'feature2': 0.5,
                'feature3': -10,
                'feature4': 1000
            }
            
            normalized_features = features._normalize_features(raw_features)
            
            self.assertIsInstance(normalized_features, dict)
            self.assertEqual(len(normalized_features), len(raw_features))
            
            # Check normalization bounds
            for value in normalized_features.values():
                self.assertGreaterEqual(value, 0)
                self.assertLessEqual(value, 1)

    @patch('modules.enhanced_text_features.nltk')
    def test_corpus_management(self, mock_nltk):
        """Test corpus management for BM25"""
        from modules.enhanced_text_features import EnhancedTextFeatures
        
        # Mock NLTK components
        mock_nltk.data.find.return_value = True
        mock_nltk.word_tokenize.return_value = ['machine', 'learning', 'for', 'medical', 'diagnosis']
        
        features = EnhancedTextFeatures()
        
        # Test corpus building
        documents = [
            "Machine learning in healthcare",
            "Medical diagnosis with AI",
            "Deep learning for medical imaging"
        ]
        
        if hasattr(features, '_build_corpus'):
            features._build_corpus(documents)
            
            self.assertGreater(len(features.corpus), 0)
            self.assertGreater(len(features.corpus_tokens), 0)

    @patch('modules.enhanced_text_features.nltk')
    def test_performance_with_large_text(self, mock_nltk):
        """Test performance with large text documents"""
        from modules.enhanced_text_features import EnhancedTextFeatures
        
        # Mock NLTK components
        mock_nltk.data.find.return_value = True
        mock_nltk.word_tokenize.return_value = ['machine'] * 100 + ['learning'] * 100 + ['medical'] * 100
        
        features = EnhancedTextFeatures()
        
        # Test with large text
        large_result = {
            'id': 'large',
            'title': 'Machine Learning' * 50,
            'description': 'This is a very long description ' * 200,
            'author': 'Dr. Test',
            'type': 'article'
        }
        
        extracted_features = features.extract_all_features(
            "machine learning" * 10, large_result, self.processed_query
        )
        
        self.assertIsInstance(extracted_features, dict)
        self.assertGreater(len(extracted_features), 0)


class TestEnhancedTextFeaturesIntegration(unittest.TestCase):
    """Integration tests for EnhancedTextFeatures with real components"""

    def test_enhanced_text_features_with_real_nltk(self):
        """Test EnhancedTextFeatures with actual NLTK (if available)"""
        try:
            import nltk
            from modules.enhanced_text_features import EnhancedTextFeatures
            
            # This test only runs if NLTK is available
            features = EnhancedTextFeatures()
            
            # Test that NLTK components are properly initialized
            self.assertIsNotNone(features)
            
        except ImportError:
            self.skipTest("NLTK not available")

    def test_enhanced_text_features_with_real_textstat(self):
        """Test EnhancedTextFeatures with actual textstat (if available)"""
        try:
            import textstat
            from modules.enhanced_text_features import EnhancedTextFeatures
            
            # This test only runs if textstat is available
            features = EnhancedTextFeatures()
            
            # Test that textstat functionality is available
            self.assertIsNotNone(textstat)
            
        except ImportError:
            self.skipTest("textstat not available")

    def test_enhanced_text_features_with_real_bm25(self):
        """Test EnhancedTextFeatures with actual BM25 (if available)"""
        try:
            from rank_bm25 import BM25Okapi
            from modules.enhanced_text_features import EnhancedTextFeatures
            
            # This test only runs if rank-bm25 is available
            features = EnhancedTextFeatures()
            
            # Test that BM25 functionality is available
            self.assertIsNotNone(BM25Okapi)
            
        except ImportError:
            self.skipTest("rank-bm25 not available")


if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2) 