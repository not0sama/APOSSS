#!/usr/bin/env python3
"""
Unit tests for APOSSS LTRRanker class
Tests Learning-to-Rank functionality including feature extraction, model training, and predictions
"""
import sys
import os
import unittest
import logging
import numpy as np
import pandas as pd
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestLTRRanker(unittest.TestCase):
    """Unit tests for LTRRanker class"""

    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directory for test models
        self.temp_dir = tempfile.mkdtemp()
        
        # Sample test data
        self.test_query = "machine learning for medical diagnosis"
        self.test_results = [
            {
                'id': 'result1',
                'title': 'Machine Learning in Healthcare',
                'description': 'Comprehensive overview of ML applications in medical diagnosis and treatment',
                'author': 'Dr. Sarah Johnson',
                'type': 'article',
                'metadata': {
                    'year': 2023,
                    'category': 'Healthcare Technology',
                    'citations': 15,
                    'status': 'published'
                }
            },
            {
                'id': 'result2', 
                'title': 'Deep Learning for Medical Imaging',
                'description': 'Advanced deep learning techniques for analyzing medical images and scans',
                'author': 'Prof. Michael Chen',
                'type': 'book',
                'metadata': {
                    'year': 2022,
                    'category': 'Computer Science',
                    'citations': 45,
                    'status': 'available'
                }
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
            'academic_fields': {
                'primary_field': 'Computer Science',
                'related_fields': ['Healthcare Technology']
            },
            'intent': {
                'primary_intent': 'find_research',
                'confidence': 0.85
            }
        }
        
        self.current_scores = {
            'heuristic': [0.8, 0.7],
            'tfidf': [0.75, 0.65],
            'intent': [0.9, 0.8],
            'embedding': [0.85, 0.75]
        }

    def tearDown(self):
        """Clean up test fixtures"""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('modules.ltr_ranker.XGBOOST_AVAILABLE', True)
    @patch('modules.ltr_ranker.xgb')
    def test_ltr_ranker_initialization(self, mock_xgb):
        """Test LTRRanker initialization"""
        from modules.ltr_ranker import LTRRanker
        
        # Test successful initialization
        ltr_ranker = LTRRanker(model_dir=self.temp_dir)
        
        self.assertIsNotNone(ltr_ranker)
        self.assertEqual(ltr_ranker.model_dir, self.temp_dir)
        self.assertEqual(ltr_ranker.is_trained, False)
        self.assertIsNotNone(ltr_ranker.feature_extractor)

    @patch('modules.ltr_ranker.XGBOOST_AVAILABLE', False)
    def test_ltr_ranker_initialization_no_xgboost(self):
        """Test LTRRanker initialization without XGBoost"""
        from modules.ltr_ranker import LTRRanker
        
        # Test initialization without XGBoost
        ltr_ranker = LTRRanker(model_dir=self.temp_dir)
        
        self.assertIsNotNone(ltr_ranker)
        self.assertEqual(ltr_ranker.is_trained, False)

    @patch('modules.ltr_ranker.XGBOOST_AVAILABLE', True)
    @patch('modules.ltr_ranker.xgb')
    def test_feature_extraction(self, mock_xgb):
        """Test feature extraction from query and results"""
        from modules.ltr_ranker import LTRRanker
        
        ltr_ranker = LTRRanker(model_dir=self.temp_dir)
        
        # Test feature extraction
        features_df = ltr_ranker.extract_features(
            self.test_query, self.test_results, self.processed_query, self.current_scores
        )
        
        self.assertIsInstance(features_df, pd.DataFrame)
        self.assertEqual(len(features_df), len(self.test_results))
        self.assertGreater(len(features_df.columns), 0)
        
        # Check for expected feature columns
        expected_features = [
            'result_index', 'heuristic_score', 'tfidf_score', 'intent_score',
            'embedding_score', 'title_length', 'description_length'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features_df.columns)

    @patch('modules.ltr_ranker.XGBOOST_AVAILABLE', True)
    @patch('modules.ltr_ranker.xgb')
    def test_feature_extraction_with_feedback(self, mock_xgb):
        """Test feature extraction with user feedback data"""
        from modules.ltr_ranker import LTRRanker
        
        ltr_ranker = LTRRanker(model_dir=self.temp_dir)
        
        # Sample user feedback data
        user_feedback_data = {
            'clicked_results': ['result1'],
            'bookmarked_results': ['result1'],
            'avg_rating': 4.5,
            'feedback_count': 10
        }
        
        # Test feature extraction with feedback
        features_df = ltr_ranker.extract_features(
            self.test_query, self.test_results, self.processed_query, 
            self.current_scores, user_feedback_data
        )
        
        self.assertIsInstance(features_df, pd.DataFrame)
        self.assertEqual(len(features_df), len(self.test_results))
        
        # Check for feedback-based features
        feedback_features = ['avg_rating', 'feedback_count']
        for feature in feedback_features:
            self.assertIn(feature, features_df.columns)

    @patch('modules.ltr_ranker.XGBOOST_AVAILABLE', True)
    @patch('modules.ltr_ranker.xgb')
    def test_model_training(self, mock_xgb):
        """Test LTR model training"""
        from modules.ltr_ranker import LTRRanker
        
        # Mock XGBoost model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.8, 0.6, 0.4])
        mock_xgb.train.return_value = mock_model
        
        ltr_ranker = LTRRanker(model_dir=self.temp_dir)
        
        # Create sample training data
        training_data = []
        for query_idx in range(3):
            for result_idx in range(3):
                training_data.append({
                    'query_id': f'query_{query_idx}',
                    'result_id': f'result_{result_idx}',
                    'relevance_score': 2 - result_idx,  # Decreasing relevance
                    'heuristic_score': 0.8 - result_idx * 0.1,
                    'tfidf_score': 0.7 - result_idx * 0.1,
                    'intent_score': 0.9 - result_idx * 0.1
                })
        
        # Test model training
        success = ltr_ranker.train_model(training_data)
        
        self.assertTrue(success)
        self.assertTrue(ltr_ranker.is_trained)
        mock_xgb.train.assert_called_once()

    @patch('modules.ltr_ranker.XGBOOST_AVAILABLE', True)
    @patch('modules.ltr_ranker.xgb')
    def test_model_training_with_validation(self, mock_xgb):
        """Test LTR model training with validation split"""
        from modules.ltr_ranker import LTRRanker
        
        # Mock XGBoost model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.8, 0.6, 0.4])
        mock_xgb.train.return_value = mock_model
        
        ltr_ranker = LTRRanker(model_dir=self.temp_dir)
        
        # Create sample training data (larger dataset for validation split)
        training_data = []
        for query_idx in range(10):
            for result_idx in range(5):
                training_data.append({
                    'query_id': f'query_{query_idx}',
                    'result_id': f'result_{result_idx}',
                    'relevance_score': 4 - result_idx,  # Relevance scores 4,3,2,1,0
                    'heuristic_score': 0.9 - result_idx * 0.1,
                    'tfidf_score': 0.8 - result_idx * 0.1,
                    'intent_score': 0.95 - result_idx * 0.05
                })
        
        # Test model training with validation
        success = ltr_ranker.train_model(training_data, validation_split=0.2)
        
        self.assertTrue(success)
        self.assertTrue(ltr_ranker.is_trained)

    @patch('modules.ltr_ranker.XGBOOST_AVAILABLE', True)
    @patch('modules.ltr_ranker.xgb')
    def test_model_prediction(self, mock_xgb):
        """Test LTR model prediction"""
        from modules.ltr_ranker import LTRRanker
        
        # Mock XGBoost model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.8, 0.6])
        mock_xgb.train.return_value = mock_model
        
        ltr_ranker = LTRRanker(model_dir=self.temp_dir)
        ltr_ranker.model = mock_model
        ltr_ranker.is_trained = True
        
        # Create sample features
        features_df = pd.DataFrame({
            'heuristic_score': [0.8, 0.7],
            'tfidf_score': [0.75, 0.65],
            'intent_score': [0.9, 0.8]
        })
        
        # Test prediction
        scores = ltr_ranker.predict_scores(features_df)
        
        self.assertEqual(len(scores), 2)
        self.assertTrue(all(isinstance(score, (int, float)) for score in scores))
        mock_model.predict.assert_called_once()

    @patch('modules.ltr_ranker.XGBOOST_AVAILABLE', True)
    @patch('modules.ltr_ranker.xgb')
    def test_rank_results(self, mock_xgb):
        """Test ranking results using LTR model"""
        from modules.ltr_ranker import LTRRanker
        
        # Mock XGBoost model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.8, 0.6])
        mock_xgb.train.return_value = mock_model
        
        ltr_ranker = LTRRanker(model_dir=self.temp_dir)
        ltr_ranker.model = mock_model
        ltr_ranker.is_trained = True
        
        # Test ranking results
        ranked_results = ltr_ranker.rank_results(
            self.test_query, self.test_results, self.processed_query, self.current_scores
        )
        
        self.assertEqual(len(ranked_results), len(self.test_results))
        
        # Check that results have LTR scores and ranks
        for result in ranked_results:
            self.assertIn('ltr_score', result)
            self.assertIn('ltr_rank', result)
        
        # Check that results are sorted by LTR score
        scores = [result['ltr_score'] for result in ranked_results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    @patch('modules.ltr_ranker.XGBOOST_AVAILABLE', True)
    @patch('modules.ltr_ranker.xgb')
    def test_rank_results_untrained_model(self, mock_xgb):
        """Test ranking results with untrained model"""
        from modules.ltr_ranker import LTRRanker
        
        ltr_ranker = LTRRanker(model_dir=self.temp_dir)
        ltr_ranker.is_trained = False
        
        # Test ranking with untrained model
        ranked_results = ltr_ranker.rank_results(
            self.test_query, self.test_results, self.processed_query, self.current_scores
        )
        
        self.assertEqual(len(ranked_results), len(self.test_results))
        
        # Check that results have placeholder LTR scores
        for result in ranked_results:
            self.assertIn('ltr_score', result)
            self.assertEqual(result['ltr_score'], 0.5)  # Default neutral score

    @patch('modules.ltr_ranker.XGBOOST_AVAILABLE', True)
    @patch('modules.ltr_ranker.xgb')
    def test_model_save_load(self, mock_xgb):
        """Test model saving and loading"""
        from modules.ltr_ranker import LTRRanker
        
        # Mock XGBoost model
        mock_model = Mock()
        mock_model.save_model = Mock()
        mock_xgb.train.return_value = mock_model
        mock_xgb.Booster.return_value = mock_model
        
        ltr_ranker = LTRRanker(model_dir=self.temp_dir)
        ltr_ranker.model = mock_model
        ltr_ranker.is_trained = True
        ltr_ranker.feature_names = ['feature1', 'feature2']
        
        # Test model saving
        ltr_ranker._save_model()
        
        # Test model loading
        ltr_ranker2 = LTRRanker(model_dir=self.temp_dir)
        # The load should be attempted during initialization

    @patch('modules.ltr_ranker.XGBOOST_AVAILABLE', True)
    @patch('modules.ltr_ranker.xgb')
    def test_empty_results_handling(self, mock_xgb):
        """Test handling of empty results"""
        from modules.ltr_ranker import LTRRanker
        
        ltr_ranker = LTRRanker(model_dir=self.temp_dir)
        
        # Test with empty results
        empty_results = []
        ranked_results = ltr_ranker.rank_results(
            self.test_query, empty_results, self.processed_query, self.current_scores
        )
        
        self.assertEqual(ranked_results, empty_results)

    @patch('modules.ltr_ranker.XGBOOST_AVAILABLE', True)
    @patch('modules.ltr_ranker.xgb')
    def test_feature_importance(self, mock_xgb):
        """Test feature importance analysis"""
        from modules.ltr_ranker import LTRRanker
        
        # Mock XGBoost model with feature importance
        mock_model = Mock()
        mock_model.get_importance.return_value = {
            'feature1': 0.5,
            'feature2': 0.3,
            'feature3': 0.2
        }
        mock_xgb.train.return_value = mock_model
        
        ltr_ranker = LTRRanker(model_dir=self.temp_dir)
        ltr_ranker.model = mock_model
        ltr_ranker.is_trained = True
        
        # Test feature importance
        importance = ltr_ranker.get_feature_importance()
        
        self.assertIsInstance(importance, dict)
        mock_model.get_importance.assert_called_once()

    def test_feature_extractor_initialization(self):
        """Test FeatureExtractor initialization"""
        from modules.ltr_ranker import FeatureExtractor
        
        feature_extractor = FeatureExtractor()
        
        self.assertIsNotNone(feature_extractor)

    def test_feature_extractor_textual_features(self):
        """Test textual feature extraction"""
        from modules.ltr_ranker import FeatureExtractor
        
        feature_extractor = FeatureExtractor()
        
        # Test textual feature extraction
        textual_features = feature_extractor.extract_textual_features(
            self.test_query, self.test_results[0], self.processed_query
        )
        
        self.assertIsInstance(textual_features, dict)
        self.assertGreater(len(textual_features), 0)
        
        # Check for expected textual features
        expected_features = [
            'title_length', 'description_length', 'query_coverage'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, textual_features)

    def test_feature_extractor_metadata_features(self):
        """Test metadata feature extraction"""
        from modules.ltr_ranker import FeatureExtractor
        
        feature_extractor = FeatureExtractor()
        
        # Test metadata feature extraction
        metadata_features = feature_extractor.extract_metadata_features(
            self.test_results[0], self.processed_query
        )
        
        self.assertIsInstance(metadata_features, dict)
        self.assertGreater(len(metadata_features), 0)
        
        # Check for expected metadata features
        expected_features = [
            'recency_score', 'citation_count', 'type_importance'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, metadata_features)

    def test_feature_extractor_intent_features(self):
        """Test intent-based feature extraction"""
        from modules.ltr_ranker import FeatureExtractor
        
        feature_extractor = FeatureExtractor()
        
        # Test intent feature extraction
        intent_features = feature_extractor.extract_intent_features(
            self.test_results[0], self.processed_query
        )
        
        self.assertIsInstance(intent_features, dict)
        self.assertGreater(len(intent_features), 0)
        
        # Check for expected intent features
        expected_features = [
            'intent_confidence', 'field_match_score', 'entity_match_score'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, intent_features)

    def test_feature_extractor_feedback_features(self):
        """Test feedback-based feature extraction"""
        from modules.ltr_ranker import FeatureExtractor
        
        feature_extractor = FeatureExtractor()
        
        # Sample feedback data
        feedback_data = {
            'clicked_results': ['result1'],
            'avg_rating': 4.2,
            'feedback_count': 15
        }
        
        # Test feedback feature extraction
        feedback_features = feature_extractor.extract_feedback_features(
            self.test_results[0], feedback_data
        )
        
        self.assertIsInstance(feedback_features, dict)
        self.assertGreater(len(feedback_features), 0)
        
        # Check for expected feedback features
        expected_features = ['avg_rating', 'feedback_count']
        
        for feature in expected_features:
            self.assertIn(feature, feedback_features)

    @patch('modules.ltr_ranker.XGBOOST_AVAILABLE', True)
    @patch('modules.ltr_ranker.xgb')
    def test_training_data_validation(self, mock_xgb):
        """Test validation of training data"""
        from modules.ltr_ranker import LTRRanker
        
        ltr_ranker = LTRRanker(model_dir=self.temp_dir)
        
        # Test with invalid training data
        invalid_training_data = []  # Empty data
        
        success = ltr_ranker.train_model(invalid_training_data)
        
        self.assertFalse(success)
        self.assertFalse(ltr_ranker.is_trained)

    @patch('modules.ltr_ranker.XGBOOST_AVAILABLE', True)
    @patch('modules.ltr_ranker.xgb')
    def test_model_evaluation_metrics(self, mock_xgb):
        """Test model evaluation metrics calculation"""
        from modules.ltr_ranker import LTRRanker
        
        # Mock XGBoost model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.8, 0.6, 0.4])
        mock_xgb.train.return_value = mock_model
        
        ltr_ranker = LTRRanker(model_dir=self.temp_dir)
        ltr_ranker.model = mock_model
        ltr_ranker.is_trained = True
        
        # Test metrics calculation
        if hasattr(ltr_ranker, 'get_training_metrics'):
            metrics = ltr_ranker.get_training_metrics()
            
            self.assertIsInstance(metrics, dict)


class TestLTRRankerIntegration(unittest.TestCase):
    """Integration tests for LTRRanker with real components"""

    def setUp(self):
        """Set up integration test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up integration test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_ltr_ranker_with_real_xgboost(self):
        """Test LTRRanker with actual XGBoost (if available)"""
        try:
            import xgboost as xgb
            from modules.ltr_ranker import LTRRanker
            
            # This test only runs if XGBoost is available
            ltr_ranker = LTRRanker(model_dir=self.temp_dir)
            
            # Test that XGBoost is properly detected
            self.assertIsNotNone(ltr_ranker)
            
        except ImportError:
            self.skipTest("XGBoost not available")

    def test_ltr_ranker_with_enhanced_text_features(self):
        """Test LTRRanker integration with EnhancedTextFeatures"""
        try:
            from modules.ltr_ranker import LTRRanker
            from modules.enhanced_text_features import EnhancedTextFeatures
            
            ltr_ranker = LTRRanker(model_dir=self.temp_dir)
            
            # Test that EnhancedTextFeatures is properly initialized
            self.assertIsNotNone(ltr_ranker.enhanced_text_features)
            
        except ImportError:
            self.skipTest("EnhancedTextFeatures not available")


if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2) 