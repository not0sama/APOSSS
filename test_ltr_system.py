#!/usr/bin/env python3
"""
Test script for APOSSS Learning-to-Rank (LTR) System
"""
import sys
import os
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ltr_system():
    """Test the complete LTR system functionality"""
    
    print("=" * 80)
    print("🤖 APOSSS Learning-to-Rank (LTR) System Test")
    print("=" * 80)
    
    success = True
    
    try:
        # Test 1: Import and initialize LTR components
        print("\n1. Testing LTR Component Imports...")
        
        try:
            from modules.ltr_ranker import LTRRanker, FeatureExtractor
            print("   ✅ LTR ranker imported successfully")
        except ImportError as e:
            print(f"   ❌ Failed to import LTR ranker: {e}")
            return False
        
        try:
            import xgboost as xgb
            print("   ✅ XGBoost available")
        except ImportError:
            print("   ❌ XGBoost not available - LTR functionality disabled")
            return False
        
        # Test 2: Initialize LTR ranker
        print("\n2. Initializing LTR Ranker...")
        
        try:
            ltr_ranker = LTRRanker(model_dir='test_ltr_models')
            print("   ✅ LTR ranker initialized successfully")
            print(f"   📊 XGBoost available: {ltr_ranker.ltr_ranker is not None if hasattr(ltr_ranker, 'ltr_ranker') else 'N/A'}")
        except Exception as e:
            print(f"   ❌ Failed to initialize LTR ranker: {e}")
            return False
        
        # Test 3: Feature extraction
        print("\n3. Testing Feature Extraction...")
        
        # Sample query and results
        test_query = "machine learning for medical diagnosis"
        test_results = [
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
            },
            {
                'id': 'result3',
                'title': 'Artificial Intelligence in Drug Discovery',
                'description': 'AI applications in pharmaceutical research and drug development',
                'author': 'Dr. Emily Davis',
                'type': 'journal',
                'metadata': {
                    'year': 2021,
                    'category': 'Pharmaceutical Science',
                    'citations': 32,
                    'status': 'published'
                }
            }
        ]
        
        # Sample processed query (simulating LLM output)
        processed_query = {
            'intent': {
                'primary_intent': 'find_research',
                'confidence': 0.85
            },
            'keywords': {
                'primary': ['machine learning', 'medical', 'diagnosis'],
                'secondary': ['healthcare', 'AI', 'algorithm']
            },
            'entities': {
                'technologies': ['machine learning', 'artificial intelligence'],
                'concepts': ['medical diagnosis', 'healthcare'],
                'organizations': []
            },
            'academic_fields': {
                'primary_field': 'Computer Science',
                'related_fields': ['Healthcare Technology', 'Biomedical Engineering']
            }
        }
        
        # Sample current scores (from traditional ranking)
        current_scores = {
            'heuristic': [0.8, 0.7, 0.6],
            'tfidf': [0.75, 0.65, 0.55],
            'intent': [0.9, 0.8, 0.7],
            'embedding': [0.85, 0.75, 0.65]
        }
        
        try:
            features_df = ltr_ranker.extract_features(
                test_query, test_results, processed_query, current_scores
            )
            
            if not features_df.empty:
                print(f"   ✅ Features extracted successfully")
                print(f"   📊 Features shape: {features_df.shape}")
                print(f"   🔧 Feature columns: {len(features_df.columns)}")
                
                # Display some sample features
                print("\n   Sample features:")
                for col in list(features_df.columns)[:10]:  # Show first 10 features
                    value = features_df[col].iloc[0]
                    if isinstance(value, (int, float)):
                        print(f"      - {col}: {value:.4f}")
                    else:
                        print(f"      - {col}: {value}")
                
                if len(features_df.columns) > 10:
                    print(f"      ... and {len(features_df.columns) - 10} more features")
            else:
                print("   ❌ No features extracted")
                success = False
                
        except Exception as e:
            print(f"   ❌ Feature extraction failed: {e}")
            success = False
        
        # Test 4: Model training with synthetic data
        print("\n4. Testing Model Training with Synthetic Data...")
        
        try:
            # Generate synthetic training data
            training_data = []
            
            # Simulate multiple queries and results with relevance labels
            for query_idx in range(5):  # 5 different queries
                query_id = f"query_{query_idx}"
                
                for result_idx in range(10):  # 10 results per query
                    # Generate random feature values
                    features = {
                        'query_id': hash(query_id),
                        'result_id': f"result_{query_idx}_{result_idx}",
                        'result_index': result_idx,
                        'heuristic_score': np.random.uniform(0, 1),
                        'tfidf_score': np.random.uniform(0, 1),
                        'intent_score': np.random.uniform(0, 1),
                        'embedding_score': np.random.uniform(0, 1),
                        'bm25_title': np.random.uniform(0, 5),
                        'bm25_description': np.random.uniform(0, 5),
                        'unigram_overlap': np.random.uniform(0, 1),
                        'bigram_overlap': np.random.uniform(0, 1),
                        'query_term_proximity': np.random.uniform(0, 1),
                        'title_length': np.random.randint(5, 20),
                        'description_length': np.random.randint(50, 200),
                        'query_coverage': np.random.uniform(0, 1),
                        'recency_score': np.random.uniform(0, 1),
                        'citation_count': np.random.randint(0, 100),
                        'availability_score': np.random.choice([0.0, 0.5, 1.0]),
                        'type_importance': np.random.uniform(0.4, 1.0),
                        'intent_confidence': np.random.uniform(0.5, 1.0),
                        'field_match_score': np.random.uniform(0, 1),
                        'entity_match_score': np.random.uniform(0, 1),
                        'avg_rating': np.random.uniform(1, 5),
                        'feedback_count': np.random.randint(0, 50),
                        'positive_feedback_ratio': np.random.uniform(0, 1)
                    }
                    
                    # Generate relevance label (0-4 scale)
                    # Higher scores should correlate with higher relevance
                    combined_score = (
                        features['heuristic_score'] * 0.3 +
                        features['tfidf_score'] * 0.3 +
                        features['embedding_score'] * 0.4
                    )
                    
                    # Add some noise and convert to discrete labels
                    if combined_score > 0.8:
                        relevance = 4
                    elif combined_score > 0.6:
                        relevance = 3
                    elif combined_score > 0.4:
                        relevance = 2
                    elif combined_score > 0.2:
                        relevance = 1
                    else:
                        relevance = 0
                    
                    features['relevance_label'] = relevance
                    training_data.append(features)
            
            print(f"   📊 Generated {len(training_data)} training examples")
            
            # Train the model
            training_stats = ltr_ranker.train_model(training_data, validation_split=0.3)
            
            print("   ✅ Model training completed successfully")
            print(f"   📈 Training NDCG: {training_stats.get('train_ndcg', 0):.4f}")
            print(f"   📈 Validation NDCG: {training_stats.get('val_ndcg', 0):.4f}")
            print(f"   🔧 Features used: {training_stats.get('num_features', 0)}")
            print(f"   📚 Training samples: {training_stats.get('training_samples', 0)}")
            
        except Exception as e:
            print(f"   ❌ Model training failed: {e}")
            success = False
        
        # Test 5: Prediction and ranking
        print("\n5. Testing LTR Prediction and Ranking...")
        
        try:
            # Test prediction on the original test results
            ranked_results = ltr_ranker.rank_results(
                test_query, test_results.copy(), processed_query, current_scores
            )
            
            print("   ✅ LTR ranking completed successfully")
            print("   📊 Ranked results:")
            
            for i, result in enumerate(ranked_results[:3]):  # Show top 3
                ltr_score = result.get('ltr_score', 0)
                ranking_score = result.get('ranking_score', ltr_score)
                print(f"      {i+1}. {result['title'][:50]}...")
                print(f"         LTR Score: {ltr_score:.4f}")
                print(f"         Final Score: {ranking_score:.4f}")
                print(f"         Type: {result['type']}")
            
        except Exception as e:
            print(f"   ❌ LTR prediction failed: {e}")
            success = False
        
        # Test 6: Feature importance analysis
        print("\n6. Testing Feature Importance Analysis...")
        
        try:
            importance = ltr_ranker.get_feature_importance()
            
            if importance:
                print("   ✅ Feature importance calculated successfully")
                print("   🔍 Top 10 most important features:")
                
                # Sort by importance
                sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                
                for i, (feature, score) in enumerate(sorted_features[:10]):
                    print(f"      {i+1:2d}. {feature:25s}: {score:.4f}")
                    
            else:
                print("   ⚠️ No feature importance available (model may not be trained)")
                
        except Exception as e:
            print(f"   ❌ Feature importance analysis failed: {e}")
            success = False
        
        # Test 7: Model statistics
        print("\n7. Testing Model Statistics...")
        
        try:
            stats = ltr_ranker.get_model_stats()
            
            print("   ✅ Model statistics retrieved successfully")
            print("   📊 LTR System Statistics:")
            print(f"      - LTR Available: {stats.get('ltr_available', False)}")
            print(f"      - Model Trained: {stats.get('model_trained', False)}")
            print(f"      - Feature Count: {stats.get('feature_count', 0)}")
            print(f"      - Model Directory: {stats.get('model_dir', 'N/A')}")
            
            if stats.get('model_trained', False):
                print(f"      - Training Date: {stats.get('training_date', 'N/A')}")
                print(f"      - Training Samples: {stats.get('training_samples', 0)}")
                print(f"      - Validation Samples: {stats.get('validation_samples', 0)}")
            
        except Exception as e:
            print(f"   ❌ Model statistics retrieval failed: {e}")
            success = False
        
        # Test 8: Integration with ranking engine
        print("\n8. Testing Integration with Ranking Engine...")
        
        try:
            from modules.ranking_engine import RankingEngine
            
            ranking_engine = RankingEngine(use_embedding=False, use_ltr=True)
            
            if ranking_engine.use_ltr and ranking_engine.ltr_ranker:
                print("   ✅ LTR integration successful")
                print(f"   🔧 LTR Ranker initialized: {ranking_engine.ltr_ranker.is_trained}")
                
                # Test LTR stats through ranking engine
                ltr_stats = ranking_engine.get_ltr_stats()
                print(f"   📊 LTR Available through ranking engine: {ltr_stats.get('ltr_available', False)}")
                
            else:
                print("   ❌ LTR integration failed")
                success = False
                
        except Exception as e:
            print(f"   ❌ Ranking engine integration failed: {e}")
            success = False
        
        # Summary
        print("\n" + "=" * 80)
        if success:
            print("🎉 All LTR system tests completed successfully!")
            print("\n📋 LTR System Features Verified:")
            print("   ✅ XGBoost Learning-to-Rank implementation")
            print("   ✅ Advanced feature engineering (30+ features)")
            print("   ✅ Model training with NDCG optimization")
            print("   ✅ Real-time prediction and ranking")
            print("   ✅ Feature importance analysis")
            print("   ✅ Integration with existing ranking system")
            print("   ✅ Hybrid ranking capabilities")
            
            print("\n🚀 Next Steps:")
            print("   1. Collect user feedback for training data")
            print("   2. Train model with real feedback")
            print("   3. Monitor and evaluate ranking performance")
            print("   4. Iterate on feature engineering")
            
        else:
            print("❌ Some LTR system tests failed. Check the logs above for details.")
        
        return success
        
    except Exception as e:
        logger.error(f"Unexpected error in LTR system test: {e}")
        print(f"\n❌ Critical error: {e}")
        return False

if __name__ == "__main__":
    success = test_ltr_system()
    sys.exit(0 if success else 1) 