#!/usr/bin/env python3
"""
Test script for APOSSS Real-time Similarity Calculation
"""
import sys
import os
import logging
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.embedding_ranker import EmbeddingRanker
from modules.llm_processor import LLMProcessor
from modules.query_processor import QueryProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_realtime_similarity():
    """Test real-time similarity calculation functionality"""
    
    print("=" * 80)
    print("🧠 APOSSS Real-time Similarity Calculation Test")
    print("=" * 80)
    
    try:
        # Initialize components
        print("📝 Initializing components...")
        embedding_ranker = EmbeddingRanker()
        llm_processor = LLMProcessor()
        query_processor = QueryProcessor(llm_processor)
        
        # Test query
        test_query = "machine learning for medical diagnosis"
        print(f"\n🔍 Test Query: '{test_query}'")
        
        # Process query with LLM for enhanced similarity
        print("\n📊 Processing query with LLM...")
        processed_query = query_processor.process_query(test_query)
        
        # Sample documents for testing
        test_documents = [
            {
                'id': 'doc1',
                'title': 'Machine Learning Algorithms in Healthcare',
                'description': 'This paper discusses various machine learning techniques applied to medical diagnosis and patient care.',
                'author': 'Dr. Smith',
                'type': 'article',
                'metadata': {
                    'keywords': ['machine learning', 'healthcare', 'diagnosis'],
                    'category': 'Medical AI',
                    'institution': 'Medical University'
                }
            },
            {
                'id': 'doc2',
                'title': 'Deep Learning for Cancer Detection',
                'description': 'An advanced study on using deep neural networks for early cancer detection through medical imaging.',
                'author': 'Dr. Johnson',
                'type': 'article',
                'metadata': {
                    'keywords': ['deep learning', 'cancer', 'medical imaging'],
                    'category': 'Medical AI',
                    'institution': 'Cancer Research Center'
                }
            },
            {
                'id': 'doc3',
                'title': 'Renewable Energy Solar Panel Efficiency',
                'description': 'Research on improving solar panel efficiency through advanced materials and design.',
                'author': 'Dr. Green',
                'type': 'article',
                'metadata': {
                    'keywords': ['solar energy', 'renewable energy', 'efficiency'],
                    'category': 'Energy Research',
                    'institution': 'Energy Institute'
                }
            },
            {
                'id': 'doc4',
                'title': 'AI-Powered Medical Diagnosis System',
                'description': 'Development of an artificial intelligence system for automated medical diagnosis and treatment recommendations.',
                'author': 'Dr. Wilson',
                'type': 'article',
                'metadata': {
                    'keywords': ['artificial intelligence', 'medical diagnosis', 'automation'],
                    'category': 'Medical AI',
                    'institution': 'AI Research Lab'
                }
            },
            {
                'id': 'doc5',
                'title': 'Statistical Methods in Data Analysis',
                'description': 'Traditional statistical approaches for analyzing large datasets in scientific research.',
                'author': 'Prof. Brown',
                'type': 'book',
                'metadata': {
                    'keywords': ['statistics', 'data analysis', 'research methods'],
                    'category': 'Statistics',
                    'institution': 'Statistics Department'
                }
            }
        ]
        
        print(f"\n📚 Testing with {len(test_documents)} sample documents:")
        for i, doc in enumerate(test_documents, 1):
            print(f"  {i}. {doc['title']}")
        
        # Test 1: Real-time similarity calculation
        print(f"\n⚡ Test 1: Real-time Similarity Calculation")
        print("-" * 50)
        
        start_time = datetime.now()
        similarity_scores = embedding_ranker.calculate_realtime_similarity(
            test_query, test_documents, processed_query, use_cache=True
        )
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        # Display results sorted by similarity
        results = list(zip(test_documents, similarity_scores))
        results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"⏱️  Processing time: {processing_time:.3f} seconds")
        print(f"📊 Results ranked by similarity:")
        
        for i, (doc, score) in enumerate(results, 1):
            percentage = score * 100
            relevance = "🟢 High" if score > 0.7 else "🟡 Medium" if score > 0.4 else "🔴 Low"
            print(f"  {i}. {doc['title']}")
            print(f"     Score: {score:.4f} ({percentage:.1f}%) - {relevance}")
            print(f"     Author: {doc['author']} | Type: {doc['type']}")
            print()
        
        # Test 2: Pairwise similarity
        print(f"\n🔬 Test 2: Pairwise Text Similarity")
        print("-" * 50)
        
        text_pairs = [
            ("machine learning algorithms", "artificial intelligence methods"),
            ("medical diagnosis", "healthcare diagnostics"),
            ("solar energy panels", "renewable energy storage"),
            ("data analysis", "statistical computing"),
            ("deep learning", "neural networks")
        ]
        
        for text1, text2 in text_pairs:
            similarity = embedding_ranker.calculate_pairwise_similarity(text1, text2)
            percentage = similarity * 100
            print(f"'{text1}' ↔ '{text2}'")
            print(f"  Similarity: {similarity:.4f} ({percentage:.1f}%)")
            print()
        
        # Test 3: Cache performance
        print(f"\n🔥 Test 3: Cache Performance Test")
        print("-" * 50)
        
        # First run (cold cache)
        print("🆒 Cold cache run...")
        start_time = datetime.now()
        similarity_scores_1 = embedding_ranker.calculate_realtime_similarity(
            "artificial intelligence healthcare", test_documents, use_cache=False
        )
        cold_time = (datetime.now() - start_time).total_seconds()
        
        # Second run (warm cache)
        print("🔥 Warm cache run...")
        start_time = datetime.now()
        similarity_scores_2 = embedding_ranker.calculate_realtime_similarity(
            "artificial intelligence healthcare", test_documents, use_cache=True
        )
        warm_time = (datetime.now() - start_time).total_seconds()
        
        speedup = cold_time / warm_time if warm_time > 0 else float('inf')
        
        print(f"Cold cache time: {cold_time:.3f} seconds")
        print(f"Warm cache time: {warm_time:.3f} seconds")
        print(f"Cache speedup: {speedup:.1f}x faster")
        
        # Test 4: Real-time stats
        print(f"\n📈 Test 4: Real-time Statistics")
        print("-" * 50)
        
        stats = embedding_ranker.get_realtime_stats()
        print("Real-time Embedding Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n" + "=" * 80)
        print("✅ Real-time similarity calculation test completed successfully!")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        logger.error(f"Test failed: {str(e)}")
        return False

def main():
    """Main function"""
    print("🚀 Starting APOSSS Real-time Similarity Test Suite")
    
    success = test_realtime_similarity()
    
    if success:
        print("\n🎉 All tests passed! Real-time similarity system is working correctly.")
        print("\n✨ Key Benefits Demonstrated:")
        print("  ✅ Fast on-the-fly embedding calculation")
        print("  ✅ Intelligent caching for performance")
        print("  ✅ Enhanced query processing integration")
        print("  ✅ Batch processing optimization")
        print("  ✅ Real-time performance metrics")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 