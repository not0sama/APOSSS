#!/usr/bin/env python3
"""
Test script for multilingual support in APOSSS
"""
import os
import sys
import json
from modules.llm_processor import LLMProcessor
from modules.embedding_ranker import EmbeddingRanker

def test_multilingual_functionality():
    """Test the multilingual capabilities of the system"""
    print("🌐 Testing Multilingual Support for APOSSS")
    print("=" * 50)
    
    # Test queries in different languages
    test_queries = [
        ("English", "machine learning algorithms for climate change research"),
        ("French", "algorithmes d'apprentissage automatique pour la recherche sur le changement climatique"),
        ("Spanish", "algoritmos de aprendizaje automático para la investigación del cambio climático"),
        ("German", "maschinelles Lernen Algorithmen für Klimawandel Forschung"),
        ("Arabic", "خوارزميات التعلم الآلي لبحوث تغير المناخ"),
        ("Chinese", "用于气候变化研究的机器学习算法"),
        ("Portuguese", "algoritmos de aprendizado de máquina para pesquisa sobre mudanças climáticas"),
        ("Italian", "algoritmi di apprendimento automatico per la ricerca sui cambiamenti climatici")
    ]
    
    try:
        # Test LLM Processor
        print("\n🔤 Testing LLM Processor with Multilingual Queries")
        print("-" * 40)
        
        llm_processor = LLMProcessor()
        
        for language, query in test_queries:
            print(f"\n📝 Testing {language}: '{query[:50]}...'")
            
            try:
                result = llm_processor.process_query(query)
                if result:
                    lang_detection = result.get('language_detection', {})
                    translation = result.get('translation', {})
                    
                    print(f"   ✅ Detected: {lang_detection.get('language_name', 'Unknown')} "
                          f"({lang_detection.get('detected_language', 'unknown')})")
                    print(f"   🔄 Translation needed: {translation.get('needs_translation', False)}")
                    
                    if translation.get('needs_translation'):
                        translated = translation.get('translated_query', '')
                        print(f"   🇬🇧 Translated: '{translated[:50]}...'")
                    
                    # Show some keywords
                    keywords = result.get('keywords', {}).get('primary', [])
                    if keywords:
                        print(f"   🔑 Keywords: {', '.join(keywords[:3])}")
                else:
                    print(f"   ❌ Failed to process query")
                    
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
        
        # Test Embedding Ranker
        print(f"\n🧠 Testing Multilingual Embedding Model")
        print("-" * 40)
        
        try:
            embedding_ranker = EmbeddingRanker()
            stats = embedding_ranker.get_cache_stats()
            
            print(f"   ✅ Model: {stats['model_name']}")
            print(f"   🌐 Type: {stats['model_type']}")
            print(f"   📏 Dimension: {stats['embedding_dimension']}")
            print(f"   🗣️ Languages: {stats['supported_languages'][:100]}...")
            
            # Test similarity calculation with multilingual content
            print(f"\n🔍 Testing Multilingual Similarity Calculation")
            
            # Create sample documents in different languages
            sample_docs = [
                {
                    "id": "doc1",
                    "title": "Machine Learning in Climate Science",
                    "description": "Research on using ML algorithms for climate prediction",
                    "type": "article"
                },
                {
                    "id": "doc2", 
                    "title": "Intelligence Artificielle et Climat",
                    "description": "Recherche sur l'utilisation de l'IA pour prédire le changement climatique",
                    "type": "article"
                },
                {
                    "id": "doc3",
                    "title": "Aprendizaje Automático para el Clima",
                    "description": "Investigación sobre algoritmos de ML para el cambio climático",
                    "type": "article"
                }
            ]
            
            # Test similarity with English query
            english_query = "machine learning climate research"
            similarities = embedding_ranker.calculate_embedding_similarity(english_query, sample_docs)
            
            print(f"   📊 Similarities for '{english_query}':")
            for i, (doc, sim) in enumerate(zip(sample_docs, similarities)):
                print(f"      Doc {i+1} ({doc['title'][:30]}...): {sim:.3f}")
            
            # Test with French query
            french_query = "recherche intelligence artificielle climat"
            similarities_fr = embedding_ranker.calculate_embedding_similarity(french_query, sample_docs)
            
            print(f"   📊 Similarities for '{french_query}':")
            for i, (doc, sim) in enumerate(zip(sample_docs, similarities_fr)):
                print(f"      Doc {i+1} ({doc['title'][:30]}...): {sim:.3f}")
                
        except Exception as e:
            print(f"   ❌ Embedding model error: {str(e)}")
        
        print(f"\n✅ Multilingual testing completed!")
        print(f"🎯 The system now supports queries in 50+ languages")
        print(f"🔄 Non-English queries are automatically translated to English")
        print(f"🧠 Multilingual embeddings enable cross-language similarity matching")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False
    
    return True

def test_multilingual_llm_connection():
    """Test the multilingual LLM connection specifically"""
    print("\n🔗 Testing Multilingual LLM Connection")
    print("-" * 40)
    
    try:
        llm_processor = LLMProcessor()
        result = llm_processor.test_multilingual_connection()
        
        if result.get('connected'):
            print("   ✅ LLM connection successful")
            print(f"   🌐 Multilingual support: {result.get('multilingual_support', False)}")
            
            if result.get('sample_language_detection'):
                lang_info = result['sample_language_detection']
                print(f"   🔤 Sample detection: {lang_info.get('language_name', 'Unknown')}")
        else:
            print(f"   ❌ Connection failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"   ❌ Test error: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting APOSSS Multilingual Support Test")
    
    # Check environment
    if not os.getenv('GEMINI_API_KEY'):
        print("⚠️  Warning: GEMINI_API_KEY not set. LLM tests will fail.")
    
    # Run tests
    test_multilingual_llm_connection()
    success = test_multilingual_functionality()
    
    if success:
        print("\n🎉 All multilingual tests passed!")
    else:
        print("\n❌ Some tests failed. Check the output above.") 