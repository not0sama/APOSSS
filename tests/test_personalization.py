#!/usr/bin/env python3
"""
Test script for personalization integration in APOSSS
"""
import sys
import os
import json
from datetime import datetime

# Add current directory to path
sys.path.append('.')

def test_personalization():
    """Test personalization functionality"""
    print("🧪 Testing APOSSS Personalization Integration")
    print("=" * 50)
    
    try:
        # Import modules
        from modules.ranking_engine import RankingEngine
        from modules.user_manager import UserManager
        from modules.database_manager import DatabaseManager
        print("✅ Modules imported successfully")
        
        # Initialize components
        db_manager = DatabaseManager()
        user_manager = UserManager(db_manager)
        ranking_engine = RankingEngine(use_embedding=False, use_ltr=False)  # Disable complex features for testing
        print("✅ Components initialized successfully")
        
        # Test 1: Personalization data structure
        print("\n📊 Test 1: User Personalization Data Structure")
        sample_personalization_data = {
            'profile': {
                'name': 'Test User',
                'institution': 'Test University',
                'role': 'researcher',
                'academic_fields': ['computer science', 'artificial intelligence'],
                'languages': ['english', 'spanish']
            },
            'preferences': {
                'recency_preference': 0.8,
                'complexity_preference': 0.6,
                'availability_preference': 0.9,
                'result_type_preferences': {
                    'article': 0.9,
                    'book': 0.6,
                    'expert': 0.7
                }
            },
            'interaction_history': [
                {
                    'action': 'search',
                    'query': 'machine learning algorithms',
                    'timestamp': datetime.now().isoformat()
                },
                {
                    'action': 'feedback',
                    'metadata': {
                        'rating': 5,
                        'result_type': 'article'
                    },
                    'timestamp': datetime.now().isoformat()
                }
            ]
        }
        print(f"📄 Sample personalization data: {len(sample_personalization_data['interaction_history'])} interactions")
        
        # Test 2: Personalization scoring
        print("\n🎯 Test 2: Personalization Scoring")
        sample_results = [
            {
                'id': 'result1',
                'title': 'Machine Learning in Computer Science',
                'description': 'Advanced algorithms for AI systems',
                'author': 'Dr. Smith',
                'type': 'article',
                'metadata': {
                    'category': 'computer science',
                    'year': '2023',
                    'status': 'available',
                    'language': 'english'
                }
            },
            {
                'id': 'result2',
                'title': 'Basic Programming Concepts',
                'description': 'Introduction to programming',
                'author': 'Prof. Johnson',
                'type': 'book',
                'metadata': {
                    'category': 'programming',
                    'year': '2020',
                    'status': 'available',
                    'language': 'english'
                }
            },
            {
                'id': 'result3',
                'title': 'AI Expert Profile',
                'description': 'Expert in artificial intelligence',
                'author': 'Dr. AI Expert',
                'type': 'expert',
                'metadata': {
                    'category': 'artificial intelligence',
                    'institution': 'Test University',
                    'status': 'available'
                }
            }
        ]
        
        sample_query = {
            'keywords': {
                'primary': ['machine', 'learning'],
                'secondary': ['artificial', 'intelligence']
            },
            'intent': {
                'primary_intent': 'find_research',
                'confidence': 0.8
            }
        }
        
        # Calculate personalization scores
        personalization_scores = ranking_engine._calculate_personalization_scores(
            sample_results, sample_personalization_data, sample_query
        )
        
        print(f"📈 Personalization scores calculated: {len(personalization_scores)} results")
        for i, (result, score) in enumerate(zip(sample_results, personalization_scores)):
            print(f"   Result {i+1} ({result['type']}): {score:.3f}")
        
        # Test 3: Full ranking with personalization
        print("\n🏆 Test 3: Full Ranking with Personalization")
        search_results = {
            'results': sample_results,
            'total_results': len(sample_results)
        }
        
        ranked_results = ranking_engine.rank_search_results(
            search_results=search_results,
            processed_query=sample_query,
            user_feedback_data={},
            ranking_mode='traditional',
            user_personalization_data=sample_personalization_data
        )
        
        print(f"🎯 Ranking completed with {len(ranked_results['results'])} results")
        print("📋 Final rankings:")
        for i, result in enumerate(ranked_results['results']):
            score_breakdown = result.get('score_breakdown', {})
            personalization_score = score_breakdown.get('personalization_score', 0.0)
            ranking_score = result.get('ranking_score', 0.0)
            print(f"   Rank {i+1}: {result['title'][:40]}... (Total: {ranking_score:.3f}, Personal: {personalization_score:.3f})")
        
        # Test 4: User manager personalization methods
        print("\n👤 Test 4: User Manager Personalization Methods")
        
        # Test default preferences
        default_prefs = user_manager._generate_default_preferences()
        print(f"📋 Default preferences generated: {len(default_prefs)} items")
        
        # Test anonymous preferences
        sample_interactions = [
            {
                'action': 'search',
                'query': 'recent machine learning research',
                'timestamp': datetime.now().isoformat()
            },
            {
                'action': 'feedback',
                'metadata': {'rating': 5, 'result_type': 'article'},
                'timestamp': datetime.now().isoformat()
            }
        ]
        
        anon_prefs = user_manager._generate_anonymous_preferences(sample_interactions)
        print(f"🔍 Anonymous preferences generated from {len(sample_interactions)} interactions")
        
        # Test user stats calculation
        stats = user_manager._calculate_user_stats(sample_interactions)
        print(f"📊 User stats calculated: {stats['searches_count']} searches, {stats['feedback_count']} feedback")
        
        # Test 5: Integration test with metadata
        print("\n🔗 Test 5: Integration Test")
        metadata = ranked_results.get('ranking_metadata', {})
        personalization_enabled = metadata.get('personalization_enabled', False)
        print(f"✅ Personalization integration: {'Enabled' if personalization_enabled else 'Disabled'}")
        print(f"🔧 Ranking algorithm: {metadata.get('ranking_algorithm', 'Unknown')}")
        print(f"📊 Score components: {', '.join(metadata.get('score_components', []))}")
        
        print("\n🎉 All personalization tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_personalization()
    sys.exit(0 if success else 1) 