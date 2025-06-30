#!/usr/bin/env python3
"""
Test script for API integration with personalization
"""
import requests
import json
import time
from datetime import datetime

def test_api_integration():
    """Test API endpoints with personalization"""
    print("🌐 Testing APOSSS API Integration with Personalization")
    print("=" * 60)
    
    base_url = "http://localhost:5000"
    
    # Test data
    test_user = {
        "name": "Test User",
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpassword123",
        "institution": "Test University",
        "role": "researcher",
        "academic_fields": ["computer science", "artificial intelligence"]
    }
    
    test_query = "machine learning algorithms for research"
    
    try:
        print("🚀 Starting Flask application test...")
        
        # Test 1: Health Check
        print("\n💓 Test 1: Health Check")
        try:
            response = requests.get(f"{base_url}/api/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Health check passed: {health_data['status']}")
                print(f"📊 Components: {sum(health_data['components'].values())}/{len(health_data['components'])} working")
            else:
                print(f"⚠️ Health check returned status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Health check failed: {e}")
            print("💡 Make sure Flask app is running: python app.py")
            return False
        
        # Test 2: User Registration
        print("\n👤 Test 2: User Registration")
        try:
            response = requests.post(f"{base_url}/api/auth/register", 
                                   json=test_user, timeout=10)
            if response.status_code in [200, 201]:
                reg_data = response.json()
                print(f"✅ User registration: {reg_data.get('message', 'Success')}")
                auth_token = reg_data.get('token')
            else:
                print(f"⚠️ Registration status: {response.status_code}")
                auth_token = None
        except requests.exceptions.RequestException as e:
            print(f"❌ Registration failed: {e}")
            auth_token = None
        
        # Test 3: Search Without Authentication (Anonymous)
        print("\n🔍 Test 3: Anonymous Search")
        try:
            search_data = {
                "query": test_query,
                "ranking_mode": "traditional"
            }
            response = requests.post(f"{base_url}/api/search", 
                                   json=search_data, timeout=30)
            if response.status_code == 200:
                search_results = response.json()
                print(f"✅ Anonymous search successful")
                print(f"📊 Results: {len(search_results.get('search_results', {}).get('results', []))}")
                print(f"🎯 Personalization: {search_results.get('personalization_applied', False)}")
                print(f"👤 User type: {search_results.get('user_type', 'unknown')}")
                
                # Check ranking metadata
                ranking_meta = search_results.get('search_results', {}).get('ranking_metadata', {})
                print(f"🔧 Ranking algorithm: {ranking_meta.get('ranking_algorithm', 'Unknown')}")
                
            else:
                print(f"❌ Anonymous search failed: {response.status_code}")
                print(f"Response: {response.text[:200]}...")
        except requests.exceptions.RequestException as e:
            print(f"❌ Anonymous search error: {e}")
        
        # Test 4: Search With Authentication (if token available)
        if auth_token:
            print("\n🔐 Test 4: Authenticated Search")
            try:
                headers = {"Authorization": f"Bearer {auth_token}"}
                search_data = {
                    "query": test_query,
                    "ranking_mode": "hybrid"
                }
                response = requests.post(f"{base_url}/api/search", 
                                       json=search_data, headers=headers, timeout=30)
                if response.status_code == 200:
                    search_results = response.json()
                    print(f"✅ Authenticated search successful")
                    print(f"📊 Results: {len(search_results.get('search_results', {}).get('results', []))}")
                    print(f"🎯 Personalization: {search_results.get('personalization_applied', False)}")
                    print(f"👤 User type: {search_results.get('user_type', 'unknown')}")
                    
                    # Show score breakdown for first result
                    results = search_results.get('search_results', {}).get('results', [])
                    if results:
                        first_result = results[0]
                        score_breakdown = first_result.get('score_breakdown', {})
                        print(f"🏆 Top result: {first_result.get('title', 'Unknown')[:50]}...")
                        print(f"📈 Score breakdown:")
                        for component, score in score_breakdown.items():
                            print(f"   - {component}: {score:.3f}")
                    
                else:
                    print(f"❌ Authenticated search failed: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"❌ Authenticated search error: {e}")
        
        # Test 5: Feedback Submission
        print("\n📝 Test 5: Feedback Submission")
        try:
            feedback_data = {
                "query_id": f"test_query_{int(time.time())}",
                "result_id": "test_result_1",
                "rating": 5,
                "feedback_type": "rating",
                "user_session": "test_session",
                "additional_data": {"test": True}
            }
            response = requests.post(f"{base_url}/api/feedback", 
                                   json=feedback_data, timeout=10)
            if response.status_code == 200:
                feedback_result = response.json()
                print(f"✅ Feedback submission: {feedback_result.get('message', 'Success')}")
            else:
                print(f"❌ Feedback submission failed: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Feedback submission error: {e}")
        
        # Test 6: Feedback Stats
        print("\n📊 Test 6: Feedback Statistics")
        try:
            response = requests.get(f"{base_url}/api/feedback/stats", timeout=10)
            if response.status_code == 200:
                stats = response.json().get('stats', {})
                print(f"✅ Feedback stats retrieved")
                print(f"📈 Total feedback: {stats.get('total_feedback', 0)}")
                print(f"⭐ Average rating: {stats.get('average_rating', 0)}")
                print(f"💾 Storage type: {stats.get('storage_type', 'unknown')}")
            else:
                print(f"❌ Feedback stats failed: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Feedback stats error: {e}")
        
        # Test 7: LTR Statistics
        print("\n🤖 Test 7: LTR Model Statistics")
        try:
            response = requests.get(f"{base_url}/api/ltr/stats", timeout=10)
            if response.status_code == 200:
                ltr_stats = response.json().get('stats', {})
                print(f"✅ LTR stats retrieved")
                print(f"🔧 LTR available: {ltr_stats.get('ltr_available', False)}")
                print(f"📚 Model trained: {ltr_stats.get('model_trained', False)}")
                print(f"🔢 Feature count: {ltr_stats.get('feature_count', 0)}")
            else:
                print(f"❌ LTR stats failed: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ LTR stats error: {e}")
        
        print("\n🎉 API integration tests completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ API test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_test_instructions():
    """Print instructions for running API tests"""
    print("📋 API Test Instructions:")
    print("1. Start the Flask application: python app.py")
    print("2. Run this test script: python test_api_integration.py")
    print("3. Check that all tests pass")
    print("4. Note: Some tests may fail if databases are not connected")
    print()

if __name__ == "__main__":
    print_test_instructions()
    success = test_api_integration()
    exit(0 if success else 1) 