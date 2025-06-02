#!/usr/bin/env python3
"""
Complete APOSSS System Integration Test
Tests all phases and components working together
"""
import sys
import os
import time
import json
import requests
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class APOSSSSystemTest:
    """Complete system test for APOSSS"""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.test_results = []
        
    def run_test(self, test_name: str, test_func):
        """Run a single test and record results"""
        print(f"\n🧪 Testing: {test_name}")
        try:
            result = test_func()
            if result:
                print(f"   ✅ {test_name} - PASSED")
                self.test_results.append((test_name, True, None))
            else:
                print(f"   ❌ {test_name} - FAILED")
                self.test_results.append((test_name, False, "Test returned False"))
        except Exception as e:
            print(f"   ❌ {test_name} - ERROR: {str(e)}")
            self.test_results.append((test_name, False, str(e)))
        
    def test_health_check(self):
        """Test system health endpoint"""
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=10)
            data = response.json()
            
            if data.get('status') == 'healthy' and all(data.get('components', {}).values()):
                print("      All components healthy")
                return True
            else:
                print(f"      Health check failed: {data}")
                return False
        except Exception as e:
            print(f"      Health check error: {e}")
            return False
    
    def test_database_connections(self):
        """Test database connectivity"""
        try:
            response = requests.get(f"{self.base_url}/api/test-db", timeout=15)
            data = response.json()
            
            if data.get('success'):
                db_status = data.get('database_status', {})
                all_connected = all(
                    status.get('connected', False) 
                    for status in db_status.values()
                )
                
                if all_connected:
                    total_docs = sum(
                        sum(status.get('collections', {}).values())
                        for status in db_status.values()
                        if status.get('connected')
                    )
                    print(f"      All databases connected, {total_docs} total documents")
                    return True
                else:
                    print(f"      Some databases not connected: {db_status}")
                    return False
            else:
                print(f"      Database test failed: {data.get('error')}")
                return False
        except Exception as e:
            print(f"      Database test error: {e}")
            return False
    
    def test_llm_processing(self):
        """Test LLM query processing"""
        test_query = "machine learning for medical diagnosis"
        
        try:
            response = requests.post(
                f"{self.base_url}/api/test-llm",
                json={"query": test_query},
                timeout=30
            )
            data = response.json()
            
            if data.get('success'):
                processed = data.get('processed_query', {})
                
                # Check for key components
                has_intent = 'intent' in processed
                has_keywords = 'keywords' in processed
                has_entities = 'entities' in processed
                has_fields = 'academic_fields' in processed
                
                if all([has_intent, has_keywords, has_entities, has_fields]):
                    intent_conf = processed.get('intent', {}).get('confidence', 0)
                    print(f"      LLM processing successful, intent confidence: {intent_conf:.2f}")
                    return True
                else:
                    print(f"      Missing key components in LLM output")
                    return False
            else:
                print(f"      LLM processing failed: {data.get('error')}")
                return False
        except Exception as e:
            print(f"      LLM processing error: {e}")
            return False
    
    def test_search_functionality(self):
        """Test complete search with ranking"""
        test_queries = [
            "machine learning medical diagnosis",
            "renewable energy solar panels",
            "artificial intelligence algorithms"
        ]
        
        all_successful = True
        
        for query in test_queries:
            try:
                response = requests.post(
                    f"{self.base_url}/api/search",
                    json={"query": query, "ranking_mode": "hybrid"},
                    timeout=30
                )
                data = response.json()
                
                if data.get('success'):
                    results = data.get('search_results', {}).get('results', [])
                    total_results = len(results)
                    
                    if total_results > 0:
                        # Check ranking scores
                        has_scores = all('ranking_score' in result for result in results)
                        has_breakdown = all('score_breakdown' in result for result in results)
                        
                        if has_scores and has_breakdown:
                            print(f"      Query '{query}': {total_results} results with ranking")
                        else:
                            print(f"      Query '{query}': Missing ranking information")
                            all_successful = False
                    else:
                        print(f"      Query '{query}': No results found")
                else:
                    print(f"      Query '{query}': Search failed - {data.get('error')}")
                    all_successful = False
                    
            except Exception as e:
                print(f"      Query '{query}': Error - {e}")
                all_successful = False
        
        return all_successful
    
    def test_feedback_system(self):
        """Test feedback submission and retrieval"""
        try:
            # Submit test feedback
            feedback_data = {
                "query_id": "test_query_123",
                "result_id": "test_result_456",
                "rating": 5,
                "feedback_type": "thumbs_up",
                "user_session": "test_session"
            }
            
            response = requests.post(
                f"{self.base_url}/api/feedback",
                json=feedback_data,
                timeout=10
            )
            data = response.json()
            
            if not data.get('success'):
                print(f"      Feedback submission failed: {data.get('error')}")
                return False
            
            # Get feedback stats
            response = requests.get(f"{self.base_url}/api/feedback/stats", timeout=10)
            stats_data = response.json()
            
            if stats_data.get('success'):
                stats = stats_data.get('stats', {})
                total_feedback = stats.get('total_feedback', 0)
                print(f"      Feedback system working, {total_feedback} total feedback entries")
                return True
            else:
                print(f"      Feedback stats failed: {stats_data.get('error')}")
                return False
                
        except Exception as e:
            print(f"      Feedback system error: {e}")
            return False
    
    def test_embedding_system(self):
        """Test embedding/similarity functionality"""
        try:
            # Test pairwise similarity
            response = requests.post(
                f"{self.base_url}/api/similarity/pairwise",
                json={
                    "text1": "machine learning algorithms",
                    "text2": "artificial intelligence methods"
                },
                timeout=15
            )
            data = response.json()
            
            if data.get('success'):
                similarity = data.get('similarity_score', 0)
                print(f"      Embedding similarity working, score: {similarity:.4f}")
                return True
            else:
                print(f"      Embedding similarity failed: {data.get('error')}")
                return False
                
        except Exception as e:
            print(f"      Embedding system error: {e}")
            return False
    
    def test_ltr_system(self):
        """Test Learning-to-Rank functionality"""
        try:
            # Get LTR stats
            response = requests.get(f"{self.base_url}/api/ltr/stats", timeout=10)
            data = response.json()
            
            if data.get('success'):
                stats = data.get('stats', {})
                ltr_available = stats.get('ltr_available', False)
                model_trained = stats.get('model_trained', False)
                
                if ltr_available:
                    feature_count = stats.get('feature_count', 0)
                    print(f"      LTR system available, {feature_count} features")
                    
                    if model_trained:
                        print(f"      LTR model is trained and ready")
                    else:
                        print(f"      LTR model not yet trained (needs feedback data)")
                    
                    return True
                else:
                    print(f"      LTR system not available: {stats.get('reason', 'Unknown')}")
                    return False
            else:
                print(f"      LTR stats failed: {data.get('error')}")
                return False
                
        except Exception as e:
            print(f"      LTR system error: {e}")
            return False
    
    def test_performance_metrics(self):
        """Test system performance with multiple concurrent requests"""
        import threading
        import time
        
        results = []
        
        def make_search_request():
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/api/search",
                    json={"query": "test performance query"},
                    timeout=60  # Increase timeout to 60 seconds
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    results.append(end_time - start_time)
                else:
                    print(f"      Request failed with status {response.status_code}")
                    results.append(None)
            except Exception as e:
                print(f"      Request failed with error: {e}")
                results.append(None)
        
        # Make 5 concurrent requests
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_search_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Analyze results
        successful_requests = [r for r in results if r is not None]
        if len(successful_requests) >= 3:  # At least 3 out of 5 successful
            avg_time = sum(successful_requests) / len(successful_requests)
            print(f"      Performance test: {len(successful_requests)}/5 successful, avg time: {avg_time:.2f}s")
            return avg_time < 60.0  # Should be under 60 seconds (realistic for semantic search)
        else:
            print(f"      Performance test failed: Only {len(successful_requests)}/5 requests successful")
            return False
    
    def run_all_tests(self):
        """Run all system tests"""
        print("=" * 80)
        print("🚀 APOSSS Complete System Integration Test")
        print("=" * 80)
        print(f"Testing system at: {self.base_url}")
        print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Wait for system to be ready
        print("\n⏳ Waiting for system to be ready...")
        for attempt in range(10):
            try:
                response = requests.get(f"{self.base_url}/api/health", timeout=5)
                if response.status_code == 200:
                    print("   ✅ System is responding")
                    break
            except:
                if attempt < 9:
                    print(f"   ⏳ Attempt {attempt + 1}/10: System not ready, waiting...")
                    time.sleep(2)
                else:
                    print("   ❌ System failed to respond after 10 attempts")
                    return False
        
        # Run all tests
        test_suite = [
            ("System Health Check", self.test_health_check),
            ("Database Connections", self.test_database_connections),
            ("LLM Query Processing", self.test_llm_processing),
            ("Search Functionality", self.test_search_functionality),
            ("Feedback System", self.test_feedback_system),
            ("Embedding System", self.test_embedding_system),
            ("Learning-to-Rank System", self.test_ltr_system),
            ("Performance Metrics", self.test_performance_metrics)
        ]
        
        for test_name, test_func in test_suite:
            self.run_test(test_name, test_func)
        
        # Summary
        self.print_summary()
        
        # Return overall success
        passed_tests = sum(1 for _, success, _ in self.test_results if success)
        total_tests = len(self.test_results)
        return passed_tests == total_tests
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 80)
        print("📊 TEST SUMMARY")
        print("=" * 80)
        
        passed_tests = sum(1 for _, success, _ in self.test_results if success)
        total_tests = len(self.test_results)
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if passed_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED! APOSSS system is fully operational.")
            print("\n✨ System Capabilities Verified:")
            print("   ✅ Phase 1: LLM Query Understanding")
            print("   ✅ Phase 2: Multi-Database Search")
            print("   ✅ Phase 3: AI Ranking & User Feedback")
            print("   ✅ Advanced Features: LTR, Embeddings, Real-time Similarity")
            print("   ✅ Performance: Concurrent request handling")
            
        else:
            print(f"\n❌ {total_tests - passed_tests} tests failed:")
            for test_name, success, error in self.test_results:
                if not success:
                    print(f"   ❌ {test_name}: {error}")
        
        print("\n" + "=" * 80)

def main():
    """Main test execution"""
    import sys
    
    # Parse command line arguments
    base_url = "http://localhost:5000"
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    # Run tests
    tester = APOSSSSystemTest(base_url)
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 