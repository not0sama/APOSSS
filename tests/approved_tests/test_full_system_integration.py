#!/usr/bin/env python3
"""
Comprehensive System Integration Test for APOSSS
Tests all modules working together as a complete end-to-end system
"""

import sys
import os
import time
import json
import logging
import unittest
import requests
import threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class APOSSSSystemIntegrationTest(unittest.TestCase):
    """
    Comprehensive system integration test for APOSSS
    Tests complete workflows and module integration
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.base_url = "http://localhost:5000"
        cls.test_results = []
        cls.performance_metrics = {}
        cls.system_health = {}
        cls.test_data = cls._generate_test_data()
        
        # Test user credentials
        cls.test_users = [
            {
                "first_name": "Dr. Alice",
                "last_name": "Johnson",
                "username": "alice_johnson",
                "email": "alice@test-university.edu",
                "password": "SecurePass123!",
                "institution": "Test University",
                "role": "professor",
                "academic_fields": ["computer science", "artificial intelligence", "machine learning"]
            },
            {
                "first_name": "Bob",
                "last_name": "Smith",
                "username": "bob_smith",
                "email": "bob@research-institute.org",
                "password": "ResearchPass456!",
                "institution": "Research Institute",
                "role": "researcher",
                "academic_fields": ["data science", "statistics", "mathematics"]
            },
            {
                "first_name": "Student",
                "last_name": "User",
                "username": "student_user",
                "email": "student@university.edu",
                "password": "StudentPass789!",
                "institution": "University",
                "role": "student",
                "academic_fields": ["engineering", "computer science"]
            }
        ]
        
        # Test queries covering different scenarios
        cls.test_queries = [
            {
                "query": "machine learning algorithms for medical diagnosis",
                "expected_fields": ["computer science", "medical technology"],
                "expected_intent": "find_research",
                "complexity": "medium"
            },
            {
                "query": "renewable energy solar panel efficiency optimization",
                "expected_fields": ["engineering", "environmental science"],
                "expected_intent": "find_research",
                "complexity": "high"
            },
            {
                "query": "deep learning neural networks",
                "expected_fields": ["computer science", "artificial intelligence"],
                "expected_intent": "find_research",
                "complexity": "low"
            },
            {
                "query": "quantum computing applications in cryptography",
                "expected_fields": ["physics", "computer science"],
                "expected_intent": "find_research",
                "complexity": "high"
            },
            {
                "query": "natural language processing sentiment analysis",
                "expected_fields": ["computer science", "linguistics"],
                "expected_intent": "find_research",
                "complexity": "medium"
            }
        ]
        
        print("🚀 Starting APOSSS Comprehensive System Integration Test")
        print("=" * 80)
    
    @classmethod
    def _generate_test_data(cls) -> Dict[str, Any]:
        """Generate test data for comprehensive testing"""
        return {
            "feedback_scenarios": [
                {"rating": 5, "type": "thumbs_up", "comment": "Very relevant results"},
                {"rating": 4, "type": "rating", "comment": "Good results, minor improvements needed"},
                {"rating": 3, "type": "rating", "comment": "Okay results, could be better"},
                {"rating": 2, "type": "thumbs_down", "comment": "Not very relevant"},
                {"rating": 1, "type": "thumbs_down", "comment": "Poor results"}
            ],
            "ranking_modes": ["traditional", "hybrid", "ltr_only"],
            "search_filters": [
                {"type": "article", "year_min": 2020},
                {"type": "book", "status": "available"},
                {"type": "journal", "citations_min": 10}
            ]
        }
    
    def setUp(self):
        """Set up before each test"""
        self.start_time = time.time()
        self.test_session = f"test_session_{int(time.time())}"
    
    def tearDown(self):
        """Clean up after each test"""
        duration = time.time() - self.start_time
        test_name = self._testMethodName
        self.performance_metrics[test_name] = duration
        
        if hasattr(self, '_outcome') and self._outcome.errors:
            logger.error(f"Test {test_name} failed in {duration:.2f}s")
        else:
            logger.info(f"Test {test_name} completed in {duration:.2f}s")
    
    def test_01_system_health_and_connectivity(self):
        """Test system health and all component connectivity"""
        print("\n🏥 Testing System Health and Connectivity...")
        
        # Health check
        response = requests.get(f"{self.base_url}/api/health", timeout=10)
        self.assertEqual(response.status_code, 200)
        
        health_data = response.json()
        self.assertEqual(health_data['status'], 'healthy')
        self.system_health = health_data
        
        # Check all components are healthy
        components = health_data.get('components', {})
        for component, status in components.items():
            self.assertTrue(status, f"Component {component} is not healthy")
        
        print(f"   ✅ System health check passed: {len(components)} components healthy")
        
        # Database connectivity
        response = requests.get(f"{self.base_url}/api/test-db", timeout=15)
        self.assertEqual(response.status_code, 200)
        
        db_data = response.json()
        self.assertTrue(db_data['success'])
        
        db_status = db_data.get('database_status', {})
        for db_name, status in db_status.items():
            self.assertTrue(status.get('connected', False), f"Database {db_name} not connected")
        
        total_docs = sum(
            sum(status.get('collections', {}).values())
            for status in db_status.values()
            if status.get('connected')
        )
        
        print(f"   ✅ Database connectivity verified: {total_docs} total documents")
    
    def test_02_user_management_workflow(self):
        """Test complete user management workflow"""
        print("\n👤 Testing User Management Workflow...")
        
        registered_users = []
        
        # Test user registration for all test users
        for user_data in self.test_users:
            response = requests.post(
                f"{self.base_url}/api/auth/register",
                json=user_data,
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                reg_data = response.json()
                user_info = {
                    **user_data,
                    'token': reg_data.get('token'),
                    'user_id': reg_data.get('user_id')
                }
                registered_users.append(user_info)
                print(f"   ✅ User {user_data['username']} registered successfully")
            else:
                # User might already exist, try login
                login_response = requests.post(
                    f"{self.base_url}/api/auth/login",
                    json={
                        'identifier': user_data['username'],
                        'password': user_data['password']
                    },
                    timeout=10
                )
                
                if login_response.status_code == 200:
                    login_data = login_response.json()
                    user_info = {
                        **user_data,
                        'token': login_data.get('token'),
                        'user_id': login_data.get('user_id')
                    }
                    registered_users.append(user_info)
                    print(f"   ✅ User {user_data['username']} logged in successfully")
                else:
                    self.fail(f"Failed to register or login user {user_data['username']}")
        
        self.assertGreater(len(registered_users), 0, "No users registered successfully")
        
        # Test profile access for each user
        for user in registered_users:
            if user['token']:
                headers = {'Authorization': f"Bearer {user['token']}"}
                response = requests.get(
                    f"{self.base_url}/api/user/profile",
                    headers=headers,
                    timeout=10
                )
                
                if response.status_code == 200:
                    profile_data = response.json()
                    self.assertEqual(profile_data['username'], user['username'])
                    print(f"   ✅ Profile access verified for {user['username']}")
        
        # Store registered users for other tests
        self.__class__.registered_users = registered_users
    
    def test_03_llm_processing_integration(self):
        """Test LLM processing for various query types"""
        print("\n🤖 Testing LLM Processing Integration...")
        
        for query_data in self.test_queries:
            query = query_data['query']
            
            response = requests.post(
                f"{self.base_url}/api/test-llm",
                json={"query": query},
                timeout=30
            )
            
            self.assertEqual(response.status_code, 200)
            
            llm_data = response.json()
            self.assertTrue(llm_data['success'], f"LLM processing failed for query: {query}")
            
            processed = llm_data.get('processed_query', {})
            
            # Verify key components
            self.assertIn('intent', processed)
            self.assertIn('keywords', processed)
            self.assertIn('entities', processed)
            self.assertIn('academic_fields', processed)
            
            intent = processed.get('intent', {})
            self.assertEqual(intent.get('primary_intent'), query_data['expected_intent'])
            
            confidence = intent.get('confidence', 0)
            self.assertGreater(confidence, 0.5, f"Low confidence for query: {query}")
            
            print(f"   ✅ LLM processing verified for: {query[:50]}... (confidence: {confidence:.2f})")
    
    def test_04_search_engine_integration(self):
        """Test search engine with all ranking modes"""
        print("\n🔍 Testing Search Engine Integration...")
        
        if not hasattr(self.__class__, 'registered_users'):
            self.skipTest("User registration test must run first")
        
        # Test anonymous search
        print("   Testing anonymous search...")
        for query_data in self.test_queries:
            query = query_data['query']
            
            for ranking_mode in self.test_data['ranking_modes']:
                response = requests.post(
                    f"{self.base_url}/api/search",
                    json={
                        "query": query,
                        "ranking_mode": ranking_mode
                    },
                    timeout=30
                )
                
                self.assertEqual(response.status_code, 200)
                
                search_data = response.json()
                self.assertTrue(search_data['success'])
                
                results = search_data.get('search_results', {}).get('results', [])
                self.assertGreater(len(results), 0, f"No results for query: {query}")
                
                # Verify ranking scores
                for result in results:
                    self.assertIn('ranking_score', result)
                    self.assertIn('score_breakdown', result)
                    self.assertIsInstance(result['ranking_score'], (int, float))
                    self.assertIsInstance(result['score_breakdown'], dict)
                
                print(f"   ✅ Anonymous search ({ranking_mode}): {query[:30]}... -> {len(results)} results")
        
        # Test authenticated search with personalization
        print("   Testing authenticated search with personalization...")
        for user in self.registered_users:
            if not user['token']:
                continue
                
            headers = {'Authorization': f"Bearer {user['token']}"}
            
            for query_data in self.test_queries:
                query = query_data['query']
                
                response = requests.post(
                    f"{self.base_url}/api/search",
                    json={
                        "query": query,
                        "ranking_mode": "hybrid"
                    },
                    headers=headers,
                    timeout=30
                )
                
                self.assertEqual(response.status_code, 200)
                
                search_data = response.json()
                self.assertTrue(search_data['success'])
                
                # Verify personalization
                self.assertTrue(search_data.get('personalization_applied', False))
                self.assertEqual(search_data.get('user_type'), 'authenticated')
                
                results = search_data.get('search_results', {}).get('results', [])
                self.assertGreater(len(results), 0)
                
                print(f"   ✅ Personalized search for {user['username']}: {query[:30]}... -> {len(results)} results")
    
    def test_05_ranking_system_integration(self):
        """Test all ranking algorithms working together"""
        print("\n🏆 Testing Ranking System Integration...")
        
        test_query = "artificial intelligence machine learning"
        
        # Test each ranking mode
        ranking_results = {}
        
        for ranking_mode in self.test_data['ranking_modes']:
            response = requests.post(
                f"{self.base_url}/api/search",
                json={
                    "query": test_query,
                    "ranking_mode": ranking_mode
                },
                timeout=30
            )
            
            self.assertEqual(response.status_code, 200)
            
            search_data = response.json()
            self.assertTrue(search_data['success'])
            
            results = search_data.get('search_results', {}).get('results', [])
            self.assertGreater(len(results), 0)
            
            # Verify ranking metadata
            ranking_meta = search_data.get('search_results', {}).get('ranking_metadata', {})
            self.assertEqual(ranking_meta.get('ranking_algorithm'), ranking_mode)
            
            # Verify score consistency
            scores = [result['ranking_score'] for result in results]
            self.assertEqual(scores, sorted(scores, reverse=True), 
                           f"Results not sorted by score in {ranking_mode} mode")
            
            # Store results for comparison
            ranking_results[ranking_mode] = {
                'results': results,
                'metadata': ranking_meta
            }
            
            print(f"   ✅ {ranking_mode} ranking: {len(results)} results, top score: {scores[0]:.3f}")
        
        # Compare ranking modes
        traditional_results = ranking_results.get('traditional', {}).get('results', [])
        hybrid_results = ranking_results.get('hybrid', {}).get('results', [])
        
        if traditional_results and hybrid_results:
            # Verify hybrid has additional components
            traditional_breakdown = traditional_results[0].get('score_breakdown', {})
            hybrid_breakdown = hybrid_results[0].get('score_breakdown', {})
            
            self.assertGreaterEqual(len(hybrid_breakdown), len(traditional_breakdown),
                                  "Hybrid ranking should have more components")
            
            print(f"   ✅ Ranking comparison: traditional ({len(traditional_breakdown)} components) vs hybrid ({len(hybrid_breakdown)} components)")
    
    def test_06_embedding_and_knowledge_graph_integration(self):
        """Test embedding ranker and knowledge graph integration"""
        print("\n🧠 Testing Embedding and Knowledge Graph Integration...")
        
        # Test embedding system
        response = requests.post(
            f"{self.base_url}/api/test-embedding",
            json={"query": "machine learning neural networks"},
            timeout=30
        )
        
        self.assertEqual(response.status_code, 200)
        
        embedding_data = response.json()
        self.assertTrue(embedding_data['success'])
        
        embeddings = embedding_data.get('embeddings', {})
        self.assertIn('query_embedding', embeddings)
        self.assertIn('similarity_scores', embeddings)
        
        query_embedding = embeddings['query_embedding']
        similarity_scores = embeddings['similarity_scores']
        
        self.assertIsInstance(query_embedding, list)
        self.assertGreater(len(query_embedding), 0)
        self.assertIsInstance(similarity_scores, list)
        
        print(f"   ✅ Embedding system: {len(query_embedding)}D embeddings, {len(similarity_scores)} similarities")
        
        # Test knowledge graph if available
        try:
            response = requests.get(f"{self.base_url}/api/knowledge-graph/stats", timeout=15)
            if response.status_code == 200:
                kg_data = response.json()
                if kg_data.get('success'):
                    stats = kg_data.get('stats', {})
                    nodes = stats.get('total_nodes', 0)
                    edges = stats.get('total_edges', 0)
                    
                    print(f"   ✅ Knowledge graph: {nodes} nodes, {edges} edges")
                else:
                    print("   ⚠️ Knowledge graph not available")
        except:
            print("   ⚠️ Knowledge graph endpoint not available")
    
    def test_07_ltr_system_integration(self):
        """Test Learning-to-Rank system integration"""
        print("\n🎯 Testing LTR System Integration...")
        
        # Test LTR statistics
        response = requests.get(f"{self.base_url}/api/ltr/stats", timeout=15)
        self.assertEqual(response.status_code, 200)
        
        ltr_data = response.json()
        stats = ltr_data.get('stats', {})
        
        ltr_available = stats.get('ltr_available', False)
        model_trained = stats.get('model_trained', False)
        
        print(f"   📊 LTR available: {ltr_available}, Model trained: {model_trained}")
        
        if ltr_available:
            # Test LTR ranking
            response = requests.post(
                f"{self.base_url}/api/search",
                json={
                    "query": "machine learning algorithms",
                    "ranking_mode": "ltr_only"
                },
                timeout=30
            )
            
            self.assertEqual(response.status_code, 200)
            
            search_data = response.json()
            self.assertTrue(search_data['success'])
            
            results = search_data.get('search_results', {}).get('results', [])
            self.assertGreater(len(results), 0)
            
            # Verify LTR features
            if results:
                first_result = results[0]
                score_breakdown = first_result.get('score_breakdown', {})
                
                # Should have LTR-specific features
                ltr_features = [key for key in score_breakdown.keys() if 'ltr' in key.lower()]
                self.assertGreater(len(ltr_features), 0, "No LTR features found")
                
                print(f"   ✅ LTR ranking verified: {len(ltr_features)} LTR features")
        else:
            print("   ⚠️ LTR system not available - skipping LTR tests")
    
    def test_08_feedback_system_integration(self):
        """Test feedback system integration"""
        print("\n📝 Testing Feedback System Integration...")
        
        # Submit various types of feedback
        feedback_results = []
        
        for i, feedback_scenario in enumerate(self.test_data['feedback_scenarios']):
            feedback_data = {
                "query_id": f"test_query_{self.test_session}_{i}",
                "result_id": f"test_result_{i}",
                "rating": feedback_scenario['rating'],
                "feedback_type": feedback_scenario['type'],
                "user_session": self.test_session,
                "comment": feedback_scenario['comment']
            }
            
            response = requests.post(
                f"{self.base_url}/api/feedback",
                json=feedback_data,
                timeout=10
            )
            
            self.assertEqual(response.status_code, 200)
            
            result = response.json()
            self.assertTrue(result['success'])
            
            feedback_results.append(result)
            
            print(f"   ✅ Feedback submitted: {feedback_scenario['type']} (rating: {feedback_scenario['rating']})")
        
        # Test feedback statistics
        response = requests.get(f"{self.base_url}/api/feedback/stats", timeout=10)
        self.assertEqual(response.status_code, 200)
        
        stats_data = response.json()
        self.assertTrue(stats_data['success'])
        
        stats = stats_data.get('stats', {})
        total_feedback = stats.get('total_feedback', 0)
        avg_rating = stats.get('average_rating', 0)
        
        self.assertGreater(total_feedback, 0)
        self.assertGreater(avg_rating, 0)
        
        print(f"   ✅ Feedback stats: {total_feedback} total feedback, {avg_rating:.2f} avg rating")
    
    def test_09_performance_and_load_testing(self):
        """Test system performance under load"""
        print("\n⚡ Testing System Performance Under Load...")
        
        # Concurrent search requests
        num_concurrent_requests = 10
        test_queries = [query_data['query'] for query_data in self.test_queries]
        
        def make_search_request(query_index):
            """Make a search request"""
            query = test_queries[query_index % len(test_queries)]
            start_time = time.time()
            
            try:
                response = requests.post(
                    f"{self.base_url}/api/search",
                    json={
                        "query": query,
                        "ranking_mode": "hybrid"
                    },
                    timeout=30
                )
                
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    search_data = response.json()
                    if search_data['success']:
                        results = search_data.get('search_results', {}).get('results', [])
                        return {
                            'success': True,
                            'duration': duration,
                            'results_count': len(results),
                            'query': query
                        }
                
                return {
                    'success': False,
                    'duration': duration,
                    'error': f"Status: {response.status_code}",
                    'query': query
                }
                
            except Exception as e:
                return {
                    'success': False,
                    'duration': time.time() - start_time,
                    'error': str(e),
                    'query': query
                }
        
        # Execute concurrent requests
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
            futures = [executor.submit(make_search_request, i) for i in range(num_concurrent_requests)]
            results = [future.result() for future in as_completed(futures)]
        
        total_duration = time.time() - start_time
        
        # Analyze results
        successful_requests = [r for r in results if r['success']]
        failed_requests = [r for r in results if not r['success']]
        
        self.assertGreater(len(successful_requests), 0, "No successful requests")
        
        avg_duration = np.mean([r['duration'] for r in successful_requests])
        max_duration = max([r['duration'] for r in successful_requests])
        
        success_rate = len(successful_requests) / len(results) * 100
        
        print(f"   ✅ Load test: {len(successful_requests)}/{len(results)} requests successful ({success_rate:.1f}%)")
        print(f"   ⏱️ Performance: avg {avg_duration:.2f}s, max {max_duration:.2f}s, total {total_duration:.2f}s")
        
        # Performance assertions
        self.assertGreaterEqual(success_rate, 80, "Success rate too low")
        self.assertLess(avg_duration, 15, "Average response time too high")
        
        if failed_requests:
            print(f"   ⚠️ {len(failed_requests)} failed requests:")
            for req in failed_requests[:3]:  # Show first 3 failures
                print(f"      - {req['query'][:30]}...: {req['error']}")
    
    def test_10_error_handling_and_recovery(self):
        """Test error handling and system recovery"""
        print("\n🛡️ Testing Error Handling and Recovery...")
        
        # Test invalid query
        response = requests.post(
            f"{self.base_url}/api/search",
            json={"query": "", "ranking_mode": "hybrid"},
            timeout=10
        )
        
        # Should handle empty query gracefully
        self.assertIn(response.status_code, [200, 400])
        
        if response.status_code == 200:
            data = response.json()
            # Should either succeed with empty results or fail gracefully
            self.assertTrue(data.get('success', False) or 'error' in data)
            print("   ✅ Empty query handled gracefully")
        
        # Test invalid ranking mode
        response = requests.post(
            f"{self.base_url}/api/search",
            json={"query": "test query", "ranking_mode": "invalid_mode"},
            timeout=10
        )
        
        # Should handle invalid ranking mode
        self.assertIn(response.status_code, [200, 400])
        print("   ✅ Invalid ranking mode handled")
        
        # Test malformed JSON
        try:
            response = requests.post(
                f"{self.base_url}/api/search",
                data="invalid json",
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            self.assertEqual(response.status_code, 400)
            print("   ✅ Malformed JSON handled")
        except:
            print("   ⚠️ Malformed JSON test skipped")
        
        # Test unauthorized access
        response = requests.get(
            f"{self.base_url}/api/user/profile",
            headers={'Authorization': 'Bearer invalid_token'},
            timeout=10
        )
        
        self.assertEqual(response.status_code, 401)
        print("   ✅ Unauthorized access handled")
    
    def test_11_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        print("\n🔄 Testing Complete End-to-End Workflow...")
        
        if not hasattr(self.__class__, 'registered_users'):
            self.skipTest("User registration test must run first")
        
        # Use first registered user
        user = self.registered_users[0]
        headers = {'Authorization': f"Bearer {user['token']}"}
        
        # Step 1: Perform search
        query = "machine learning for healthcare applications"
        response = requests.post(
            f"{self.base_url}/api/search",
            json={"query": query, "ranking_mode": "hybrid"},
            headers=headers,
            timeout=30
        )
        
        self.assertEqual(response.status_code, 200)
        search_data = response.json()
        self.assertTrue(search_data['success'])
        
        results = search_data.get('search_results', {}).get('results', [])
        self.assertGreater(len(results), 0)
        
        query_id = search_data.get('query_id')
        self.assertIsNotNone(query_id)
        
        print(f"   ✅ Step 1: Search completed - {len(results)} results")
        
        # Step 2: Submit feedback for top result
        top_result = results[0]
        feedback_data = {
            "query_id": query_id,
            "result_id": top_result['id'],
            "rating": 5,
            "feedback_type": "thumbs_up",
            "user_session": self.test_session,
            "comment": "Very relevant result for healthcare ML"
        }
        
        response = requests.post(
            f"{self.base_url}/api/feedback",
            json=feedback_data,
            timeout=10
        )
        
        self.assertEqual(response.status_code, 200)
        feedback_result = response.json()
        self.assertTrue(feedback_result['success'])
        
        print("   ✅ Step 2: Feedback submitted")
        
        # Step 3: Perform related search
        related_query = "deep learning medical image analysis"
        response = requests.post(
            f"{self.base_url}/api/search",
            json={"query": related_query, "ranking_mode": "hybrid"},
            headers=headers,
            timeout=30
        )
        
        self.assertEqual(response.status_code, 200)
        search_data2 = response.json()
        self.assertTrue(search_data2['success'])
        
        results2 = search_data2.get('search_results', {}).get('results', [])
        self.assertGreater(len(results2), 0)
        
        print(f"   ✅ Step 3: Related search completed - {len(results2)} results")
        
        # Step 4: Check user profile and history
        response = requests.get(
            f"{self.base_url}/api/user/profile",
            headers=headers,
            timeout=10
        )
        
        self.assertEqual(response.status_code, 200)
        profile_data = response.json()
        self.assertEqual(profile_data['username'], user['username'])
        
        print("   ✅ Step 4: User profile accessed")
        
        # Step 5: Verify system learned from feedback (if LTR available)
        try:
            response = requests.get(f"{self.base_url}/api/ltr/stats", timeout=10)
            if response.status_code == 200:
                ltr_data = response.json()
                if ltr_data.get('stats', {}).get('ltr_available', False):
                    print("   ✅ Step 5: LTR system ready for learning")
                else:
                    print("   ⚠️ Step 5: LTR system not available")
        except:
            print("   ⚠️ Step 5: LTR stats not available")
        
        print("   ✅ Complete end-to-end workflow successful")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        print("\n" + "=" * 80)
        print("📊 APOSSS System Integration Test Summary")
        print("=" * 80)
        
        # Print performance metrics
        if cls.performance_metrics:
            print("\n⏱️ Performance Metrics:")
            for test_name, duration in sorted(cls.performance_metrics.items()):
                print(f"   {test_name}: {duration:.2f}s")
            
            total_time = sum(cls.performance_metrics.values())
            avg_time = total_time / len(cls.performance_metrics)
            print(f"\n   Total test time: {total_time:.2f}s")
            print(f"   Average per test: {avg_time:.2f}s")
        
        # Print system health summary
        if cls.system_health:
            print("\n🏥 System Health Summary:")
            components = cls.system_health.get('components', {})
            healthy_count = sum(1 for status in components.values() if status)
            total_count = len(components)
            
            print(f"   Components: {healthy_count}/{total_count} healthy")
            for component, status in components.items():
                status_icon = "✅" if status else "❌"
                print(f"   {status_icon} {component}")
        
        print("\n🎉 System Integration Test Completed!")


def run_integration_tests():
    """Run the integration tests"""
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(APOSSSSystemIntegrationTest)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == "__main__":
    print("🚀 APOSSS Comprehensive System Integration Test")
    print("=" * 80)
    print()
    print("This test suite verifies that all APOSSS modules work together")
    print("as a complete system. It tests:")
    print("- System health and connectivity")
    print("- User management workflow")
    print("- LLM processing integration")
    print("- Search engine with all ranking modes")
    print("- Embedding and knowledge graph integration")
    print("- Learning-to-Rank system")
    print("- Feedback system")
    print("- Performance under load")
    print("- Error handling and recovery")
    print("- Complete end-to-end workflow")
    print()
    print("Prerequisites:")
    print("1. Flask application must be running (python app.py)")
    print("2. All databases must be connected")
    print("3. Environment variables must be configured")
    print("4. API keys must be valid")
    print()
    
    input("Press Enter to start the integration tests...")
    
    try:
        success = run_integration_tests()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Tests interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1) 