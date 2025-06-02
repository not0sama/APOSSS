#!/usr/bin/env python3
"""
User Management System Test for APOSSS
Tests user registration, authentication, profile management, and interaction tracking
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

class UserSystemTest:
    """Test suite for user management system"""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.test_results = []
        self.test_user_data = {
            'first_name': 'John',
            'last_name': 'Doe',
            'email': 'john.doe.test@example.com',
            'username': 'johndoe_test',
            'password': 'TestPassword123!',
            'organization': 'Test University',
            'department': 'Computer Science',
            'role': 'Graduate Student',
            'country': 'United States',
            'research_interests': ['AI', 'Machine Learning', 'Natural Language Processing'],
            'phone': '+1-555-0123',
            'website': 'https://johndoe.example.com'
        }
        self.auth_token = None
        
    def run_test(self, test_name: str, test_func):
        """Run a single test and record results"""
        print(f"\n🧪 Testing: {test_name}")
        try:
            result = test_func()
            if result:
                print(f"   ✅ PASSED")
                self.test_results.append({'test': test_name, 'status': 'PASSED'})
                return True
            else:
                print(f"   ❌ FAILED")
                self.test_results.append({'test': test_name, 'status': 'FAILED'})
                return False
        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}")
            self.test_results.append({'test': test_name, 'status': 'ERROR', 'error': str(e)})
            return False
    
    def test_health_check(self):
        """Test system health check"""
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=10)
            data = response.json()
            
            if response.status_code == 200:
                components = data.get('components', {})
                print(f"      Health Status: {data.get('status')}")
                print(f"      User Manager: {'✅' if components.get('user_manager') else '❌'}")
                return components.get('user_manager', False)
            return False
            
        except Exception as e:
            print(f"      Health check failed: {e}")
            return False
    
    def test_user_registration(self):
        """Test user registration"""
        try:
            # First, clean up any existing test user
            self.cleanup_test_user()
            
            response = requests.post(
                f"{self.base_url}/api/auth/register",
                json=self.test_user_data,
                timeout=30
            )
            
            if response.status_code == 201:
                data = response.json()
                print(f"      Registration successful: {data.get('message')}")
                print(f"      User ID: {data.get('user_id')}")
                return data.get('success', False)
            else:
                print(f"      Registration failed: {response.status_code}")
                if response.headers.get('content-type', '').startswith('application/json'):
                    error_data = response.json()
                    print(f"      Error: {error_data.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"      Registration test failed: {e}")
            return False
    
    def test_user_login(self):
        """Test user authentication"""
        try:
            login_data = {
                'login_identifier': self.test_user_data['username'],
                'password': self.test_user_data['password']
            }
            
            response = requests.post(
                f"{self.base_url}/api/auth/login",
                json=login_data,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    self.auth_token = data.get('token')
                    user_info = data.get('user', {})
                    print(f"      Login successful for: {user_info.get('username')}")
                    print(f"      Token received: {self.auth_token[:20]}...")
                    return True
            
            print(f"      Login failed: {response.status_code}")
            if response.headers.get('content-type', '').startswith('application/json'):
                error_data = response.json()
                print(f"      Error: {error_data.get('error', 'Unknown error')}")
            return False
            
        except Exception as e:
            print(f"      Login test failed: {e}")
            return False
    
    def test_token_verification(self):
        """Test JWT token verification"""
        if not self.auth_token:
            print("      No auth token available for verification")
            return False
        
        try:
            headers = {'Authorization': f'Bearer {self.auth_token}'}
            response = requests.post(
                f"{self.base_url}/api/auth/verify",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    user_info = data.get('user', {})
                    print(f"      Token verified for: {user_info.get('username')}")
                    return True
            
            print(f"      Token verification failed: {response.status_code}")
            return False
            
        except Exception as e:
            print(f"      Token verification test failed: {e}")
            return False
    
    def test_user_profile(self):
        """Test user profile retrieval"""
        if not self.auth_token:
            print("      No auth token available for profile test")
            return False
        
        try:
            headers = {'Authorization': f'Bearer {self.auth_token}'}
            response = requests.get(
                f"{self.base_url}/api/user/profile",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    user = data.get('user', {})
                    print(f"      Profile retrieved for: {user.get('first_name')} {user.get('last_name')}")
                    print(f"      Organization: {user.get('organization')}")
                    print(f"      Research interests: {len(user.get('research_interests', []))}")
                    return True
            
            print(f"      Profile retrieval failed: {response.status_code}")
            return False
            
        except Exception as e:
            print(f"      Profile test failed: {e}")
            return False
    
    def test_search_with_user_context(self):
        """Test search functionality with user authentication"""
        if not self.auth_token:
            print("      No auth token available for search test")
            return False
        
        try:
            headers = {
                'Authorization': f'Bearer {self.auth_token}',
                'Content-Type': 'application/json'
            }
            
            search_data = {
                'query': 'machine learning algorithms',
                'ranking_mode': 'hybrid',
                'session_id': f'test_session_{int(time.time())}'
            }
            
            response = requests.post(
                f"{self.base_url}/api/search",
                headers=headers,
                json=search_data,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    search_results = data.get('search_results', {})
                    print(f"      Search completed with {search_results.get('total_results', 0)} results")
                    print(f"      Personalized: {data.get('personalized', False)}")
                    print(f"      User logged in: {data.get('user_logged_in', False)}")
                    return True
            
            print(f"      Search test failed: {response.status_code}")
            return False
            
        except Exception as e:
            print(f"      Search with user context test failed: {e}")
            return False
    
    def test_feedback_with_user_context(self):
        """Test feedback submission with user authentication"""
        if not self.auth_token:
            print("      No auth token available for feedback test")
            return False
        
        try:
            headers = {
                'Authorization': f'Bearer {self.auth_token}',
                'Content-Type': 'application/json'
            }
            
            feedback_data = {
                'query_id': f'test_query_{int(time.time())}',
                'result_id': f'test_result_{int(time.time())}',
                'rating': 5,
                'feedback_type': 'relevance',
                'session_id': f'test_session_{int(time.time())}'
            }
            
            response = requests.post(
                f"{self.base_url}/api/feedback",
                headers=headers,
                json=feedback_data,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    print(f"      Feedback submitted: {data.get('feedback_id')}")
                    print(f"      Points earned: {data.get('points_earned', 0)}")
                    print(f"      User logged in: {data.get('user_logged_in', False)}")
                    return True
            
            print(f"      Feedback test failed: {response.status_code}")
            return False
            
        except Exception as e:
            print(f"      Feedback with user context test failed: {e}")
            return False
    
    def test_user_recommendations(self):
        """Test personalized recommendations"""
        if not self.auth_token:
            print("      No auth token available for recommendations test")
            return False
        
        try:
            headers = {'Authorization': f'Bearer {self.auth_token}'}
            response = requests.get(
                f"{self.base_url}/api/user/recommendations",
                headers=headers,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    recommendations = data.get('recommendations', {})
                    print(f"      Recommendations generated successfully")
                    print(f"      Suggested topics: {len(recommendations.get('suggested_topics', []))}")
                    print(f"      Research interests: {len(recommendations.get('research_interests', []))}")
                    return True
            
            print(f"      Recommendations test failed: {response.status_code}")
            return False
            
        except Exception as e:
            print(f"      Recommendations test failed: {e}")
            return False
    
    def test_user_analytics(self):
        """Test user analytics"""
        if not self.auth_token:
            print("      No auth token available for analytics test")
            return False
        
        try:
            headers = {'Authorization': f'Bearer {self.auth_token}'}
            response = requests.get(
                f"{self.base_url}/api/user/analytics",
                headers=headers,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    analytics = data.get('analytics', {})
                    print(f"      Analytics retrieved successfully")
                    print(f"      Total interactions: {analytics.get('total_interactions', 0)}")
                    print(f"      Search count: {analytics.get('search_count', 0)}")
                    print(f"      Feedback count: {analytics.get('feedback_count', 0)}")
                    return True
            
            print(f"      Analytics test failed: {response.status_code}")
            return False
            
        except Exception as e:
            print(f"      Analytics test failed: {e}")
            return False
    
    def test_logout(self):
        """Test user logout"""
        if not self.auth_token:
            print("      No auth token available for logout test")
            return False
        
        try:
            headers = {'Authorization': f'Bearer {self.auth_token}'}
            response = requests.post(
                f"{self.base_url}/api/auth/logout",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"      Logout successful: {data.get('message')}")
                return data.get('success', False)
            
            print(f"      Logout failed: {response.status_code}")
            return False
            
        except Exception as e:
            print(f"      Logout test failed: {e}")
            return False
    
    def cleanup_test_user(self):
        """Clean up test user from database"""
        try:
            # This would normally connect to the database and remove the test user
            # For now, we'll just skip if the user already exists
            pass
        except Exception as e:
            print(f"      Cleanup warning: {e}")
    
    def run_all_tests(self):
        """Run all user system tests"""
        print("🚀 Starting APOSSS User Management System Tests")
        print("=" * 60)
        
        tests = [
            ("System Health Check", self.test_health_check),
            ("User Registration", self.test_user_registration),
            ("User Login", self.test_user_login),
            ("Token Verification", self.test_token_verification),
            ("User Profile Retrieval", self.test_user_profile),
            ("Search with User Context", self.test_search_with_user_context),
            ("Feedback with User Context", self.test_feedback_with_user_context),
            ("User Recommendations", self.test_user_recommendations),
            ("User Analytics", self.test_user_analytics),
            ("User Logout", self.test_logout)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            if self.run_test(test_name, test_func):
                passed_tests += 1
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        for result in self.test_results:
            status_icon = "✅" if result['status'] == 'PASSED' else "❌"
            print(f"{status_icon} {result['test']}: {result['status']}")
            if 'error' in result:
                print(f"    Error: {result['error']}")
        
        success_rate = (passed_tests / total_tests) * 100
        print(f"\n🎯 Success Rate: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        
        if success_rate >= 80:
            print("🎉 User Management System: OPERATIONAL")
        elif success_rate >= 60:
            print("⚠️  User Management System: PARTIALLY FUNCTIONAL")
        else:
            print("❌ User Management System: NEEDS ATTENTION")
        
        return success_rate

if __name__ == "__main__":
    print("Starting APOSSS User Management System Tests...")
    print("Make sure the Flask application is running on http://localhost:5000")
    print()
    
    # Wait a moment for any startup
    time.sleep(2)
    
    # Run tests
    tester = UserSystemTest()
    success_rate = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success_rate >= 80 else 1) 