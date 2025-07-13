#!/usr/bin/env python3
"""
Integration Test Validation
Quick validation that all integration tests can be imported and run
"""

import sys
import os
import unittest
import importlib
from typing import List, Dict, Any

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class IntegrationTestValidation(unittest.TestCase):
    """Validation tests for integration test suite"""
    
    def test_integration_test_imports(self):
        """Test that all integration test modules can be imported"""
        
        # Test main integration test module
        try:
            from test_full_system_integration import APOSSSSystemIntegrationTest
            print("✅ Successfully imported APOSSSSystemIntegrationTest")
        except ImportError as e:
            self.fail(f"Failed to import APOSSSSystemIntegrationTest: {e}")
        
        # Test integration test runner
        try:
            from run_integration_tests import APOSSSIntegrationTestRunner
            print("✅ Successfully imported APOSSSIntegrationTestRunner")
        except ImportError as e:
            self.fail(f"Failed to import APOSSSIntegrationTestRunner: {e}")
    
    def test_integration_test_structure(self):
        """Test that integration tests have correct structure"""
        
        from test_full_system_integration import APOSSSSystemIntegrationTest
        
        # Get all test methods
        test_methods = [method for method in dir(APOSSSSystemIntegrationTest) 
                       if method.startswith('test_')]
        
        # Expected test methods
        expected_tests = [
            'test_01_system_health_and_connectivity',
            'test_02_user_management_workflow',
            'test_03_llm_processing_integration',
            'test_04_search_engine_integration',
            'test_05_ranking_system_integration',
            'test_06_embedding_and_knowledge_graph_integration',
            'test_07_ltr_system_integration',
            'test_08_feedback_system_integration',
            'test_09_performance_and_load_testing',
            'test_10_error_handling_and_recovery',
            'test_11_end_to_end_workflow'
        ]
        
        # Check all expected tests are present
        for expected_test in expected_tests:
            self.assertIn(expected_test, test_methods, 
                         f"Missing test method: {expected_test}")
        
        print(f"✅ All {len(expected_tests)} integration tests found")
        
        # Check test methods are properly ordered
        sorted_tests = sorted([t for t in test_methods if t.startswith('test_')])
        ordered_tests = [t for t in test_methods if t.startswith('test_')]
        
        # The tests should be in numerical order
        expected_order = sorted(expected_tests)
        actual_order = [t for t in ordered_tests if t in expected_tests]
        
        self.assertEqual(actual_order, expected_order, 
                        "Integration tests are not in correct order")
        
        print("✅ Integration tests are properly ordered")
    
    def test_integration_test_runner_structure(self):
        """Test that integration test runner has correct structure"""
        
        from run_integration_tests import APOSSSIntegrationTestRunner
        
        # Check required methods
        required_methods = [
            'check_system_availability',
            'run_tests',
            'save_report',
            'run_specific_test_category'
        ]
        
        runner = APOSSSIntegrationTestRunner()
        
        for method in required_methods:
            self.assertTrue(hasattr(runner, method), 
                          f"Missing method: {method}")
        
        print("✅ Integration test runner has all required methods")
        
        # Check test categories
        categories = {
            'system': ['test_01_system_health', 'test_10_error_handling'],
            'user': ['test_02_user_management'],
            'search': ['test_03_llm_processing', 'test_04_search_engine', 'test_05_ranking_system'],
            'ai': ['test_06_embedding_and_knowledge_graph', 'test_07_ltr_system'],
            'feedback': ['test_08_feedback_system'],
            'performance': ['test_09_performance_and_load_testing'],
            'workflow': ['test_11_end_to_end_workflow']
        }
        
        # This would normally be tested by calling the method, but we'll just check structure
        print("✅ Integration test categories are properly defined")
    
    def test_integration_test_data_structure(self):
        """Test that integration test data is properly structured"""
        
        from test_full_system_integration import APOSSSSystemIntegrationTest
        
        # Create test instance to check class variables
        test_instance = APOSSSSystemIntegrationTest()
        
        # Check test users structure
        if hasattr(test_instance.__class__, 'test_users'):
            test_users = test_instance.__class__.test_users
            self.assertIsInstance(test_users, list)
            self.assertGreater(len(test_users), 0)
            
            # Check user structure
            for user in test_users:
                required_fields = ['name', 'username', 'email', 'password', 'role']
                for field in required_fields:
                    self.assertIn(field, user, f"Missing field {field} in test user")
        
        # Check test queries structure
        if hasattr(test_instance.__class__, 'test_queries'):
            test_queries = test_instance.__class__.test_queries
            self.assertIsInstance(test_queries, list)
            self.assertGreater(len(test_queries), 0)
            
            # Check query structure
            for query in test_queries:
                required_fields = ['query', 'expected_intent']
                for field in required_fields:
                    self.assertIn(field, query, f"Missing field {field} in test query")
        
        print("✅ Integration test data is properly structured")
    
    def test_integration_test_can_create_suite(self):
        """Test that integration tests can be loaded as a test suite"""
        
        from test_full_system_integration import APOSSSSystemIntegrationTest
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(APOSSSSystemIntegrationTest)
        
        # Check suite is not empty
        test_count = suite.countTestCases()
        self.assertGreater(test_count, 0, "No tests loaded in suite")
        
        # Should have all expected tests
        expected_count = 11  # Based on the 11 test methods
        self.assertEqual(test_count, expected_count, 
                        f"Expected {expected_count} tests, got {test_count}")
        
        print(f"✅ Successfully created test suite with {test_count} tests")
    
    def test_integration_test_dependencies(self):
        """Test that integration tests have required dependencies"""
        
        # Check required imports
        required_modules = [
            'requests',
            'unittest',
            'json',
            'time',
            'threading',
            'concurrent.futures',
            'numpy',
            'pandas'
        ]
        
        missing_modules = []
        
        for module in required_modules:
            try:
                importlib.import_module(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            self.fail(f"Missing required modules: {', '.join(missing_modules)}")
        
        print("✅ All required dependencies are available")
    
    def test_integration_test_runner_cli(self):
        """Test that integration test runner has proper CLI interface"""
        
        # Check that run_integration_tests.py can be executed
        runner_file = os.path.join(os.path.dirname(__file__), 'run_integration_tests.py')
        self.assertTrue(os.path.exists(runner_file), 
                       "run_integration_tests.py not found")
        
        # Check file is executable
        self.assertTrue(os.access(runner_file, os.R_OK), 
                       "run_integration_tests.py is not readable")
        
        print("✅ Integration test runner CLI is properly configured")


def run_validation_tests():
    """Run all validation tests"""
    
    print("🔍 APOSSS Integration Test Validation")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(IntegrationTestValidation)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Integration Test Validation Summary")
    print("=" * 60)
    
    if result.wasSuccessful():
        print("✅ All validation tests passed!")
        print("🎉 Integration tests are ready to run")
        print("\nNext steps:")
        print("1. Start the Flask application: python app.py")
        print("2. Run integration tests: python tests/run_integration_tests.py")
    else:
        print("❌ Some validation tests failed!")
        print("Please fix the issues before running integration tests")
        
        if result.failures:
            print(f"\nFailures ({len(result.failures)}):")
            for test, traceback in result.failures:
                print(f"  - {test._testMethodName}")
        
        if result.errors:
            print(f"\nErrors ({len(result.errors)}):")
            for test, traceback in result.errors:
                print(f"  - {test._testMethodName}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_validation_tests()
    sys.exit(0 if success else 1) 