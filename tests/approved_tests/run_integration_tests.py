#!/usr/bin/env python3
"""
Integration Test Runner for APOSSS System
Provides comprehensive test execution and reporting for system integration tests
"""

import sys
import os
import time
import json
import logging
import argparse
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional
import unittest
from io import StringIO

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntegrationTestResult(unittest.TestResult):
    """Custom test result class for detailed integration test reporting"""
    
    def __init__(self, stream=None, descriptions=None, verbosity=None):
        super().__init__(stream, descriptions, verbosity)
        self.test_results = []
        self.performance_metrics = {}
        self.system_info = {}
        self.start_time = None
        self.end_time = None
        
    def startTest(self, test):
        """Called when a test is about to be run"""
        super().startTest(test)
        self.start_time = time.time()
        
    def stopTest(self, test):
        """Called when a test has been run"""
        super().stopTest(test)
        self.end_time = time.time()
        
        test_name = test._testMethodName
        duration = self.end_time - self.start_time
        
        # Determine test status
        if self.failures and self.failures[-1][0] == test:
            status = 'FAILED'
            error = self.failures[-1][1]
        elif self.errors and self.errors[-1][0] == test:
            status = 'ERROR'
            error = self.errors[-1][1]
        elif self.skipped and self.skipped[-1][0] == test:
            status = 'SKIPPED'
            error = self.skipped[-1][1]
        else:
            status = 'PASSED'
            error = None
            
        self.test_results.append({
            'name': test_name,
            'status': status,
            'duration': duration,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
        
        self.performance_metrics[test_name] = duration


class APOSSSIntegrationTestRunner:
    """Comprehensive test runner for APOSSS integration tests"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.test_report = {
            'timestamp': datetime.now().isoformat(),
            'base_url': base_url,
            'system_info': {},
            'test_results': [],
            'performance_metrics': {},
            'summary': {}
        }
        
    def check_system_availability(self) -> bool:
        """Check if the APOSSS system is available"""
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                self.test_report['system_info'] = health_data
                return health_data.get('status') == 'healthy'
        except Exception as e:
            logger.error(f"System not available: {e}")
            return False
        return False
    
    def run_tests(self, test_patterns: List[str] = None, verbosity: int = 2) -> bool:
        """Run integration tests with specified patterns"""
        
        print("🚀 APOSSS Integration Test Runner")
        print("=" * 80)
        
        # Check system availability
        print("🔍 Checking system availability...")
        if not self.check_system_availability():
            print("❌ System not available. Please ensure:")
            print("   1. Flask application is running (python app.py)")
            print("   2. All databases are connected")
            print("   3. Environment variables are configured")
            return False
        
        components = self.test_report['system_info'].get('components', {})
        healthy_count = sum(1 for status in components.values() if status)
        total_count = len(components)
        
        print(f"✅ System available: {healthy_count}/{total_count} components healthy")
        
        # Import test module
        try:
            from test_full_system_integration import APOSSSSystemIntegrationTest
        except ImportError as e:
            print(f"❌ Failed to import test module: {e}")
            return False
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        if test_patterns:
            # Load specific tests
            for pattern in test_patterns:
                try:
                    if pattern.startswith('test_'):
                        # Load specific test method
                        test_case = APOSSSSystemIntegrationTest(pattern)
                        suite.addTest(test_case)
                    else:
                        # Load tests matching pattern
                        partial_suite = loader.loadTestsFromName(
                            f"test_full_system_integration.APOSSSSystemIntegrationTest.{pattern}"
                        )
                        suite.addTest(partial_suite)
                except Exception as e:
                    print(f"⚠️ Warning: Could not load test pattern '{pattern}': {e}")
        else:
            # Load all tests
            suite = loader.loadTestsFromTestCase(APOSSSSystemIntegrationTest)
        
        # Run tests
        print(f"\n🧪 Running {suite.countTestCases()} integration tests...")
        print("=" * 80)
        
        # Custom test result
        result = IntegrationTestResult()
        start_time = time.time()
        
        # Run the tests
        suite.run(result)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Update test report
        self.test_report['test_results'] = result.test_results
        self.test_report['performance_metrics'] = result.performance_metrics
        self.test_report['summary'] = {
            'total_tests': result.testsRun,
            'passed': result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
            'failed': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped),
            'total_duration': total_duration,
            'success_rate': ((result.testsRun - len(result.failures) - len(result.errors)) / max(result.testsRun, 1)) * 100
        }
        
        # Print results
        self._print_test_results(result, total_duration)
        
        return result.wasSuccessful()
    
    def _print_test_results(self, result: IntegrationTestResult, total_duration: float):
        """Print comprehensive test results"""
        
        print("\n" + "=" * 80)
        print("📊 APOSSS Integration Test Results")
        print("=" * 80)
        
        # Overall summary
        summary = self.test_report['summary']
        print(f"\n🎯 Overall Summary:")
        print(f"   Total tests: {summary['total_tests']}")
        print(f"   Passed: {summary['passed']} ✅")
        print(f"   Failed: {summary['failed']} ❌")
        print(f"   Errors: {summary['errors']} 🔥")
        print(f"   Skipped: {summary['skipped']} ⏭️")
        print(f"   Success rate: {summary['success_rate']:.1f}%")
        print(f"   Total duration: {total_duration:.2f}s")
        
        # Test breakdown
        if result.test_results:
            print(f"\n📋 Test Breakdown:")
            for test_result in result.test_results:
                status_icon = {
                    'PASSED': '✅',
                    'FAILED': '❌',
                    'ERROR': '🔥',
                    'SKIPPED': '⏭️'
                }.get(test_result['status'], '❓')
                
                print(f"   {status_icon} {test_result['name']}: {test_result['duration']:.2f}s")
        
        # Performance metrics
        if result.performance_metrics:
            print(f"\n⚡ Performance Metrics:")
            sorted_metrics = sorted(result.performance_metrics.items(), key=lambda x: x[1], reverse=True)
            
            for test_name, duration in sorted_metrics:
                print(f"   {test_name}: {duration:.2f}s")
            
            avg_duration = sum(result.performance_metrics.values()) / len(result.performance_metrics)
            print(f"\n   Average per test: {avg_duration:.2f}s")
        
        # System health
        if self.test_report['system_info']:
            print(f"\n🏥 System Health:")
            components = self.test_report['system_info'].get('components', {})
            
            for component, status in components.items():
                status_icon = "✅" if status else "❌"
                print(f"   {status_icon} {component}")
        
        # Failure details
        if result.failures:
            print(f"\n❌ Failures ({len(result.failures)}):")
            for test, traceback in result.failures:
                print(f"   {test._testMethodName}:")
                print(f"   {traceback}")
        
        # Error details
        if result.errors:
            print(f"\n🔥 Errors ({len(result.errors)}):")
            for test, traceback in result.errors:
                print(f"   {test._testMethodName}:")
                print(f"   {traceback}")
        
        # Recommendations
        self._print_recommendations(result)
    
    def _print_recommendations(self, result: IntegrationTestResult):
        """Print recommendations based on test results"""
        
        print(f"\n💡 Recommendations:")
        
        # Performance recommendations
        if result.performance_metrics:
            slow_tests = [name for name, duration in result.performance_metrics.items() if duration > 10]
            if slow_tests:
                print(f"   🐌 Slow tests detected (>10s): {len(slow_tests)}")
                print(f"      Consider optimizing: {', '.join(slow_tests[:3])}")
        
        # System health recommendations
        if self.test_report['system_info']:
            components = self.test_report['system_info'].get('components', {})
            unhealthy = [name for name, status in components.items() if not status]
            
            if unhealthy:
                print(f"   🏥 Unhealthy components: {', '.join(unhealthy)}")
                print(f"      Check configuration and dependencies")
        
        # Test failure recommendations
        if result.failures or result.errors:
            print(f"   🔧 Test failures detected:")
            print(f"      1. Check system logs for detailed error information")
            print(f"      2. Verify all dependencies are installed")
            print(f"      3. Ensure API keys and configurations are correct")
            print(f"      4. Check database connectivity")
        
        # Success recommendations
        if result.wasSuccessful():
            print(f"   🎉 All tests passed! System integration is working correctly.")
            print(f"   📈 Consider running performance benchmarks for optimization")
    
    def save_report(self, filename: str = None):
        """Save test report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"integration_test_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.test_report, f, indent=2)
        
        print(f"📁 Test report saved to: {filename}")
    
    def run_specific_test_category(self, category: str) -> bool:
        """Run tests from specific category"""
        category_patterns = {
            'system': ['test_01_system_health', 'test_10_error_handling'],
            'user': ['test_02_user_management'],
            'search': ['test_03_llm_processing', 'test_04_search_engine', 'test_05_ranking_system'],
            'ai': ['test_06_embedding_and_knowledge_graph', 'test_07_ltr_system'],
            'feedback': ['test_08_feedback_system'],
            'performance': ['test_09_performance_and_load_testing'],
            'workflow': ['test_11_end_to_end_workflow']
        }
        
        if category not in category_patterns:
            print(f"❌ Unknown category: {category}")
            print(f"Available categories: {', '.join(category_patterns.keys())}")
            return False
        
        patterns = category_patterns[category]
        print(f"🎯 Running {category} tests: {', '.join(patterns)}")
        
        return self.run_tests(patterns)


def main():
    """Main entry point for the integration test runner"""
    
    parser = argparse.ArgumentParser(description='APOSSS Integration Test Runner')
    parser.add_argument('--base-url', default='http://localhost:5000',
                       help='Base URL for APOSSS system (default: http://localhost:5000)')
    parser.add_argument('--tests', nargs='*',
                       help='Specific test patterns to run (e.g., test_01_system_health)')
    parser.add_argument('--category', choices=['system', 'user', 'search', 'ai', 'feedback', 'performance', 'workflow'],
                       help='Run tests from specific category')
    parser.add_argument('--verbosity', type=int, default=2, choices=[0, 1, 2],
                       help='Test verbosity level (default: 2)')
    parser.add_argument('--save-report', action='store_true',
                       help='Save test report to file')
    parser.add_argument('--report-file', type=str,
                       help='Custom report filename')
    
    args = parser.parse_args()
    
    # Create test runner
    runner = APOSSSIntegrationTestRunner(base_url=args.base_url)
    
    # Run tests
    if args.category:
        success = runner.run_specific_test_category(args.category)
    else:
        success = runner.run_tests(test_patterns=args.tests, verbosity=args.verbosity)
    
    # Save report if requested
    if args.save_report:
        runner.save_report(args.report_file)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 