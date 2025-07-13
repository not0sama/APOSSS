#!/usr/bin/env python3
"""
Comprehensive test runner for APOSSS unit tests
Executes all technique-specific unit tests and provides detailed reporting
"""
import sys
import os
import unittest
import logging
import time
from io import StringIO
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_results.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class DetailedTestResult(unittest.TestResult):
    """Custom test result class with detailed reporting"""
    
    def __init__(self):
        super().__init__()
        self.test_start_time = None
        self.test_times = {}
        self.test_details = []
        
    def startTest(self, test):
        super().startTest(test)
        self.test_start_time = time.time()
        
    def stopTest(self, test):
        super().stopTest(test)
        if self.test_start_time:
            duration = time.time() - self.test_start_time
            self.test_times[str(test)] = duration
            
    def addSuccess(self, test):
        super().addSuccess(test)
        self.test_details.append({
            'test': str(test),
            'status': 'PASSED',
            'duration': self.test_times.get(str(test), 0),
            'message': ''
        })
        
    def addError(self, test, err):
        super().addError(test, err)
        self.test_details.append({
            'test': str(test),
            'status': 'ERROR',
            'duration': self.test_times.get(str(test), 0),
            'message': self._exc_info_to_string(err, test)
        })
        
    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.test_details.append({
            'test': str(test),
            'status': 'FAILED',
            'duration': self.test_times.get(str(test), 0),
            'message': self._exc_info_to_string(err, test)
        })
        
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.test_details.append({
            'test': str(test),
            'status': 'SKIPPED',
            'duration': self.test_times.get(str(test), 0),
            'message': reason
        })


class APOSSSTestRunner:
    """Main test runner for APOSSS unit tests"""
    
    def __init__(self):
        self.test_modules = [
            'test_ranking_engine_unit',
            'test_ltr_ranker_unit',
            'test_knowledge_graph_unit',
            'test_embedding_ranker_unit',
            'test_enhanced_text_features_unit',
            'test_llm_processor_unit',
            'test_user_manager_unit',
            'test_search_engine_unit'
        ]
        
        self.results = {}
        self.total_start_time = None
        self.total_duration = 0
        
    def discover_tests(self, test_module_name):
        """Discover tests in a specific module"""
        try:
            # Import the test module
            test_module = __import__(test_module_name, fromlist=[''])
            
            # Create test suite
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(test_module)
            
            return suite
            
        except ImportError as e:
            logger.warning(f"Could not import test module {test_module_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error discovering tests in {test_module_name}: {e}")
            return None
    
    def run_test_module(self, test_module_name):
        """Run tests for a specific module"""
        logger.info(f"Running tests for {test_module_name}...")
        
        # Discover tests
        suite = self.discover_tests(test_module_name)
        if not suite:
            return None
        
        # Run tests with detailed results
        result = DetailedTestResult()
        start_time = time.time()
        
        try:
            suite.run(result)
            duration = time.time() - start_time
            
            # Collect results
            module_results = {
                'module': test_module_name,
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'skipped': len(result.skipped),
                'duration': duration,
                'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100 if result.testsRun > 0 else 0,
                'details': result.test_details
            }
            
            logger.info(f"✅ {test_module_name}: {result.testsRun} tests, {len(result.failures)} failures, {len(result.errors)} errors, {len(result.skipped)} skipped")
            
            return module_results
            
        except Exception as e:
            logger.error(f"Error running tests for {test_module_name}: {e}")
            return {
                'module': test_module_name,
                'tests_run': 0,
                'failures': 0,
                'errors': 1,
                'skipped': 0,
                'duration': 0,
                'success_rate': 0,
                'details': [{'test': 'module_load', 'status': 'ERROR', 'duration': 0, 'message': str(e)}]
            }
    
    def run_all_tests(self):
        """Run all unit tests"""
        logger.info("🚀 Starting APOSSS Unit Tests")
        logger.info("=" * 80)
        
        self.total_start_time = time.time()
        
        # Run tests for each module
        for module_name in self.test_modules:
            result = self.run_test_module(module_name)
            if result:
                self.results[module_name] = result
        
        self.total_duration = time.time() - self.total_start_time
        
        # Generate comprehensive report
        self.generate_report()
        
        return self.results
    
    def generate_report(self):
        """Generate comprehensive test report"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 APOSSS Unit Test Results Summary")
        logger.info("=" * 80)
        
        # Calculate totals
        total_tests = sum(result['tests_run'] for result in self.results.values())
        total_failures = sum(result['failures'] for result in self.results.values())
        total_errors = sum(result['errors'] for result in self.results.values())
        total_skipped = sum(result['skipped'] for result in self.results.values())
        total_passed = total_tests - total_failures - total_errors
        
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        # Overall statistics
        logger.info(f"📈 Overall Statistics:")
        logger.info(f"   Total Tests: {total_tests}")
        logger.info(f"   Passed: {total_passed}")
        logger.info(f"   Failed: {total_failures}")
        logger.info(f"   Errors: {total_errors}")
        logger.info(f"   Skipped: {total_skipped}")
        logger.info(f"   Success Rate: {overall_success_rate:.1f}%")
        logger.info(f"   Total Duration: {self.total_duration:.2f} seconds")
        
        # Module-by-module breakdown
        logger.info(f"\n📋 Module Breakdown:")
        for module_name, result in self.results.items():
            status_icon = "✅" if result['errors'] == 0 and result['failures'] == 0 else "❌"
            logger.info(f"   {status_icon} {module_name}: {result['tests_run']} tests, {result['success_rate']:.1f}% success, {result['duration']:.2f}s")
        
        # Detailed failures and errors
        if total_failures > 0 or total_errors > 0:
            logger.info(f"\n🔍 Detailed Failures and Errors:")
            for module_name, result in self.results.items():
                for detail in result['details']:
                    if detail['status'] in ['FAILED', 'ERROR']:
                        logger.info(f"   ❌ {detail['test']} ({detail['status']})")
                        if detail['message']:
                            # Show only first few lines of error message
                            error_lines = detail['message'].split('\n')[:3]
                            for line in error_lines:
                                logger.info(f"      {line}")
        
        # Performance analysis
        logger.info(f"\n⚡ Performance Analysis:")
        module_times = [(result['module'], result['duration']) for result in self.results.values()]
        module_times.sort(key=lambda x: x[1], reverse=True)
        
        for module_name, duration in module_times:
            logger.info(f"   {module_name}: {duration:.2f}s")
        
        # Generate technique coverage report
        self.generate_technique_coverage_report()
        
        # Save detailed report to file
        self.save_detailed_report()
        
        # Final summary
        logger.info("\n" + "=" * 80)
        if total_failures == 0 and total_errors == 0:
            logger.info("🎉 All tests passed successfully!")
        else:
            logger.info(f"⚠️  {total_failures + total_errors} tests failed or had errors")
        logger.info("=" * 80)
    
    def generate_technique_coverage_report(self):
        """Generate report on technique coverage"""
        logger.info(f"\n🔬 Technique Coverage Report:")
        
        technique_coverage = {
            'Ranking Engine': 'test_ranking_engine_unit' in self.results,
            'Learning-to-Rank (LTR)': 'test_ltr_ranker_unit' in self.results,
            'Knowledge Graph': 'test_knowledge_graph_unit' in self.results,
            'Embedding Ranker': 'test_embedding_ranker_unit' in self.results,
            'Enhanced Text Features': 'test_enhanced_text_features_unit' in self.results,
            'LLM Processor': 'test_llm_processor_unit' in self.results,
            'User Manager': 'test_user_manager_unit' in self.results,
            'Search Engine': 'test_search_engine_unit' in self.results
        }
        
        for technique, covered in technique_coverage.items():
            status_icon = "✅" if covered else "❌"
            logger.info(f"   {status_icon} {technique}")
        
        coverage_percentage = sum(technique_coverage.values()) / len(technique_coverage) * 100
        logger.info(f"   📊 Overall Coverage: {coverage_percentage:.1f}%")
    
    def save_detailed_report(self):
        """Save detailed test report to file"""
        report_filename = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        try:
            with open(report_filename, 'w') as f:
                f.write("APOSSS Unit Test Detailed Report\n")
                f.write("=" * 50 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Duration: {self.total_duration:.2f} seconds\n\n")
                
                # Write detailed results for each module
                for module_name, result in self.results.items():
                    f.write(f"\n{module_name}\n")
                    f.write("-" * len(module_name) + "\n")
                    f.write(f"Tests Run: {result['tests_run']}\n")
                    f.write(f"Failures: {result['failures']}\n")
                    f.write(f"Errors: {result['errors']}\n")
                    f.write(f"Skipped: {result['skipped']}\n")
                    f.write(f"Success Rate: {result['success_rate']:.1f}%\n")
                    f.write(f"Duration: {result['duration']:.2f}s\n")
                    
                    # Write individual test details
                    for detail in result['details']:
                        f.write(f"  {detail['status']}: {detail['test']} ({detail['duration']:.3f}s)\n")
                        if detail['message'] and detail['status'] in ['FAILED', 'ERROR']:
                            f.write(f"    Error: {detail['message'][:200]}...\n")
            
            logger.info(f"📄 Detailed report saved to: {report_filename}")
            
        except Exception as e:
            logger.error(f"Error saving detailed report: {e}")
    
    def run_specific_tests(self, test_names):
        """Run specific tests by name"""
        logger.info(f"🎯 Running specific tests: {', '.join(test_names)}")
        
        for test_name in test_names:
            if test_name in self.test_modules:
                result = self.run_test_module(test_name)
                if result:
                    self.results[test_name] = result
            else:
                logger.warning(f"Test module {test_name} not found")
        
        if self.results:
            self.generate_report()
        
        return self.results


def main():
    """Main entry point for test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='APOSSS Unit Test Runner')
    parser.add_argument('--modules', nargs='+', help='Specific test modules to run')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet output')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Create and run test runner
    runner = APOSSSTestRunner()
    
    try:
        if args.modules:
            # Run specific modules
            results = runner.run_specific_tests(args.modules)
        else:
            # Run all tests
            results = runner.run_all_tests()
        
        # Exit with appropriate code
        total_failures = sum(result['failures'] + result['errors'] for result in results.values())
        sys.exit(1 if total_failures > 0 else 0)
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Test run interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 