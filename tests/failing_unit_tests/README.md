# APOSSS Unit Testing Framework

This directory contains comprehensive unit tests for the APOSSS (Academic Papers and Other Scholarly Sources Search) system. The testing framework is designed to validate the functionality of all core techniques and components implemented in the system.

## 📋 Overview

The APOSSS unit testing framework provides comprehensive coverage for all major techniques implemented in the system, including:

- **Ranking Engine** - Multiple ranking algorithms and hybrid approaches
- **Learning-to-Rank (LTR)** - XGBoost-based ranking with feature engineering
- **Knowledge Graph** - Graph-based relationships and scoring
- **Embedding Ranker** - Semantic similarity and FAISS indexing
- **Enhanced Text Features** - Advanced text analysis and BM25 scoring
- **LLM Processor** - Query analysis and understanding using Gemini
- **User Manager** - Authentication and user management
- **Search Engine** - Hybrid search and result merging

## 🧪 Test Structure

### Unit Test Files

| Test File | Component | Description |
|-----------|-----------|-------------|
| `test_ranking_engine_unit.py` | RankingEngine | Tests heuristic, TF-IDF, intent, embedding, and hybrid ranking algorithms |
| `test_ltr_ranker_unit.py` | LTRRanker | Tests feature extraction, model training, and ranking predictions |
| `test_knowledge_graph_unit.py` | KnowledgeGraph | Tests graph construction, PageRank, authority scores, and relationships |
| `test_embedding_ranker_unit.py` | EmbeddingRanker | Tests embeddings, FAISS indexing, and similarity search |
| `test_enhanced_text_features_unit.py` | EnhancedTextFeatures | Tests BM25, n-gram features, and text complexity analysis |
| `test_llm_processor_unit.py` | LLMProcessor | Tests query analysis, keyword extraction, and semantic expansion |
| `test_user_manager_unit.py` | UserManager | Tests authentication, user profiles, and JWT token management |
| `test_search_engine_unit.py` | SearchEngine | Tests hybrid search, traditional search, and result merging |

### Test Runner

- `run_unit_tests.py` - Comprehensive test runner with detailed reporting and analysis

## 🚀 Getting Started

### Prerequisites

Before running the tests, ensure you have the following dependencies installed:

```bash
# Core testing dependencies
pip install unittest2 mock

# Optional dependencies for full functionality
pip install numpy pandas scikit-learn
pip install xgboost  # For LTR tests
pip install faiss-cpu  # For embedding tests
pip install sentence-transformers  # For embedding tests
pip install networkx  # For knowledge graph tests
pip install nltk textstat rank-bm25  # For text features tests
pip install google-generativeai  # For LLM processor tests (requires API key)
```

### Environment Setup

Some tests require environment variables:

```bash
# For LLM Processor tests
export GEMINI_API_KEY="your_gemini_api_key"

# For database tests (optional)
export MONGODB_URI_ACADEMIC_LIBRARY="mongodb://localhost:27017/Academic_Library"
export MONGODB_URI_EXPERTS_SYSTEM="mongodb://localhost:27017/Experts_System"
# ... other database URIs
```

## 🏃 Running Tests

### Run All Tests

```bash
# Run all unit tests
python tests/run_unit_tests.py

# Run with verbose output
python tests/run_unit_tests.py --verbose

# Run with quiet output
python tests/run_unit_tests.py --quiet
```

### Run Specific Test Modules

```bash
# Run specific test modules
python tests/run_unit_tests.py --modules test_ranking_engine_unit test_ltr_ranker_unit

# Run a single test module
python tests/run_unit_tests.py --modules test_embedding_ranker_unit
```

### Run Individual Test Files

```bash
# Run individual test files directly
python -m unittest tests.test_ranking_engine_unit
python -m unittest tests.test_ltr_ranker_unit -v
```

### Run Specific Test Cases

```bash
# Run specific test cases
python -m unittest tests.test_ranking_engine_unit.TestRankingEngine.test_heuristic_scoring
python -m unittest tests.test_ltr_ranker_unit.TestLTRRanker.test_feature_extraction
```

## 📊 Test Reports

The test runner generates comprehensive reports including:

### Console Output
- ✅ Real-time test progress
- 📈 Overall statistics (pass/fail/error/skip rates)
- 📋 Module-by-module breakdown
- 🔍 Detailed failure and error information
- ⚡ Performance analysis
- 🔬 Technique coverage report

### Generated Files
- `test_results.log` - Detailed test execution log
- `test_report_YYYYMMDD_HHMMSS.txt` - Comprehensive test report

### Sample Output

```
🚀 Starting APOSSS Unit Tests
================================================================================
Running tests for test_ranking_engine_unit...
✅ test_ranking_engine_unit: 15 tests, 0 failures, 0 errors, 2 skipped
Running tests for test_ltr_ranker_unit...
✅ test_ltr_ranker_unit: 12 tests, 0 failures, 0 errors, 1 skipped
...

================================================================================
📊 APOSSS Unit Test Results Summary
================================================================================
📈 Overall Statistics:
   Total Tests: 89
   Passed: 82
   Failed: 0
   Errors: 0
   Skipped: 7
   Success Rate: 92.1%
   Total Duration: 45.67 seconds

🔬 Technique Coverage Report:
   ✅ Ranking Engine
   ✅ Learning-to-Rank (LTR)
   ✅ Knowledge Graph
   ✅ Embedding Ranker
   ✅ Enhanced Text Features
   ✅ LLM Processor
   ✅ User Manager
   ✅ Search Engine
   📊 Overall Coverage: 100.0%

🎉 All tests passed successfully!
```

## 🔧 Test Categories

### Unit Tests
- **Initialization Tests** - Test component initialization and configuration
- **Functional Tests** - Test core functionality and algorithms
- **Edge Case Tests** - Test boundary conditions and error handling
- **Performance Tests** - Test performance characteristics and scalability
- **Integration Tests** - Test interaction with real external dependencies

### Mocking Strategy
- **External APIs** - Mock LLM APIs, database connections, and external services
- **Heavy Dependencies** - Mock large ML models, FAISS indices, and computation-heavy operations
- **File System** - Mock file operations and temporary directories
- **Network** - Mock network requests and responses

## 🛠️ Test Architecture

### Test Design Principles
1. **Isolated Testing** - Each test is independent and doesn't affect others
2. **Comprehensive Coverage** - Tests cover all major code paths and edge cases
3. **Realistic Scenarios** - Tests use realistic data and scenarios
4. **Performance Aware** - Tests are designed to run efficiently
5. **Maintainable** - Tests are well-organized and documented

### Mock Usage
```python
# Example: Testing with mocked dependencies
@patch('modules.embedding_ranker.SentenceTransformer')
@patch('modules.embedding_ranker.faiss')
def test_embedding_ranker_initialization(self, mock_faiss, mock_sentence_transformer):
    # Mock setup
    mock_model = Mock()
    mock_sentence_transformer.return_value = mock_model
    
    # Test execution
    ranker = EmbeddingRanker(cache_dir=self.temp_dir)
    
    # Assertions
    self.assertIsNotNone(ranker)
    self.assertEqual(ranker.model, mock_model)
```

## 📈 Continuous Integration

### GitHub Actions Integration
The tests are designed to work with CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
name: Unit Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r test-requirements.txt
      - name: Run unit tests
        run: python tests/run_unit_tests.py
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
```

## 🐛 Debugging Tests

### Common Issues and Solutions

1. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python path configuration
   - Verify module structure

2. **Mock Failures**
   - Check mock patch decorators
   - Verify mock configuration
   - Ensure correct mock assertions

3. **Environment Issues**
   - Set required environment variables
   - Check file permissions
   - Verify temporary directory creation

### Debug Mode
```bash
# Run with debug logging
python tests/run_unit_tests.py --verbose

# Run specific failing test with debug
python -m unittest tests.test_module.TestClass.test_method -v
```

## 📚 Adding New Tests

### Test Template
```python
#!/usr/bin/env python3
"""
Unit tests for APOSSS NewComponent class
"""
import sys
import os
import unittest
from unittest.mock import Mock, patch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestNewComponent(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        pass
    
    def test_component_initialization(self):
        """Test component initialization"""
        pass
    
    def test_core_functionality(self):
        """Test core functionality"""
        pass

if __name__ == '__main__':
    unittest.main(verbosity=2)
```

### Best Practices
1. **Descriptive Test Names** - Use clear, descriptive test method names
2. **Comprehensive Setup** - Use setUp() for common test data
3. **Isolated Tests** - Each test should be independent
4. **Clear Assertions** - Use specific assertions with helpful messages
5. **Error Testing** - Test both success and failure cases
6. **Documentation** - Include docstrings for test methods

## 🔄 Maintenance

### Regular Updates
- Update test data as the system evolves
- Add tests for new features and bug fixes
- Review and update mocks when dependencies change
- Monitor test performance and optimize as needed

### Test Quality Metrics
- **Coverage** - Aim for >80% code coverage
- **Performance** - Keep total test runtime under 2 minutes
- **Reliability** - Tests should pass consistently
- **Maintainability** - Tests should be easy to understand and modify

## 📞 Support

For issues with the testing framework:
1. Check the test logs in `test_results.log`
2. Review the detailed test report
3. Verify environment setup and dependencies
4. Check for known issues in the codebase

## 🎯 Future Enhancements

Planned improvements to the testing framework:
- [ ] Integration with code coverage tools
- [ ] Performance benchmarking tests
- [ ] Load testing for scalability
- [ ] Security testing for authentication
- [ ] Cross-platform compatibility tests
- [ ] Automated test data generation

---

## 📄 Summary

The APOSSS unit testing framework provides comprehensive coverage for all implemented techniques, ensuring system reliability and maintainability. With over 80+ tests covering 8 major components, the framework validates the correctness of ranking algorithms, machine learning models, graph processing, text analysis, and user management functionality.

Run `python tests/run_unit_tests.py` to execute all tests and generate detailed reports on system health and technique coverage. 