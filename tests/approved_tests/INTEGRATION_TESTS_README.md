# APOSSS System Integration Tests

## Overview

The APOSSS System Integration Tests provide comprehensive validation of the entire Academic Papers and Other Scholarly Sources Search system. These tests verify that all modules work together correctly as a complete end-to-end system.

## Test Coverage

### 🏥 System Health and Connectivity (`test_01_system_health_and_connectivity`)
- **Purpose**: Verify system availability and component health
- **Tests**:
  - API health check endpoint
  - Database connectivity for all databases
  - Component status verification
  - Document count validation
- **Expected Duration**: 5-10 seconds

### 👤 User Management Workflow (`test_02_user_management_workflow`)
- **Purpose**: Test complete user authentication and management
- **Tests**:
  - User registration for multiple user types (professor, researcher, student)
  - User login and token generation
  - Profile access and validation
  - JWT token functionality
- **Expected Duration**: 10-15 seconds

### 🤖 LLM Processing Integration (`test_03_llm_processing_integration`)
- **Purpose**: Verify LLM query processing across different query types
- **Tests**:
  - Query analysis and intent detection
  - Keyword extraction and entity recognition
  - Academic field classification
  - Confidence scoring
  - Multiple query complexities (low, medium, high)
- **Expected Duration**: 30-60 seconds

### 🔍 Search Engine Integration (`test_04_search_engine_integration`)
- **Purpose**: Test search functionality with all ranking modes
- **Tests**:
  - Anonymous search across all ranking modes
  - Authenticated search with personalization
  - Result validation and scoring
  - Personalization verification
  - Multiple query types and complexities
- **Expected Duration**: 60-90 seconds

### 🏆 Ranking System Integration (`test_05_ranking_system_integration`)
- **Purpose**: Verify all ranking algorithms work correctly
- **Tests**:
  - Traditional ranking (heuristic + TF-IDF)
  - Hybrid ranking (traditional + embedding + intent)
  - LTR-only ranking (when available)
  - Score consistency and sorting
  - Ranking metadata validation
- **Expected Duration**: 30-45 seconds

### 🧠 Embedding and Knowledge Graph Integration (`test_06_embedding_and_knowledge_graph_integration`)
- **Purpose**: Test semantic search and knowledge graph functionality
- **Tests**:
  - Embedding generation and similarity calculation
  - Knowledge graph statistics and structure
  - Semantic search integration
  - Graph-based authority scoring
- **Expected Duration**: 20-30 seconds

### 🎯 LTR System Integration (`test_07_ltr_system_integration`)
- **Purpose**: Verify Learning-to-Rank system functionality
- **Tests**:
  - LTR model availability and training status
  - Feature extraction and ranking
  - Model prediction and scoring
  - LTR-specific ranking features
- **Expected Duration**: 15-25 seconds

### 📝 Feedback System Integration (`test_08_feedback_system_integration`)
- **Purpose**: Test feedback collection and processing
- **Tests**:
  - Feedback submission for various types (thumbs up/down, ratings)
  - Feedback statistics and aggregation
  - Multiple feedback scenarios
  - Data persistence and retrieval
- **Expected Duration**: 10-15 seconds

### ⚡ Performance and Load Testing (`test_09_performance_and_load_testing`)
- **Purpose**: Test system performance under concurrent load
- **Tests**:
  - Concurrent search requests (10 simultaneous)
  - Response time measurement
  - Success rate calculation
  - Performance metrics analysis
- **Expected Duration**: 20-30 seconds

### 🛡️ Error Handling and Recovery (`test_10_error_handling_and_recovery`)
- **Purpose**: Verify system resilience and error handling
- **Tests**:
  - Empty query handling
  - Invalid ranking mode handling
  - Malformed JSON handling
  - Unauthorized access handling
- **Expected Duration**: 5-10 seconds

### 🔄 End-to-End Workflow (`test_11_end_to_end_workflow`)
- **Purpose**: Test complete user workflow from search to feedback
- **Tests**:
  - User authentication and search
  - Result retrieval and ranking
  - Feedback submission
  - Profile access and management
  - Cross-system integration
- **Expected Duration**: 15-25 seconds

## Prerequisites

### System Requirements
1. **Flask Application**: Must be running on `http://localhost:5000` (or specified URL)
2. **Databases**: All databases must be connected and accessible
3. **Environment Variables**: All required environment variables must be configured
4. **API Keys**: Valid API keys for external services (Gemini, etc.)

### Dependencies
```bash
# Install required packages
pip install -r requirements.txt

# Additional testing dependencies
pip install requests concurrent.futures
```

### Environment Setup
```bash
# Set up environment variables
cp config.env.example config.env
# Edit config.env with your actual values

# Start the Flask application
python app.py
```

## Running the Tests

### 1. Run All Integration Tests
```bash
# Run all integration tests
python tests/run_integration_tests.py

# Or run directly
python tests/test_full_system_integration.py
```

### 2. Run Specific Test Categories
```bash
# Run system health tests
python tests/run_integration_tests.py --category system

# Run search-related tests
python tests/run_integration_tests.py --category search

# Run AI/ML tests
python tests/run_integration_tests.py --category ai

# Run performance tests
python tests/run_integration_tests.py --category performance
```

### 3. Run Specific Tests
```bash
# Run specific test methods
python tests/run_integration_tests.py --tests test_01_system_health test_04_search_engine

# Run with verbose output
python tests/run_integration_tests.py --verbosity 2
```

### 4. Generate Test Reports
```bash
# Save test report to file
python tests/run_integration_tests.py --save-report

# Custom report filename
python tests/run_integration_tests.py --save-report --report-file my_integration_report.json
```

## Test Categories

| Category | Tests | Description |
|----------|-------|-------------|
| `system` | Health, Error Handling | Core system functionality |
| `user` | User Management | Authentication and user workflows |
| `search` | LLM, Search Engine, Ranking | Search functionality |
| `ai` | Embedding, Knowledge Graph, LTR | AI/ML components |
| `feedback` | Feedback System | User feedback and learning |
| `performance` | Load Testing | Performance and scalability |
| `workflow` | End-to-End | Complete user workflows |

## Expected Output

### Successful Test Run
```
🚀 APOSSS Integration Test Runner
================================================================================
🔍 Checking system availability...
✅ System available: 8/8 components healthy

🧪 Running 11 integration tests...
================================================================================

🏥 Testing System Health and Connectivity...
   ✅ System health check passed: 8 components healthy
   ✅ Database connectivity verified: 15247 total documents

👤 Testing User Management Workflow...
   ✅ User alice_johnson registered successfully
   ✅ User bob_smith registered successfully  
   ✅ User student_user registered successfully
   ✅ Profile access verified for alice_johnson

🤖 Testing LLM Processing Integration...
   ✅ LLM processing verified for: machine learning algorithms for medical diagnosis... (confidence: 0.87)
   ✅ LLM processing verified for: renewable energy solar panel efficiency optimi... (confidence: 0.82)

🔍 Testing Search Engine Integration...
   Testing anonymous search...
   ✅ Anonymous search (traditional): machine learning algorithms f... -> 25 results
   ✅ Anonymous search (hybrid): machine learning algorithms f... -> 25 results
   Testing authenticated search with personalization...
   ✅ Personalized search for alice_johnson: machine learning algorithms f... -> 28 results

🏆 Testing Ranking System Integration...
   ✅ traditional ranking: 25 results, top score: 0.892
   ✅ hybrid ranking: 25 results, top score: 0.934
   ✅ Ranking comparison: traditional (4 components) vs hybrid (7 components)

🧠 Testing Embedding and Knowledge Graph Integration...
   ✅ Embedding system: 384D embeddings, 25 similarities
   ✅ Knowledge graph: 1250 nodes, 3420 edges

🎯 Testing LTR System Integration...
   📊 LTR available: True, Model trained: True
   ✅ LTR ranking verified: 3 LTR features

📝 Testing Feedback System Integration...
   ✅ Feedback submitted: thumbs_up (rating: 5)
   ✅ Feedback submitted: rating (rating: 4)
   ✅ Feedback stats: 157 total feedback, 4.2 avg rating

⚡ Testing System Performance Under Load...
   ✅ Load test: 10/10 requests successful (100.0%)
   ⏱️ Performance: avg 2.34s, max 3.12s, total 5.67s

🛡️ Testing Error Handling and Recovery...
   ✅ Empty query handled gracefully
   ✅ Invalid ranking mode handled
   ✅ Malformed JSON handled
   ✅ Unauthorized access handled

🔄 Testing Complete End-to-End Workflow...
   ✅ Step 1: Search completed - 28 results
   ✅ Step 2: Feedback submitted
   ✅ Step 3: Related search completed - 31 results
   ✅ Step 4: User profile accessed
   ✅ Step 5: LTR system ready for learning
   ✅ Complete end-to-end workflow successful

================================================================================
📊 APOSSS Integration Test Results
================================================================================

🎯 Overall Summary:
   Total tests: 11
   Passed: 11 ✅
   Failed: 0 ❌
   Errors: 0 🔥
   Skipped: 0 ⏭️
   Success rate: 100.0%
   Total duration: 245.67s

💡 Recommendations:
   🎉 All tests passed! System integration is working correctly.
   📈 Consider running performance benchmarks for optimization
```

## Test Architecture

### Test Structure
```
test_full_system_integration.py
├── APOSSSSystemIntegrationTest (unittest.TestCase)
│   ├── setUpClass() - Initialize test environment
│   ├── setUp() - Pre-test setup
│   ├── test_01_system_health_and_connectivity()
│   ├── test_02_user_management_workflow()
│   ├── test_03_llm_processing_integration()
│   ├── test_04_search_engine_integration()
│   ├── test_05_ranking_system_integration()
│   ├── test_06_embedding_and_knowledge_graph_integration()
│   ├── test_07_ltr_system_integration()
│   ├── test_08_feedback_system_integration()
│   ├── test_09_performance_and_load_testing()
│   ├── test_10_error_handling_and_recovery()
│   ├── test_11_end_to_end_workflow()
│   ├── tearDown() - Post-test cleanup
│   └── tearDownClass() - Final cleanup and reporting
```

### Test Data
- **Test Users**: 3 different user types (professor, researcher, student)
- **Test Queries**: 5 queries with varying complexity and fields
- **Feedback Scenarios**: 5 different feedback types and ratings
- **Ranking Modes**: Traditional, Hybrid, LTR-only
- **Performance**: 10 concurrent requests for load testing

## Troubleshooting

### Common Issues

#### 1. System Not Available
```
❌ System not available. Please ensure:
   1. Flask application is running (python app.py)
   2. All databases are connected
   3. Environment variables are configured
```
**Solution**: Start the Flask app and verify database connections.

#### 2. LLM Processing Failures
```
❌ LLM processing failed for query: machine learning...
```
**Solution**: Check API keys and network connectivity for LLM services.

#### 3. Database Connection Issues
```
❌ Database connectivity test failed
```
**Solution**: Verify database configurations and network access.

#### 4. Performance Issues
```
🐌 Slow tests detected (>10s): 3
```
**Solution**: Check system resources and optimize database queries.

### Debugging Tips

1. **Enable Detailed Logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Run Individual Tests**:
   ```bash
   python tests/run_integration_tests.py --tests test_01_system_health
   ```

3. **Check System Health First**:
   ```bash
   curl http://localhost:5000/api/health
   ```

4. **Verify Database Connections**:
   ```bash
   curl http://localhost:5000/api/test-db
   ```

## Continuous Integration

### GitHub Actions Example
```yaml
name: Integration Tests

on: [push, pull_request]

jobs:
  integration-tests:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Start APOSSS system
      run: |
        python app.py &
        sleep 30
    
    - name: Run integration tests
      run: |
        python tests/run_integration_tests.py --save-report
    
    - name: Upload test report
      uses: actions/upload-artifact@v2
      with:
        name: integration-test-report
        path: integration_test_report_*.json
```

### Docker Integration
```dockerfile
# Integration test container
FROM python:3.9

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

CMD ["python", "tests/run_integration_tests.py"]
```

## Best Practices

### 1. Test Isolation
- Each test is independent and can run in any order
- Tests clean up after themselves
- No shared state between tests

### 2. Realistic Data
- Tests use realistic queries and data
- Multiple user types and scenarios
- Actual API calls to real endpoints

### 3. Comprehensive Coverage
- Tests cover all major system components
- Both success and failure scenarios
- Performance and load testing

### 4. Clear Reporting
- Detailed test results and metrics
- Performance benchmarks
- Actionable recommendations

### 5. Maintainability
- Well-documented test cases
- Modular test structure
- Easy to add new tests

## Extending the Tests

### Adding New Test Cases
1. Add new test method to `APOSSSSystemIntegrationTest`
2. Follow naming convention: `test_##_descriptive_name`
3. Add to appropriate category in `run_integration_tests.py`
4. Update documentation

### Adding New Test Categories
1. Add category to `category_patterns` in `run_integration_tests.py`
2. Update command-line parser choices
3. Document new category in README

## Performance Benchmarks

### Expected Performance
- **System Health**: < 5 seconds
- **Search Operations**: < 10 seconds per query
- **LLM Processing**: < 30 seconds per query
- **Concurrent Load**: 80%+ success rate with 10 concurrent requests
- **Total Test Suite**: < 5 minutes

### Performance Monitoring
The integration tests include performance metrics and will flag:
- Tests taking longer than expected
- Low success rates in load testing
- System response degradation

## Security Considerations

### Test Data
- Tests use synthetic test data
- No real user credentials or sensitive information
- Test users are clearly marked as test accounts

### API Security
- Tests verify authentication and authorization
- Invalid token handling
- Unauthorized access prevention

## Support

For issues with integration tests:
1. Check the troubleshooting section
2. Verify system prerequisites
3. Run individual test categories to isolate issues
4. Check system logs for detailed error information

## Conclusion

The APOSSS Integration Tests provide comprehensive validation of the entire system, ensuring that all components work together correctly in real-world scenarios. They serve as both a quality assurance tool and a system health monitor, helping maintain the reliability and performance of the APOSSS platform. 