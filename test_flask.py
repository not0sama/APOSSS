#!/usr/bin/env python3
"""Test Flask app with component initialization"""
import os
import logging
from dotenv import load_dotenv
from flask import Flask, jsonify

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Initialize components
try:
    from modules.database_manager import DatabaseManager
    from modules.llm_processor import LLMProcessor
    from modules.query_processor import QueryProcessor
    from modules.search_engine import SearchEngine
    from modules.ranking_engine import RankingEngine
    from modules.feedback_system import FeedbackSystem
    from modules.user_manager import UserManager
    
    db_manager = DatabaseManager()
    llm_processor = LLMProcessor()
    query_processor = QueryProcessor(llm_processor)
    search_engine = SearchEngine(db_manager)
    ranking_engine = RankingEngine()
    feedback_system = FeedbackSystem(db_manager)
    user_manager = UserManager(db_manager)
    
    logger.info("All components initialized successfully")
    initialization_success = True
    initialization_error = None
except Exception as e:
    logger.error(f"Failed to initialize components: {str(e)}")
    initialization_success = False
    initialization_error = str(e)
    db_manager = None
    llm_processor = None
    query_processor = None
    search_engine = None
    ranking_engine = None
    feedback_system = None
    user_manager = None

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if not initialization_success:
        return jsonify({
            'status': 'unhealthy',
            'error': initialization_error,
            'components': {
                'database_manager': False,
                'llm_processor': False,
                'query_processor': False,
                'search_engine': False,
                'ranking_engine': False,
                'feedback_system': False,
                'user_manager': False
            }
        })
    
    status = {
        'database_manager': db_manager is not None,
        'llm_processor': llm_processor is not None,
        'query_processor': query_processor is not None,
        'search_engine': search_engine is not None,
        'ranking_engine': ranking_engine is not None,
        'feedback_system': feedback_system is not None,
        'user_manager': user_manager is not None
    }
    
    return jsonify({
        'status': 'healthy' if all(status.values()) else 'unhealthy',
        'components': status
    })

@app.route('/api/test-simple', methods=['GET'])
def test_simple():
    """Simple test endpoint"""
    return jsonify({'message': 'Flask app is running', 'success': True})

if __name__ == '__main__':
    print("Starting test Flask app...")
    app.run(host='0.0.0.0', port=5001, debug=True) 