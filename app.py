from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import os
import logging

# Import our modules
from modules.database_manager import DatabaseManager
from modules.llm_processor import LLMProcessor
from modules.query_processor import QueryProcessor
from modules.search_engine import SearchEngine

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
try:
    db_manager = DatabaseManager()
    llm_processor = LLMProcessor()
    query_processor = QueryProcessor(llm_processor)
    search_engine = SearchEngine(db_manager)
    logger.info("All components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize components: {str(e)}")
    db_manager = None
    llm_processor = None
    query_processor = None
    search_engine = None

@app.route('/')
def index():
    """Serve the main interface"""
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search():
    """Full search endpoint - Phase 2 functionality"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        user_query = data['query']
        
        if not query_processor or not search_engine:
            return jsonify({'error': 'Search components not initialized'}), 500
        
        # Step 1: Process query with LLM
        logger.info(f"Processing search query: {user_query[:50]}...")
        processed_query = query_processor.process_query(user_query)
        
        if not processed_query:
            return jsonify({'error': 'Failed to process query'}), 500
        
        # Step 2: Search all databases
        search_results = search_engine.search_all_databases(processed_query)
        
        return jsonify({
            'success': True,
            'query_analysis': processed_query,
            'search_results': search_results
        })
        
    except Exception as e:
        logger.error(f"Error in search endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/test-llm', methods=['POST'])
def test_llm():
    """Test endpoint for LLM query processing"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        user_query = data['query']
        
        if not query_processor:
            return jsonify({'error': 'Query processor not initialized'}), 500
        
        # Process query with LLM
        processed_query = query_processor.process_query(user_query)
        
        return jsonify({
            'success': True,
            'original_query': user_query,
            'processed_query': processed_query
        })
        
    except Exception as e:
        logger.error(f"Error in test_llm endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/test-db', methods=['GET'])
def test_db():
    """Test endpoint for database connections"""
    try:
        if not db_manager:
            return jsonify({'error': 'Database manager not initialized'}), 500
        
        # Test all database connections
        connection_status = db_manager.test_connections()
        
        return jsonify({
            'success': True,
            'database_status': connection_status
        })
        
    except Exception as e:
        logger.error(f"Error in test_db endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        'database_manager': db_manager is not None,
        'llm_processor': llm_processor is not None,
        'query_processor': query_processor is not None,
        'search_engine': search_engine is not None
    }
    
    return jsonify({
        'status': 'healthy' if all(status.values()) else 'unhealthy',
        'components': status
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    app.run(host='0.0.0.0', port=port, debug=debug) 