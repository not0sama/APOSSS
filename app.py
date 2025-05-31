from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import os
import logging
import uuid
from datetime import datetime

# Import our modules
from modules.database_manager import DatabaseManager
from modules.llm_processor import LLMProcessor
from modules.query_processor import QueryProcessor
from modules.search_engine import SearchEngine
from modules.ranking_engine import RankingEngine
from modules.feedback_system import FeedbackSystem

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
    ranking_engine = RankingEngine()
    feedback_system = FeedbackSystem(db_manager)
    logger.info("All components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize components: {str(e)}")
    db_manager = None
    llm_processor = None
    query_processor = None
    search_engine = None
    ranking_engine = None
    feedback_system = None

@app.route('/')
def index():
    """Serve the main interface"""
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search():
    """Full search endpoint with ranking - Phase 3 functionality"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        user_query = data['query']
        
        if not query_processor or not search_engine or not ranking_engine:
            return jsonify({'error': 'Search components not initialized'}), 500
        
        # Step 1: Process query with LLM
        logger.info(f"Processing search query: {user_query[:50]}...")
        processed_query = query_processor.process_query(user_query)
        
        if not processed_query:
            return jsonify({'error': 'Failed to process query'}), 500
        
        # Step 2: Search all databases
        search_results = search_engine.search_all_databases(processed_query)
        
        # Step 3: Rank results (Phase 3)
        ranked_results = ranking_engine.rank_search_results(search_results, processed_query)
        
        # Step 4: Categorize results
        categorized_results = ranking_engine.categorize_results(ranked_results.get('results', []))
        
        # Generate unique query ID for feedback tracking
        query_id = str(uuid.uuid4())
        
        return jsonify({
            'success': True,
            'query_id': query_id,
            'query_analysis': processed_query,
            'search_results': ranked_results,
            'categorized_results': categorized_results
        })
        
    except Exception as e:
        logger.error(f"Error in search endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback for search results"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Feedback data is required'}), 400
        
        if not feedback_system:
            return jsonify({'error': 'Feedback system not initialized'}), 500
        
        # Submit feedback
        result = feedback_system.submit_feedback(data)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/feedback/stats', methods=['GET'])
def get_feedback_stats():
    """Get feedback statistics"""
    try:
        if not feedback_system:
            return jsonify({'error': 'Feedback system not initialized'}), 500
        
        stats = feedback_system.get_feedback_stats()
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Error getting feedback stats: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/feedback/recent', methods=['GET'])
def get_recent_feedback():
    """Get recent feedback entries"""
    try:
        if not feedback_system:
            return jsonify({'error': 'Feedback system not initialized'}), 500
        
        limit = request.args.get('limit', 10, type=int)
        feedback_list = feedback_system.get_recent_feedback(limit)
        
        return jsonify({
            'success': True,
            'feedback': feedback_list
        })
        
    except Exception as e:
        logger.error(f"Error getting recent feedback: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/embedding/stats', methods=['GET'])
def get_embedding_stats():
    """Get embedding system statistics"""
    try:
        if not ranking_engine:
            return jsonify({'error': 'Ranking engine not initialized'}), 500
        
        stats = ranking_engine.get_embedding_stats()
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Error getting embedding stats: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/embedding/clear-cache', methods=['POST'])
def clear_embedding_cache():
    """Clear the embedding cache"""
    try:
        if not ranking_engine:
            return jsonify({'error': 'Ranking engine not initialized'}), 500
        
        success = ranking_engine.clear_embedding_cache()
        
        return jsonify({
            'success': success,
            'message': 'Embedding cache cleared successfully' if success else 'Failed to clear cache'
        })
        
    except Exception as e:
        logger.error(f"Error clearing embedding cache: {str(e)}")
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
        'search_engine': search_engine is not None,
        'ranking_engine': ranking_engine is not None,
        'feedback_system': feedback_system is not None
    }
    
    return jsonify({
        'status': 'healthy' if all(status.values()) else 'unhealthy',
        'components': status
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    app.run(host='0.0.0.0', port=port, debug=debug) 