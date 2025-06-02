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
        ranking_mode = data.get('ranking_mode', 'hybrid')  # hybrid, ltr_only, traditional
        
        if not query_processor or not search_engine or not ranking_engine:
            return jsonify({'error': 'Search components not initialized'}), 500
        
        # Step 1: Process query with LLM
        logger.info(f"Processing search query: {user_query[:50]}...")
        processed_query = query_processor.process_query(user_query)
        
        if not processed_query:
            return jsonify({'error': 'Failed to process query'}), 500
        
        # Step 2: Search databases
        logger.info("Searching databases...")
        search_results = search_engine.search_all_databases(processed_query)
        
        if not search_results or not search_results.get('results'):
            return jsonify({
                'success': True,
                'message': 'No results found',
                'query_analysis': processed_query,
                'search_results': {'results': [], 'total_results': 0},
                'categorized_results': {'high_relevance': [], 'medium_relevance': [], 'low_relevance': []},
                'query_id': None
            })
        
        # Step 3: Get user feedback data for LTR
        user_feedback_data = {}
        try:
            if feedback_system:
                # Get aggregated feedback data for LTR features
                feedback_stats = feedback_system.get_all_feedback()
                
                # Convert to format expected by LTR ranker
                for feedback_entry in feedback_stats:
                    result_id = feedback_entry.get('result_id')
                    if result_id not in user_feedback_data:
                        user_feedback_data[result_id] = {'ratings': []}
                    user_feedback_data[result_id]['ratings'].append(feedback_entry.get('rating', 3))
        except Exception as e:
            logger.warning(f"Error getting feedback data for LTR: {e}")
            user_feedback_data = {}
        
        # Step 4: Rank results using selected mode
        logger.info(f"Ranking results using mode: {ranking_mode}...")
        ranked_results = ranking_engine.rank_search_results(
            search_results, processed_query, user_feedback_data, ranking_mode
        )
        
        # Generate unique query ID for feedback tracking
        query_id = f"query_{hash(user_query)}_{int(datetime.now().timestamp())}"
        
        return jsonify({
            'success': True,
            'query_analysis': processed_query,
            'search_results': ranked_results,
            'categorized_results': ranked_results.get('categorized_results', {}),
            'query_id': query_id,
            'ranking_mode': ranking_mode
        })
        
    except Exception as e:
        logger.error(f"Error in search endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback for result relevance"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['query_id', 'result_id', 'rating', 'feedback_type']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Submit feedback through feedback system
        feedback_result = feedback_system.submit_feedback({
            'query_id': data['query_id'],
            'result_id': data['result_id'],
            'rating': data['rating'],
            'feedback_type': data['feedback_type'],
            'user_session': data.get('user_session', 'anonymous'),
            'additional_data': data.get('additional_data', {}),
            'timestamp': datetime.now().isoformat()
        })
        
        feedback_id = feedback_result.get('feedback_id') or 'unknown'
        
        logger.info(f"Feedback submitted successfully: {feedback_id}")
        
        return jsonify({
            'success': True,
            'feedback_id': feedback_id,
            'message': 'Feedback submitted successfully'
        })
        
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

@app.route('/api/similarity/calculate', methods=['POST'])
def calculate_similarity():
    """Calculate real-time similarity between query and documents"""
    try:
        data = request.get_json()
        if not data or 'query' not in data or 'documents' not in data:
            return jsonify({'error': 'Query and documents are required'}), 400
        
        query = data['query']
        documents = data['documents']
        use_cache = data.get('use_cache', True)
        
        if not ranking_engine or not ranking_engine.use_embedding:
            return jsonify({'error': 'Real-time embedding system not available'}), 500
        
        # Optional: Process query with LLM for enhanced similarity
        processed_query = None
        if query_processor:
            try:
                processed_query = query_processor.process_query(query)
            except Exception as e:
                logger.warning(f"Failed to process query for similarity: {e}")
        
        # Calculate real-time similarity scores
        similarity_scores = ranking_engine.embedding_ranker.calculate_realtime_similarity(
            query, documents, processed_query, use_cache
        )
        
        # Combine documents with their scores
        results = []
        for i, (doc, score) in enumerate(zip(documents, similarity_scores)):
            results.append({
                'document': doc,
                'similarity_score': score,
                'rank': i + 1
            })
        
        # Sort by similarity score
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Update ranks after sorting
        for i, result in enumerate(results):
            result['rank'] = i + 1
        
        return jsonify({
            'success': True,
            'query': query,
            'total_documents': len(documents),
            'results': results,
            'processing_info': {
                'use_cache': use_cache,
                'processed_query_available': processed_query is not None,
                'similarity_calculation': 'realtime'
            }
        })
        
    except Exception as e:
        logger.error(f"Error calculating similarity: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/similarity/pairwise', methods=['POST'])
def calculate_pairwise_similarity():
    """Calculate similarity between two text strings"""
    try:
        data = request.get_json()
        if not data or 'text1' not in data or 'text2' not in data:
            return jsonify({'error': 'Both text1 and text2 are required'}), 400
        
        text1 = data['text1']
        text2 = data['text2']
        
        if not ranking_engine or not ranking_engine.use_embedding:
            return jsonify({'error': 'Real-time embedding system not available'}), 500
        
        # Calculate pairwise similarity
        similarity_score = ranking_engine.embedding_ranker.calculate_pairwise_similarity(text1, text2)
        
        # Ensure JSON serializable
        if hasattr(similarity_score, 'item'):
            similarity_score = similarity_score.item()
        else:
            similarity_score = float(similarity_score)
        
        return jsonify({
            'success': True,
            'text1': text1,
            'text2': text2,
            'similarity_score': similarity_score,
            'similarity_percentage': f"{similarity_score * 100:.2f}%"
        })
        
    except Exception as e:
        logger.error(f"Error calculating pairwise similarity: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/embedding/warmup', methods=['POST'])
def warm_up_cache():
    """Warm up the embedding cache with sample data"""
    try:
        data = request.get_json() or {}
        sample_size = data.get('sample_size', 50)
        
        if not ranking_engine or not ranking_engine.use_embedding:
            return jsonify({'error': 'Real-time embedding system not available'}), 500
        
        # Get sample documents from search if available
        sample_documents = []
        
        if search_engine and db_manager:
            try:
                # Use a generic search to get sample documents
                sample_query = {"keywords": {"primary": ["research", "science", "technology"]}}
                search_results = search_engine.search_all_databases(sample_query, hybrid_search=False)
                sample_documents = search_results.get('results', [])[:sample_size]
                
            except Exception as e:
                logger.warning(f"Failed to get sample documents from search: {e}")
        
        # If no documents from search, create dummy documents for warming
        if not sample_documents:
            sample_documents = [
                {
                    'id': f'sample_{i}',
                    'title': f'Sample Document {i}',
                    'description': f'This is a sample document for cache warming {i}',
                    'type': 'sample',
                    'metadata': {}
                }
                for i in range(min(sample_size, 10))
            ]
        
        # Warm up the cache
        success = ranking_engine.warm_up_embedding_cache(sample_documents)
        
        return jsonify({
            'success': success,
            'message': f'Cache warmed up with {len(sample_documents)} documents' if success else 'Failed to warm up cache',
            'documents_processed': len(sample_documents)
        })
        
    except Exception as e:
        logger.error(f"Error warming up cache: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/embedding/realtime-stats', methods=['GET'])
def get_realtime_embedding_stats():
    """Get detailed real-time embedding statistics"""
    try:
        if not ranking_engine or not ranking_engine.use_embedding:
            return jsonify({'error': 'Real-time embedding system not available'}), 500
        
        stats = ranking_engine.embedding_ranker.get_realtime_stats()
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Error getting real-time embedding stats: {str(e)}")
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

@app.route('/api/preindex/stats', methods=['GET'])
def get_preindex_stats():
    """Get pre-built index statistics"""
    try:
        if not search_engine:
            return jsonify({'error': 'Search engine not initialized'}), 500
        
        stats = search_engine.get_preindex_stats()
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Error getting pre-index stats: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/preindex/search', methods=['POST'])
def semantic_search():
    """Semantic search using pre-built index only"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        user_query = data['query']
        k = data.get('k', 20)  # Number of results to return
        
        if not search_engine:
            return jsonify({'error': 'Search engine not initialized'}), 500
        
        # Optional: Process query with LLM for enhanced semantic search
        processed_query = None
        if query_processor:
            try:
                processed_query = query_processor.process_query(user_query)
            except Exception as e:
                logger.warning(f"Failed to process query for semantic search: {e}")
        
        # Perform semantic search only
        results = search_engine.semantic_search_only(user_query, k, processed_query)
        
        return jsonify({
            'success': True,
            'query': user_query,
            'results': results,
            'total_results': len(results),
            'search_type': 'semantic_only'
        })
        
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/preindex/build', methods=['POST'])
def build_preindex():
    """Trigger pre-index building (asynchronous)"""
    try:
        # Import here to avoid circular imports
        from build_index import DocumentIndexBuilder
        import threading
        
        data = request.get_json() or {}
        resume = data.get('resume', True)
        
        def build_index_async():
            """Build index in background thread"""
            try:
                builder = DocumentIndexBuilder()
                success = builder.build_full_index(resume=resume)
                logger.info(f"Pre-index building completed: {'Success' if success else 'Failed'}")
            except Exception as e:
                logger.error(f"Error in async index building: {e}")
        
        # Start building in background
        thread = threading.Thread(target=build_index_async, daemon=True)
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Pre-index building started in background',
            'resume': resume
        })
        
    except Exception as e:
        logger.error(f"Error starting pre-index build: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/preindex/progress', methods=['GET'])
def get_preindex_progress():
    """Get pre-indexing progress"""
    try:
        from build_index import DocumentIndexBuilder
        import os
        import json
        
        builder = DocumentIndexBuilder()
        progress_file = builder.progress_file
        
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress = json.load(f)
            
            return jsonify({
                'success': True,
                'progress': progress,
                'in_progress': not progress.get('completed', False)
            })
        else:
            return jsonify({
                'success': True,
                'progress': None,
                'in_progress': False,
                'message': 'No indexing in progress'
            })
        
    except Exception as e:
        logger.error(f"Error getting pre-index progress: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ltr/stats', methods=['GET'])
def get_ltr_stats():
    """Get LTR model statistics"""
    try:
        if not ranking_engine:
            return jsonify({'error': 'Ranking engine not initialized'}), 500
        
        stats = ranking_engine.get_ltr_stats()
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Error getting LTR stats: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ltr/train', methods=['POST'])
def train_ltr_model():
    """Train the LTR model with collected feedback data"""
    try:
        data = request.get_json() or {}
        
        if not ranking_engine:
            return jsonify({'error': 'Ranking engine not initialized'}), 500
        
        # Check if LTR is available
        ltr_stats = ranking_engine.get_ltr_stats()
        if not ltr_stats.get('ltr_available', False):
            return jsonify({'error': 'LTR functionality not available'}), 400
        
        # Collect training data from feedback
        min_feedback_count = data.get('min_feedback_count', 50)
        feedback_data = feedback_system.get_training_data(min_feedback_count=min_feedback_count)
        
        if not feedback_data:
            return jsonify({
                'error': f'Insufficient training data. Need at least {min_feedback_count} feedback entries.'
            }), 400
        
        # Convert feedback to LTR training format
        training_data = []
        
        for feedback_entry in feedback_data:
            try:
                # Get the original search query and results
                query_id = feedback_entry.get('query_id')
                result_id = feedback_entry.get('result_id')
                rating = feedback_entry.get('rating', 3)
                
                # Convert rating to relevance label (1-5 scale to 0-4)
                relevance_label = max(0, rating - 1)
                
                # Get query details (you might need to store these separately)
                # For now, we'll use placeholder data
                query_text = feedback_entry.get('query_text', 'sample query')
                
                # Get result details
                result_data = feedback_entry.get('result_data', {})
                
                # Create training example
                training_example = {
                    'query_id': hash(query_text),
                    'result_id': result_id,
                    'relevance_label': relevance_label,
                    # Add more features as needed
                    'title_length': len(result_data.get('title', '').split()),
                    'description_length': len(result_data.get('description', '').split()),
                    'result_type': result_data.get('type', 'unknown')
                }
                
                training_data.append(training_example)
                
            except Exception as e:
                logger.warning(f"Error processing feedback entry: {e}")
                continue
        
        if not training_data:
            return jsonify({'error': 'No valid training data could be prepared'}), 400
        
        # Train the model
        training_stats = ranking_engine.train_ltr_model(training_data)
        
        logger.info(f"LTR model trained successfully with {len(training_data)} examples")
        
        return jsonify({
            'success': True,
            'message': 'LTR model trained successfully',
            'training_stats': training_stats,
            'training_data_size': len(training_data)
        })
        
    except Exception as e:
        logger.error(f"Error training LTR model: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ltr/features', methods=['POST'])
def extract_ltr_features():
    """Extract LTR features for a query and results (for testing)"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data or 'results' not in data:
            return jsonify({'error': 'Query and results required'}), 400
        
        if not ranking_engine:
            return jsonify({'error': 'Ranking engine not initialized'}), 500
        
        # Check if LTR is available
        ltr_stats = ranking_engine.get_ltr_stats()
        if not ltr_stats.get('ltr_available', False):
            return jsonify({'error': 'LTR functionality not available'}), 400
        
        # Extract features
        query = data['query']
        results = data['results']
        processed_query = data.get('processed_query', {})
        
        # Get LTR ranker and extract features
        ltr_ranker = ranking_engine.ltr_ranker
        features_df = ltr_ranker.extract_features(query, results, processed_query)
        
        # Convert to dictionary format
        features_dict = features_df.to_dict('records') if not features_df.empty else []
        
        return jsonify({
            'success': True,
            'features': features_dict,
            'feature_names': features_df.columns.tolist() if not features_df.empty else [],
            'num_features': len(features_df.columns) if not features_df.empty else 0,
            'num_results': len(features_dict)
        })
        
    except Exception as e:
        logger.error(f"Error extracting LTR features: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ltr/feature-importance', methods=['GET'])
def get_feature_importance():
    """Get feature importance from trained LTR model"""
    try:
        if not ranking_engine:
            return jsonify({'error': 'Ranking engine not initialized'}), 500
        
        # Check if LTR is available and trained
        ltr_stats = ranking_engine.get_ltr_stats()
        if not ltr_stats.get('ltr_available', False):
            return jsonify({'error': 'LTR functionality not available'}), 400
        
        if not ltr_stats.get('model_trained', False):
            return jsonify({'error': 'LTR model not trained yet'}), 400
        
        # Get feature importance
        importance = ranking_engine.ltr_ranker.get_feature_importance()
        
        # Sort by importance
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        return jsonify({
            'success': True,
            'feature_importance': dict(sorted_importance),
            'top_features': sorted_importance[:10],  # Top 10 features
            'total_features': len(importance)
        })
        
    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    app.run(host='0.0.0.0', port=port, debug=debug) 