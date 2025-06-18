import logging
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

logger = logging.getLogger(__name__)

class FeedbackSystem:
    """User feedback collection and storage system"""
    
    def __init__(self, db_manager=None):
        """Initialize feedback system"""
        self.db_manager = db_manager
        self.feedback_collection = None
        
        # Try to initialize feedback collection
        self._initialize_feedback_storage()
        logger.info("Feedback system initialized successfully")
    
    def _initialize_feedback_storage(self):
        """Initialize feedback storage (MongoDB collection or fallback to file)"""
        try:
            if self.db_manager:
                # Try to use the dedicated APOSSS feedback database
                feedback_db = self.db_manager.get_database('aposss')
                if feedback_db is not None:
                    self.feedback_collection = feedback_db['user_feedback']
                    logger.info("Using MongoDB APOSSS database for feedback storage")
                    return
        except Exception as e:
            logger.warning(f"Could not initialize MongoDB feedback storage: {str(e)}")
        
        # Fallback to file-based storage
        self.feedback_file = "feedback_data.jsonl"
        logger.info("Using file-based feedback storage")
    
    def submit_feedback(self, feedback_data: Dict[str, Any], user_manager=None) -> Dict[str, Any]:
        """
        Submit user feedback for a search result
        
        Args:
            feedback_data: Dictionary containing feedback information
                - query_id: Unique identifier for the query
                - result_id: ID of the result being rated
                - rating: User rating (1-5 or thumbs up/down)
                - feedback_type: 'rating', 'thumbs', 'detailed'
                - comment: Optional text comment
                - user_session: Session identifier
                - user_id: User ID (if authenticated)
                
        Returns:
            Success/failure status
        """
        try:
            # Validate feedback data
            if not self._validate_feedback(feedback_data):
                return {'success': False, 'error': 'Invalid feedback data'}
            
            # Add metadata
            enhanced_feedback = self._enhance_feedback(feedback_data)
            
            # Store feedback
            if self.feedback_collection is not None:
                # Store in MongoDB
                result = self.feedback_collection.insert_one(enhanced_feedback)
                feedback_id = str(result.inserted_id)
            else:
                # Store in file
                feedback_id = self._store_feedback_to_file(enhanced_feedback)
            
            # Track user interaction if user manager is available
            if user_manager:
                user_id = feedback_data.get('user_id') or feedback_data.get('user_session', 'anonymous')
                user_manager.track_user_interaction(user_id, {
                    'action': 'feedback',
                    'query': feedback_data.get('query', ''),
                    'result_id': feedback_data.get('result_id', ''),
                    'session_id': feedback_data.get('user_session', ''),
                    'metadata': {
                        'rating': feedback_data.get('rating'),
                        'feedback_type': feedback_data.get('feedback_type'),
                        'result_type': feedback_data.get('result_type', ''),
                        'feedback_id': feedback_id
                    }
                })
            
            logger.info(f"Feedback submitted successfully: {feedback_id}")
            return {
                'success': True,
                'feedback_id': feedback_id,
                'message': 'Feedback submitted successfully'
            }
            
        except Exception as e:
            logger.error(f"Error submitting feedback: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get statistics about collected feedback"""
        try:
            if self.feedback_collection is not None:
                # Get stats from MongoDB
                total_feedback = self.feedback_collection.count_documents({})
                
                # Aggregate rating statistics
                pipeline = [
                    {"$group": {
                        "_id": None,
                        "avg_rating": {"$avg": "$rating"},
                        "total_positive": {"$sum": {"$cond": [{"$gte": ["$rating", 4]}, 1, 0]}},
                        "total_negative": {"$sum": {"$cond": [{"$lte": ["$rating", 2]}, 1, 0]}}
                    }}
                ]
                
                stats_result = list(self.feedback_collection.aggregate(pipeline))
                if stats_result:
                    stats = stats_result[0]
                    return {
                        'total_feedback': total_feedback,
                        'average_rating': round(stats.get('avg_rating', 0), 2),
                        'positive_feedback': stats.get('total_positive', 0),
                        'negative_feedback': stats.get('total_negative', 0),
                        'storage_type': 'mongodb'
                    }
            
            # Fallback to file-based stats
            return self._get_file_feedback_stats()
            
        except Exception as e:
            logger.error(f"Error getting feedback stats: {str(e)}")
            return {
                'total_feedback': 0,
                'average_rating': 0,
                'positive_feedback': 0,
                'negative_feedback': 0,
                'storage_type': 'error'
            }
    
    def get_result_feedback(self, result_id: str) -> List[Dict[str, Any]]:
        """Get all feedback for a specific result"""
        try:
            if self.feedback_collection is not None:
                # Query MongoDB
                feedback_list = list(self.feedback_collection.find(
                    {'result_id': result_id},
                    {'_id': 0}  # Exclude MongoDB _id field
                ))
                return feedback_list
            else:
                # Query file storage
                return self._get_file_result_feedback(result_id)
                
        except Exception as e:
            logger.error(f"Error getting result feedback: {str(e)}")
            return []
    
    def get_query_feedback(self, query_id: str) -> List[Dict[str, Any]]:
        """Get all feedback for a specific query"""
        try:
            if self.feedback_collection is not None:
                # Query MongoDB
                feedback_list = list(self.feedback_collection.find(
                    {'query_id': query_id},
                    {'_id': 0}  # Exclude MongoDB _id field
                ))
                return feedback_list
            else:
                # Query file storage
                return self._get_file_query_feedback(query_id)
                
        except Exception as e:
            logger.error(f"Error getting query feedback: {str(e)}")
            return []
    
    def _validate_feedback(self, feedback_data: Dict[str, Any]) -> bool:
        """Validate feedback data structure"""
        required_fields = ['query_id', 'result_id', 'rating', 'feedback_type']
        
        for field in required_fields:
            if field not in feedback_data:
                logger.warning(f"Missing required field in feedback: {field}")
                return False
        
        # Validate rating value
        rating = feedback_data.get('rating')
        if not isinstance(rating, (int, float)) or rating < 1 or rating > 5:
            logger.warning(f"Invalid rating value: {rating}")
            return False
        
        return True
    
    def _enhance_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add metadata to feedback data"""
        enhanced = feedback_data.copy()
        enhanced.update({
            'submitted_at': datetime.now().isoformat(),
            'feedback_version': '1.0'
        })
        
        return enhanced
    
    def _store_feedback_to_file(self, feedback_data: Dict[str, Any]) -> str:
        """Store feedback to JSON Lines file"""
        try:
            with open(self.feedback_file, 'a', encoding='utf-8') as f:
                json.dump(feedback_data, f)
                f.write('\n')
            
            # Generate a simple ID
            feedback_id = f"fb_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            return feedback_id
            
        except Exception as e:
            logger.error(f"Error storing feedback to file: {str(e)}")
            raise
    
    def _get_file_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics from file storage"""
        try:
            if not os.path.exists(self.feedback_file):
                return {
                    'total_feedback': 0,
                    'average_rating': 0,
                    'positive_feedback': 0,
                    'negative_feedback': 0,
                    'storage_type': 'file'
                }
            
            ratings = []
            positive_count = 0
            negative_count = 0
            
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        feedback = json.loads(line.strip())
                        rating = feedback.get('rating', 0)
                        ratings.append(rating)
                        
                        if rating >= 4:
                            positive_count += 1
                        elif rating <= 2:
                            negative_count += 1
                    except json.JSONDecodeError:
                        continue
            
            avg_rating = sum(ratings) / len(ratings) if ratings else 0
            
            return {
                'total_feedback': len(ratings),
                'average_rating': round(avg_rating, 2),
                'positive_feedback': positive_count,
                'negative_feedback': negative_count,
                'storage_type': 'file'
            }
            
        except Exception as e:
            logger.error(f"Error getting file feedback stats: {str(e)}")
            return {
                'total_feedback': 0,
                'average_rating': 0,
                'positive_feedback': 0,
                'negative_feedback': 0,
                'storage_type': 'file_error'
            }
    
    def _get_file_result_feedback(self, result_id: str) -> List[Dict[str, Any]]:
        """Get feedback for a specific result from file storage"""
        feedback_list = []
        
        try:
            if not os.path.exists(self.feedback_file):
                return feedback_list
            
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        feedback = json.loads(line.strip())
                        if feedback.get('result_id') == result_id:
                            feedback_list.append(feedback)
                    except json.JSONDecodeError:
                        continue
            
        except Exception as e:
            logger.error(f"Error getting file result feedback: {str(e)}")
        
        return feedback_list
    
    def _get_file_query_feedback(self, query_id: str) -> List[Dict[str, Any]]:
        """Get feedback for a specific query from file storage"""
        feedback_list = []
        
        try:
            if not os.path.exists(self.feedback_file):
                return feedback_list
            
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        feedback = json.loads(line.strip())
                        if feedback.get('query_id') == query_id:
                            feedback_list.append(feedback)
                    except json.JSONDecodeError:
                        continue
            
        except Exception as e:
            logger.error(f"Error getting file query feedback: {str(e)}")
        
        return feedback_list

    def get_training_data(self, min_feedback_count: int = 50) -> List[Dict[str, Any]]:
        """Get feedback data formatted for LTR training"""
        try:
            if self.feedback_collection is not None:
                # Get feedback from MongoDB
                feedback_list = list(self.feedback_collection.find(
                    {},
                    {'_id': 0}
                ).limit(min_feedback_count * 2))  # Get more to ensure we have enough
                return feedback_list
            else:
                # Get from file
                feedback_list = []
                if os.path.exists(self.feedback_file):
                    with open(self.feedback_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                feedback = json.loads(line.strip())
                                feedback_list.append(feedback)
                                if len(feedback_list) >= min_feedback_count * 2:
                                    break
                            except json.JSONDecodeError:
                                continue
                return feedback_list
                
        except Exception as e:
            logger.error(f"Error getting training data: {str(e)}")
            return []

    def get_all_feedback(self) -> List[Dict[str, Any]]:
        """Get all feedback data"""
        try:
            if self.feedback_collection is not None:
                # Get all feedback from MongoDB
                feedback_list = list(self.feedback_collection.find({}, {'_id': 0}))
                return feedback_list
            else:
                # Get all from file
                feedback_list = []
                if os.path.exists(self.feedback_file):
                    with open(self.feedback_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                feedback = json.loads(line.strip())
                                feedback_list.append(feedback)
                            except json.JSONDecodeError:
                                continue
                return feedback_list
                
        except Exception as e:
            logger.error(f"Error getting all feedback: {str(e)}")
            return []

    def get_recent_feedback(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent feedback entries"""
        try:
            if self.feedback_collection is not None:
                # Get from MongoDB, sorted by submission time
                feedback_list = list(self.feedback_collection.find(
                    {},
                    {'_id': 0}
                ).sort('submitted_at', -1).limit(limit))
                return feedback_list
            else:
                # Get from file (this is less efficient for large files)
                all_feedback = []
                if os.path.exists(self.feedback_file):
                    with open(self.feedback_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                feedback = json.loads(line.strip())
                                all_feedback.append(feedback)
                            except json.JSONDecodeError:
                                continue
                
                # Sort by submission time and return recent ones
                all_feedback.sort(key=lambda x: x.get('submitted_at', ''), reverse=True)
                return all_feedback[:limit]
                
        except Exception as e:
            logger.error(f"Error getting recent feedback: {str(e)}")
            return []

    def get_user_feedback_stats(self, user_id: str) -> Dict[str, Any]:
        """Get feedback statistics for a specific user"""
        try:
            if self.feedback_collection is not None:
                # Get stats from MongoDB for specific user
                total_feedback = self.feedback_collection.count_documents({'user_id': user_id})
                
                if total_feedback == 0:
                    return {
                        'total_feedback': 0,
                        'positive_feedback': 0,
                        'negative_feedback': 0,
                        'average_rating': 0,
                        'storage_type': 'mongodb'
                    }
                
                # Aggregate rating statistics for this user
                pipeline = [
                    {"$match": {"user_id": user_id}},
                    {"$group": {
                        "_id": None,
                        "avg_rating": {"$avg": "$rating"},
                        "total_positive": {"$sum": {"$cond": [{"$gte": ["$rating", 4]}, 1, 0]}},
                        "total_negative": {"$sum": {"$cond": [{"$lte": ["$rating", 2]}, 1, 0]}}
                    }}
                ]
                
                stats_result = list(self.feedback_collection.aggregate(pipeline))
                if stats_result:
                    stats = stats_result[0]
                    return {
                        'total_feedback': total_feedback,
                        'average_rating': round(stats.get('avg_rating', 0), 2),
                        'positive_feedback': stats.get('total_positive', 0),
                        'negative_feedback': stats.get('total_negative', 0),
                        'storage_type': 'mongodb'
                    }
            
            # Fallback to file-based stats for specific user
            return self._get_file_user_feedback_stats(user_id)
            
        except Exception as e:
            logger.error(f"Error getting user feedback stats: {str(e)}")
            return {
                'total_feedback': 0,
                'average_rating': 0,
                'positive_feedback': 0,
                'negative_feedback': 0,
                'storage_type': 'error'
            }

    def _get_file_user_feedback_stats(self, user_id: str) -> Dict[str, Any]:
        """Get feedback statistics for a specific user from file storage"""
        try:
            if not os.path.exists(self.feedback_file):
                return {
                    'total_feedback': 0,
                    'average_rating': 0,
                    'positive_feedback': 0,
                    'negative_feedback': 0,
                    'storage_type': 'file'
                }
            
            ratings = []
            positive_count = 0
            negative_count = 0
            
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        feedback = json.loads(line.strip())
                        # Only count feedback from this user
                        if feedback.get('user_id') == user_id:
                            rating = feedback.get('rating', 0)
                            ratings.append(rating)
                            
                            if rating >= 4:
                                positive_count += 1
                            elif rating <= 2:
                                negative_count += 1
                    except json.JSONDecodeError:
                        continue
            
            avg_rating = sum(ratings) / len(ratings) if ratings else 0
            
            return {
                'total_feedback': len(ratings),
                'average_rating': round(avg_rating, 2),
                'positive_feedback': positive_count,
                'negative_feedback': negative_count,
                'storage_type': 'file'
            }
            
        except Exception as e:
            logger.error(f"Error getting file user feedback stats: {str(e)}")
            return {
                'total_feedback': 0,
                'average_rating': 0,
                'positive_feedback': 0,
                'negative_feedback': 0,
                'storage_type': 'file_error'
            } 