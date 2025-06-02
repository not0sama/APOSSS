#!/usr/bin/env python3
"""
User Management System for APOSSS
Handles authentication, user profiles, and interaction tracking
"""
import logging
import hashlib
import secrets
import jwt
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from werkzeug.security import generate_password_hash, check_password_hash
from bson import ObjectId
from .database_manager import DatabaseManager

logger = logging.getLogger(__name__)

class UserManager:
    """Comprehensive user management with authentication and personalization"""
    
    def __init__(self, db_manager: DatabaseManager, secret_key: str = None):
        """Initialize user manager"""
        self.db_manager = db_manager
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        
        # Connect to APOSSS database for user management
        try:
            if hasattr(db_manager, 'get_database'):
                self.aposss_db = db_manager.get_database('aposss')
                logger.info("Connected to APOSSS database for user management")
            else:
                # Fallback: create direct connection
                from pymongo import MongoClient
                client = MongoClient('mongodb://localhost:27017/')
                self.aposss_db = client['APOSSS']
                logger.info("Created direct connection to APOSSS database")
                
        except Exception as e:
            logger.error(f"Failed to connect to APOSSS database: {e}")
            raise
    
    def register_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new user"""
        try:
            # Validate required fields
            required_fields = ['first_name', 'last_name', 'email', 'username', 'password']
            for field in required_fields:
                if field not in user_data or not user_data[field]:
                    return {
                        'success': False,
                        'error': f'Missing required field: {field}'
                    }
            
            # Check if email or username already exists
            users_collection = self.aposss_db['users']
            
            existing_email = users_collection.find_one({'email': user_data['email'].lower()})
            if existing_email:
                return {
                    'success': False,
                    'error': 'Email address already registered'
                }
            
            existing_username = users_collection.find_one({'username': user_data['username']})
            if existing_username:
                return {
                    'success': False,
                    'error': 'Username already taken'
                }
            
            # Hash password
            password_hash = generate_password_hash(user_data['password'])
            
            # Prepare user document
            user_doc = {
                'first_name': user_data['first_name'].strip(),
                'last_name': user_data['last_name'].strip(),
                'email': user_data['email'].lower().strip(),
                'username': user_data['username'].strip(),
                'password_hash': password_hash,
                'organization': user_data.get('organization', '').strip(),
                'department': user_data.get('department', '').strip(),
                'role': user_data.get('role', '').strip(),
                'country': user_data.get('country', '').strip(),
                'phone': user_data.get('phone', '').strip(),
                'website': user_data.get('website', '').strip(),
                'research_interests': user_data.get('research_interests', []),
                'preferences': {
                    'language': 'english',
                    'results_per_page': 20,
                    'email_notifications': True
                },
                'profile_visibility': 'public',
                'is_verified': False,
                'is_active': True,
                'created_at': datetime.now(),
                'updated_at': datetime.now(),
                'last_login': None,
                'login_count': 0,
                'interaction_stats': {
                    'total_searches': 0,
                    'total_feedback_given': 0,
                    'total_login_count': 0,
                    'avg_session_duration': 0
                }
            }
            
            # Insert user
            result = users_collection.insert_one(user_doc)
            user_id = str(result.inserted_id)
            
            logger.info(f"New user registered: {user_data['username']} (ID: {user_id})")
            
            return {
                'success': True,
                'message': 'User registered successfully',
                'user_id': user_id
            }
            
        except Exception as e:
            logger.error(f"Error registering user: {e}")
            return {
                'success': False,
                'error': 'Registration failed due to server error'
            }
    
    def authenticate_user(self, login_identifier: str, password: str) -> Dict[str, Any]:
        """Authenticate user login"""
        try:
            users_collection = self.aposss_db['users']
            
            # Find user by email or username
            user = users_collection.find_one({
                '$or': [
                    {'email': login_identifier.lower()},
                    {'username': login_identifier}
                ]
            })
            
            if not user:
                return {
                    'success': False,
                    'error': 'Invalid login credentials'
                }
            
            if not user.get('is_active', True):
                return {
                    'success': False,
                    'error': 'Account is deactivated'
                }
            
            # Check password
            if not check_password_hash(user['password_hash'], password):
                return {
                    'success': False,
                    'error': 'Invalid login credentials'
                }
            
            # Update login statistics
            users_collection.update_one(
                {'_id': user['_id']},
                {
                    '$set': {
                        'last_login': datetime.now(),
                        'updated_at': datetime.now()
                    },
                    '$inc': {
                        'login_count': 1,
                        'interaction_stats.total_login_count': 1
                    }
                }
            )
            
            # Generate JWT token
            token_payload = {
                'user_id': str(user['_id']),
                'username': user['username'],
                'email': user['email'],
                'exp': datetime.utcnow() + timedelta(days=7)  # Token expires in 7 days
            }
            
            token = jwt.encode(token_payload, self.secret_key, algorithm='HS256')
            
            # Prepare user data for response (remove sensitive info)
            user_data = {
                '_id': str(user['_id']),
                'username': user['username'],
                'email': user['email'],
                'first_name': user['first_name'],
                'last_name': user['last_name'],
                'organization': user.get('organization', ''),
                'department': user.get('department', ''),
                'role': user.get('role', ''),
                'country': user.get('country', ''),
                'research_interests': user.get('research_interests', []),
                'preferences': user.get('preferences', {}),
                'created_at': user['created_at'].isoformat() if user.get('created_at') else None,
                'last_login': datetime.now().isoformat()
            }
            
            logger.info(f"User authenticated: {user['username']}")
            
            return {
                'success': True,
                'message': 'Login successful',
                'token': token,
                'user': user_data
            }
            
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return {
                'success': False,
                'error': 'Authentication failed due to server error'
            }
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token and return user data"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            user_id = payload['user_id']
            
            # Get fresh user data from database
            users_collection = self.aposss_db['users']
            user = users_collection.find_one({'_id': user_id})
            
            if not user or not user.get('is_active', True):
                return {
                    'success': False,
                    'error': 'User not found or deactivated'
                }
            
            # Prepare user data for response
            user_data = {
                '_id': str(user['_id']),
                'username': user['username'],
                'email': user['email'],
                'first_name': user['first_name'],
                'last_name': user['last_name'],
                'organization': user.get('organization', ''),
                'department': user.get('department', ''),
                'role': user.get('role', ''),
                'country': user.get('country', ''),
                'research_interests': user.get('research_interests', []),
                'preferences': user.get('preferences', {}),
                'created_at': user['created_at'].isoformat() if user.get('created_at') else None,
                'last_login': user.get('last_login').isoformat() if user.get('last_login') else None
            }
            
            return {
                'success': True,
                'user': user_data
            }
            
        except jwt.ExpiredSignatureError:
            return {
                'success': False,
                'error': 'Token has expired'
            }
        except jwt.InvalidTokenError:
            return {
                'success': False,
                'error': 'Invalid token'
            }
        except Exception as e:
            logger.error(f"Error verifying token: {e}")
            return {
                'success': False,
                'error': 'Token verification failed'
            }
    
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile data"""
        try:
            users_collection = self.aposss_db['users']
            user = users_collection.find_one({'_id': ObjectId(user_id)})
            
            if not user:
                return {
                    'success': False,
                    'error': 'User not found'
                }
            
            # Get interaction statistics
            interactions_collection = self.aposss_db['user_interactions']
            interaction_stats = interactions_collection.aggregate([
                {'$match': {'user_id': user_id}},
                {'$group': {
                    '_id': None,
                    'total_searches': {
                        '$sum': {'$cond': [{'$eq': ['$interaction_type', 'search']}, 1, 0]}
                    },
                    'total_feedback_given': {
                        '$sum': {'$cond': [{'$eq': ['$interaction_type', 'feedback']}, 1, 0]}
                    },
                    'total_interactions': {'$sum': 1}
                }}
            ])
            
            stats = list(interaction_stats)
            if stats:
                user['interaction_stats'] = stats[0]
            
            # Prepare response
            user_data = {
                '_id': str(user['_id']),
                'username': user['username'],
                'email': user['email'],
                'first_name': user['first_name'],
                'last_name': user['last_name'],
                'organization': user.get('organization', ''),
                'department': user.get('department', ''),
                'role': user.get('role', ''),
                'country': user.get('country', ''),
                'phone': user.get('phone', ''),
                'website': user.get('website', ''),
                'research_interests': user.get('research_interests', []),
                'preferences': user.get('preferences', {}),
                'profile_visibility': user.get('profile_visibility', 'public'),
                'created_at': user['created_at'].isoformat() if user.get('created_at') else None,
                'last_login': user.get('last_login').isoformat() if user.get('last_login') else None,
                'login_count': user.get('login_count', 0),
                'interaction_stats': user.get('interaction_stats', {})
            }
            
            return {
                'success': True,
                'user': user_data
            }
            
        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
            return {
                'success': False,
                'error': 'Failed to get profile'
            }
    
    def update_user_profile(self, user_id: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update user profile"""
        try:
            users_collection = self.aposss_db['users']
            
            # Remove sensitive fields that shouldn't be updated this way
            forbidden_fields = ['_id', 'password_hash', 'email', 'username', 'created_at', 'login_count']
            clean_data = {k: v for k, v in update_data.items() if k not in forbidden_fields}
            
            clean_data['updated_at'] = datetime.now()
            
            result = users_collection.update_one(
                {'_id': ObjectId(user_id)},
                {'$set': clean_data}
            )
            
            if result.modified_count > 0:
                return {
                    'success': True,
                    'message': 'Profile updated successfully'
                }
            else:
                return {
                    'success': False,
                    'error': 'No changes made to profile'
                }
                
        except Exception as e:
            logger.error(f"Error updating user profile: {e}")
            return {
                'success': False,
                'error': 'Failed to update profile'
            }
    
    def track_user_interaction(self, user_id: str, interaction_type: str, interaction_data: Dict[str, Any]):
        """Track user interaction for analytics and personalization"""
        try:
            if not user_id:
                # Anonymous user - still track for general analytics
                user_id = 'anonymous'
            
            interactions_collection = self.aposss_db['user_interactions']
            
            interaction_doc = {
                'user_id': user_id,
                'interaction_type': interaction_type,
                'timestamp': datetime.now(),
                'data': interaction_data
            }
            
            interactions_collection.insert_one(interaction_doc)
            
            # Update user statistics if not anonymous
            if user_id != 'anonymous':
                users_collection = self.aposss_db['users']
                
                update_operations = {'$set': {'updated_at': datetime.now()}}
                
                if interaction_type == 'search':
                    update_operations['$inc'] = {'interaction_stats.total_searches': 1}
                elif interaction_type == 'feedback':
                    update_operations['$inc'] = {'interaction_stats.total_feedback_given': 1}
                
                users_collection.update_one(
                    {'_id': ObjectId(user_id)},
                    update_operations
                )
            
        except Exception as e:
            logger.error(f"Error tracking user interaction: {e}")
    
    def get_user_recommendations(self, user_id: str) -> Dict[str, Any]:
        """Generate personalized recommendations for user"""
        try:
            users_collection = self.aposss_db['users']
            interactions_collection = self.aposss_db['user_interactions']
            
            user = users_collection.find_one({'_id': user_id})
            if not user:
                return {
                    'success': False,
                    'error': 'User not found'
                }
            
            # Get user's search history
            search_interactions = list(interactions_collection.find({
                'user_id': user_id,
                'interaction_type': 'search'
            }).sort('timestamp', -1).limit(50))
            
            # Extract search terms and patterns
            search_terms = []
            for interaction in search_interactions:
                query = interaction.get('data', {}).get('query', '')
                if query:
                    search_terms.append(query.lower())
            
            # Generate recommendations based on research interests
            research_interests = user.get('research_interests', [])
            
            # Basic topic suggestions (could be enhanced with ML)
            suggested_topics = []
            if 'AI' in research_interests or 'artificial intelligence' in search_terms:
                suggested_topics.extend(['machine learning', 'deep learning', 'neural networks'])
            if 'machine learning' in research_interests:
                suggested_topics.extend(['supervised learning', 'unsupervised learning', 'reinforcement learning'])
            if 'computer science' in user.get('department', '').lower():
                suggested_topics.extend(['algorithms', 'data structures', 'software engineering'])
            
            # Remove duplicates and limit
            suggested_topics = list(set(suggested_topics))[:10]
            
            # Analyze preferred resource types from feedback
            feedback_interactions = list(interactions_collection.find({
                'user_id': user_id,
                'interaction_type': 'feedback'
            }))
            
            preferred_types = []
            for feedback in feedback_interactions:
                rating = feedback.get('data', {}).get('rating', 0)
                if rating >= 4:  # High ratings
                    # Could analyze result types that received high ratings
                    preferred_types.append('research_papers')  # Placeholder
            
            # Create personalized search suggestions
            personalized_suggestions = []
            for interest in research_interests[:5]:
                personalized_suggestions.append(f"{interest} applications")
                personalized_suggestions.append(f"recent advances in {interest}")
            
            recommendations = {
                'suggested_topics': suggested_topics,
                'personalized_search_suggestions': personalized_suggestions[:8],
                'preferred_resource_types': list(set(preferred_types))[:5],
                'research_interests': research_interests,
                'based_on_searches': len(search_interactions),
                'based_on_feedback': len(feedback_interactions)
            }
            
            return {
                'success': True,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return {
                'success': False,
                'error': 'Failed to generate recommendations'
            }
    
    def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get user analytics and statistics"""
        try:
            users_collection = self.aposss_db['users']
            interactions_collection = self.aposss_db['user_interactions']
            
            user = users_collection.find_one({'_id': user_id})
            if not user:
                return {
                    'success': False,
                    'error': 'User not found'
                }
            
            # Get interaction statistics
            total_interactions = interactions_collection.count_documents({'user_id': user_id})
            search_count = interactions_collection.count_documents({
                'user_id': user_id,
                'interaction_type': 'search'
            })
            feedback_count = interactions_collection.count_documents({
                'user_id': user_id,
                'interaction_type': 'feedback'
            })
            
            # Get recent activity
            recent_activity = list(interactions_collection.find({
                'user_id': user_id
            }).sort('timestamp', -1).limit(20))
            
            # Format recent activity
            formatted_activity = []
            for activity in recent_activity:
                formatted_activity.append({
                    'interaction_type': activity['interaction_type'],
                    'timestamp': activity['timestamp'].isoformat(),
                    'summary': self._format_activity_summary(activity)
                })
            
            analytics = {
                'total_interactions': total_interactions,
                'search_count': search_count,
                'feedback_count': feedback_count,
                'user_stats': {
                    'total_login_count': user.get('login_count', 0),
                    'profile_completion': self._calculate_profile_completion(user)
                },
                'member_since': user.get('created_at').isoformat() if user.get('created_at') else None,
                'last_active': user.get('last_login').isoformat() if user.get('last_login') else None,
                'recent_activity': formatted_activity
            }
            
            return {
                'success': True,
                'analytics': analytics
            }
            
        except Exception as e:
            logger.error(f"Error getting user analytics: {e}")
            return {
                'success': False,
                'error': 'Failed to get analytics'
            }
    
    def _format_activity_summary(self, activity: Dict[str, Any]) -> str:
        """Format activity summary for display"""
        activity_type = activity['interaction_type']
        data = activity.get('data', {})
        
        if activity_type == 'search':
            query = data.get('query', 'Unknown query')[:50]
            return f"Searched for: {query}"
        elif activity_type == 'feedback':
            rating = data.get('rating', 0)
            return f"Provided {rating}-star feedback"
        elif activity_type == 'login':
            return "Logged in"
        elif activity_type == 'logout':
            return "Logged out"
        else:
            return f"{activity_type.title()} activity"
    
    def _calculate_profile_completion(self, user: Dict[str, Any]) -> int:
        """Calculate profile completion percentage"""
        fields_to_check = [
            'first_name', 'last_name', 'email', 'username',
            'organization', 'department', 'role', 'country'
        ]
        
        completed_fields = 0
        for field in fields_to_check:
            if user.get(field) and str(user[field]).strip():
                completed_fields += 1
        
        # Bonus for research interests
        if user.get('research_interests') and len(user['research_interests']) > 0:
            completed_fields += 1
            fields_to_check.append('research_interests')
        
        return int((completed_fields / len(fields_to_check)) * 100) 