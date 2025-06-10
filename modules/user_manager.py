#!/usr/bin/env python3
"""
User Management System for APOSSS
Handles user authentication, profiles, and interaction tracking
"""
import logging
import hashlib
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pymongo.errors import DuplicateKeyError
import jwt
import bcrypt

logger = logging.getLogger(__name__)

class UserManager:
    """Manages user authentication, profiles, and interactions"""
    
    def __init__(self, db_manager=None):
        """Initialize user manager"""
        self.db_manager = db_manager
        self.users_collection = None
        self.interactions_collection = None
        self.preferences_collection = None
        self.sessions_collection = None
        
        # JWT secret key (in production, use environment variable)
        self.jwt_secret = "aposss_secret_key_2024"  # Change this in production
        self.jwt_algorithm = "HS256"
        self.token_expiry_hours = 24
        
        # Initialize collections
        self._initialize_collections()
        logger.info("User Manager initialized successfully")
    
    def _initialize_collections(self):
        """Initialize MongoDB collections for user management"""
        try:
            if self.db_manager:
                aposss_db = self.db_manager.get_database('aposss')
                if aposss_db is not None:
                    self.users_collection = aposss_db['users']
                    self.interactions_collection = aposss_db['user_interactions']
                    self.preferences_collection = aposss_db['user_preferences']
                    self.sessions_collection = aposss_db['user_sessions']
                    
                    # Create indexes for better performance
                    self._create_indexes()
                    logger.info("User management collections initialized")
                    return
        except Exception as e:
            logger.warning(f"Could not initialize user management collections: {str(e)}")
        
        logger.warning("Using fallback mode - user management features limited")
    
    def _create_indexes(self):
        """Create database indexes for user collections"""
        try:
            # Users collection indexes
            self.users_collection.create_index("email", unique=True)
            self.users_collection.create_index("username", unique=True)
            self.users_collection.create_index("user_id", unique=True)
            
            # Interactions collection indexes
            self.interactions_collection.create_index("user_id")
            self.interactions_collection.create_index("timestamp")
            self.interactions_collection.create_index([("user_id", 1), ("timestamp", -1)])
            
            # Preferences collection indexes
            self.preferences_collection.create_index("user_id", unique=True)
            
            # Sessions collection indexes
            self.sessions_collection.create_index("user_id")
            self.sessions_collection.create_index("expires_at")
            
            logger.info("Database indexes created successfully")
        except Exception as e:
            logger.warning(f"Error creating indexes: {str(e)}")
    
    def register_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a new user
        
        Args:
            user_data: Dictionary containing user registration information
                
        Returns:
            Registration result with success status and user info
        """
        try:
            # Validate required fields
            required_fields = ['username', 'email', 'password']
            for field in required_fields:
                if field not in user_data or not user_data[field]:
                    return {'success': False, 'error': f'Missing required field: {field}'}
            
            # Check if user already exists
            if self._user_exists(user_data['email'], user_data['username']):
                return {'success': False, 'error': 'User with this email or username already exists'}
            
            # Hash password
            password_hash = bcrypt.hashpw(user_data['password'].encode('utf-8'), bcrypt.gensalt())
            
            # Generate unique user ID
            user_id = str(uuid.uuid4())
            
            # Create user document
            user_doc = {
                'user_id': user_id,
                'username': user_data['username'].lower().strip(),
                'email': user_data['email'].lower().strip(),
                'password_hash': password_hash,
                'profile': {
                    'first_name': user_data.get('first_name', ''),
                    'last_name': user_data.get('last_name', ''),
                    'academic_fields': user_data.get('academic_fields', []),
                    'institution': user_data.get('institution', ''),
                    'role': user_data.get('role', 'researcher'),
                    'bio': user_data.get('bio', ''),
                    'avatar_url': user_data.get('avatar_url', '')
                },
                'preferences': {
                    'search_preferences': {
                        'preferred_resource_types': ['article', 'book', 'expert'],
                        'preferred_databases': [],
                        'language_preference': 'en',
                        'results_per_page': 20
                    },
                    'notification_preferences': {
                        'email_notifications': True,
                        'search_alerts': False,
                        'feedback_requests': True
                    },
                    'privacy_settings': {
                        'profile_visibility': 'public',
                        'interaction_tracking': True,
                        'personalized_recommendations': True
                    }
                },
                'statistics': {
                    'total_searches': 0,
                    'total_feedback': 0,
                    'average_rating': 0.0,
                    'favorite_topics': [],
                    'most_used_resources': []
                },
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'last_login': None,
                'is_active': True,
                'email_verified': False,
                'login_count': 0
            }
            
            # Insert user into database
            if self.users_collection is not None:
                result = self.users_collection.insert_one(user_doc)
                user_doc['_id'] = str(result.inserted_id)
            
            # Create initial user preferences
            self._create_user_preferences(user_id)
            
            # Remove password hash from response
            user_doc.pop('password_hash', None)
            
            logger.info(f"User registered successfully: {user_data['username']}")
            
            return {
                'success': True,
                'user': user_doc,
                'message': 'User registered successfully'
            }
            
        except DuplicateKeyError:
            return {'success': False, 'error': 'User with this email or username already exists'}
        except Exception as e:
            logger.error(f"Error registering user: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def authenticate_user(self, identifier: str, password: str) -> Dict[str, Any]:
        """Authenticate user login"""
        try:
            if self.users_collection is None:
                return {'success': False, 'error': 'User authentication not available'}
            
            # Find user by email or username
            user = self.users_collection.find_one({
                '$or': [
                    {'email': identifier.lower().strip()},
                    {'username': identifier.lower().strip()}
                ],
                'is_active': True
            })
            
            if not user:
                return {'success': False, 'error': 'Invalid credentials'}
            
            # Verify password
            if not bcrypt.checkpw(password.encode('utf-8'), user['password_hash']):
                return {'success': False, 'error': 'Invalid credentials'}
            
            # Generate JWT token
            token = self._generate_token(user['user_id'])
            
            # Update login statistics
            self.users_collection.update_one(
                {'user_id': user['user_id']},
                {
                    '$set': {
                        'last_login': datetime.now().isoformat(),
                        'updated_at': datetime.now().isoformat()
                    },
                    '$inc': {'login_count': 1}
                }
            )
            
            # Create session record
            self._create_session(user['user_id'], token)
            
            # Remove password hash from response
            user.pop('password_hash', None)
            user.pop('_id', None)
            
            logger.info(f"User authenticated successfully: {user['username']}")
            
            return {
                'success': True,
                'token': token,
                'user': user,
                'message': 'Authentication successful'
            }
            
        except Exception as e:
            logger.error(f"Error authenticating user: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token and return user info"""
        try:
            # Decode token
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            user_id = payload.get('user_id')
            
            if not user_id:
                return {'success': False, 'error': 'Invalid token'}
            
            # Get user from database
            user = self.get_user_by_id(user_id)
            if not user:
                return {'success': False, 'error': 'User not found'}
            
            return {'success': True, 'user': user}
            
        except jwt.ExpiredSignatureError:
            return {'success': False, 'error': 'Token expired'}
        except jwt.InvalidTokenError:
            return {'success': False, 'error': 'Invalid token'}
        except Exception as e:
            logger.error(f"Error verifying token: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by user ID"""
        try:
            if self.users_collection is None:
                return None
            
            user = self.users_collection.find_one(
                {'user_id': user_id, 'is_active': True},
                {'password_hash': 0, '_id': 0}
            )
            return user
        except Exception as e:
            logger.error(f"Error getting user by ID: {str(e)}")
            return None
    
    def update_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update user profile information"""
        try:
            if self.users_collection is None:
                return {'success': False, 'error': 'User database not available'}
            
            # Validate user exists
            user = self.users_collection.find_one({'user_id': user_id, 'is_active': True})
            if not user:
                return {'success': False, 'error': 'User not found'}
            
            # Prepare update data
            update_data = {
                'updated_at': datetime.now().isoformat()
            }
            
            # Update profile fields
            for field, value in profile_data.items():
                if field in ['first_name', 'last_name', 'institution', 'role', 'bio', 'academic_fields', 'avatar_url']:
                    update_data[f'profile.{field}'] = value
            
            # Perform update
            result = self.users_collection.update_one(
                {'user_id': user_id},
                {'$set': update_data}
            )
            
            if result.modified_count > 0:
                # Get updated user data
                updated_user = self.get_user_by_id(user_id)
                logger.info(f"Profile updated successfully for user: {user_id}")
                return {
                    'success': True,
                    'user': updated_user,
                    'message': 'Profile updated successfully'
                }
            else:
                return {'success': False, 'error': 'No changes made to profile'}
                
        except Exception as e:
            logger.error(f"Error updating user profile: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def track_user_interaction(self, user_id: str, interaction_data: Dict[str, Any]) -> bool:
        """Track user interaction for personalization"""
        try:
            # Create interaction record
            interaction = {
                'user_id': user_id,
                'action': interaction_data.get('action', 'unknown'),
                'query': interaction_data.get('query', ''),
                'result_id': interaction_data.get('result_id', ''),
                'session_id': interaction_data.get('session_id', ''),
                'timestamp': datetime.now().isoformat(),
                'metadata': interaction_data.get('metadata', {})
            }
            
            # Store interaction
            if self.interactions_collection is not None:
                self.interactions_collection.insert_one(interaction)
            
            # Update user statistics if registered user
            if user_id and not user_id.startswith('anon_'):
                self._update_user_statistics(user_id, interaction_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error tracking user interaction: {str(e)}")
            return False
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences for personalization"""
        try:
            if not user_id or user_id.startswith('anon_'):
                return self._get_default_preferences()
            
            if self.preferences_collection is None:
                return self._get_default_preferences()
            
            prefs = self.preferences_collection.find_one({'user_id': user_id})
            if prefs:
                prefs.pop('_id', None)
                return prefs
            
            return self._create_user_preferences(user_id)
            
        except Exception as e:
            logger.error(f"Error getting user preferences: {str(e)}")
            return self._get_default_preferences()
    
    def get_personalization_data(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user data for personalization"""
        try:
            # Get user preferences
            preferences = self.get_user_preferences(user_id)
            
            # Get interaction history
            interactions = self.get_user_interaction_history(user_id, 50)
            
            # Analyze interaction patterns
            interaction_patterns = self._analyze_interaction_patterns(interactions)
            
            # Get user profile if registered
            user_profile = None
            if user_id and not user_id.startswith('anon_'):
                user = self.get_user_by_id(user_id)
                if user:
                    user_profile = user.get('profile', {})
            
            return {
                'user_id': user_id,
                'preferences': preferences,
                'interaction_patterns': interaction_patterns,
                'profile': user_profile,
                'is_registered': not (user_id.startswith('anon_') if user_id else True)
            }
            
        except Exception as e:
            logger.error(f"Error getting personalization data: {str(e)}")
            return {
                'user_id': user_id,
                'preferences': self._get_default_preferences(),
                'interaction_patterns': {},
                'profile': None,
                'is_registered': False
            }
    
    def get_user_interaction_history(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get user interaction history for analysis"""
        try:
            if self.interactions_collection is None or not user_id:
                return []
            
            interactions = list(self.interactions_collection.find(
                {'user_id': user_id},
                {'_id': 0}
            ).sort('timestamp', -1).limit(limit))
            
            return interactions
            
        except Exception as e:
            logger.error(f"Error getting user interaction history: {str(e)}")
            return []
    
    def generate_anonymous_user_id(self) -> str:
        """Generate anonymous user ID for session tracking"""
        return f"anon_{uuid.uuid4().hex[:12]}"
    
    def _user_exists(self, email: str, username: str) -> bool:
        """Check if user exists"""
        if self.users_collection is None:
            return False
        
        user = self.users_collection.find_one({
            '$or': [
                {'email': email.lower().strip()},
                {'username': username.lower().strip()}
            ]
        })
        return user is not None
    
    def _generate_token(self, user_id: str) -> str:
        """Generate JWT token"""
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def _create_session(self, user_id: str, token: str):
        """Create user session record"""
        try:
            if self.sessions_collection is not None:
                session = {
                    'user_id': user_id,
                    'token': token,
                    'created_at': datetime.now().isoformat(),
                    'expires_at': (datetime.now() + timedelta(hours=self.token_expiry_hours)).isoformat(),
                    'is_active': True
                }
                self.sessions_collection.insert_one(session)
        except Exception as e:
            logger.warning(f"Error creating session: {str(e)}")
    
    def _create_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Create default user preferences"""
        try:
            preferences = {
                'user_id': user_id,
                'search_preferences': {
                    'preferred_resource_types': [],
                    'preferred_databases': [],
                    'language_preference': 'en',
                    'results_per_page': 20,
                    'sort_preference': 'relevance'
                },
                'ranking_preferences': {
                    'weight_recency': 0.2,
                    'weight_relevance': 0.4,
                    'weight_authority': 0.2,
                    'weight_user_feedback': 0.2
                },
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            if self.preferences_collection is not None:
                self.preferences_collection.insert_one(preferences.copy())
            
            preferences.pop('_id', None)
            return preferences
            
        except Exception as e:
            logger.error(f"Error creating user preferences: {str(e)}")
            return self._get_default_preferences()
    
    def _get_default_preferences(self) -> Dict[str, Any]:
        """Get default preferences"""
        return {
            'search_preferences': {
                'preferred_resource_types': [],
                'preferred_databases': [],
                'language_preference': 'en',
                'results_per_page': 20,
                'sort_preference': 'relevance'
            },
            'ranking_preferences': {
                'weight_recency': 0.2,
                'weight_relevance': 0.4,
                'weight_authority': 0.2,
                'weight_user_feedback': 0.2
            }
        }
    
    def _update_user_statistics(self, user_id: str, interaction_data: Dict[str, Any]):
        """Update user statistics"""
        try:
            if self.users_collection is None:
                return
            
            action = interaction_data.get('action', '')
            update_data = {'updated_at': datetime.now().isoformat()}
            
            if action == 'search':
                update_data['$inc'] = {'statistics.total_searches': 1}
            elif action == 'feedback':
                update_data['$inc'] = {'statistics.total_feedback': 1}
            
            self.users_collection.update_one(
                {'user_id': user_id},
                {'$set': update_data}
            )
            
        except Exception as e:
            logger.warning(f"Error updating user statistics: {str(e)}")
    
    def _analyze_interaction_patterns(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze interaction patterns"""
        try:
            patterns = {
                'frequent_queries': [],
                'preferred_result_types': [],
                'search_frequency': 0,
                'feedback_tendency': 0
            }
            
            if not interactions:
                return patterns
            
            # Analyze queries
            queries = [i.get('query', '') for i in interactions if i.get('action') == 'search' and i.get('query')]
            query_counter = {}
            for query in queries:
                query_counter[query] = query_counter.get(query, 0) + 1
            
            patterns['frequent_queries'] = sorted(query_counter.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing interaction patterns: {str(e)}")
            return {
                'frequent_queries': [],
                'preferred_result_types': [],
                'search_frequency': 0,
                'feedback_tendency': 0
            }
    
    def get_user_personalization_data(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive personalization data for a user"""
        try:
            if self.users_collection is None:
                return None
            
            # Get user profile
            user_profile = self.users_collection.find_one({'user_id': user_id})
            if not user_profile:
                return None
            
            # Get user interactions
            interaction_history = list(self.interactions_collection.find(
                {'user_id': user_id}
            ).sort('timestamp', -1).limit(100))  # Last 100 interactions
            
            # Get user preferences
            preferences = self.preferences_collection.find_one({'user_id': user_id})
            if not preferences:
                preferences = self._generate_default_preferences()
            
            # Build personalization data
            personalization_data = {
                'profile': {
                    'name': user_profile.get('name', ''),
                    'institution': user_profile.get('institution', ''),
                    'role': user_profile.get('role', ''),
                    'academic_fields': user_profile.get('academic_fields', []),
                    'languages': user_profile.get('bio', {}).get('languages', ['english'])
                },
                'preferences': preferences,
                'interaction_history': interaction_history,
                'stats': self._calculate_user_stats(interaction_history)
            }
            
            return personalization_data
            
        except Exception as e:
            logger.error(f"Error getting user personalization data: {e}")
            return None
    
    def get_anonymous_personalization_data(self, user_id: str) -> Dict[str, Any]:
        """Get basic personalization data for anonymous users"""
        try:
            if self.interactions_collection is None:
                return None
            
            # Get recent interactions for anonymous user
            interaction_history = list(self.interactions_collection.find(
                {'user_id': user_id}
            ).sort('timestamp', -1).limit(20))  # Last 20 interactions
            
            if not interaction_history:
                return None
            
            # Generate basic preferences from interaction patterns
            preferences = self._generate_anonymous_preferences(interaction_history)
            
            # Build minimal personalization data
            personalization_data = {
                'profile': {
                    'academic_fields': [],
                    'languages': ['english']
                },
                'preferences': preferences,
                'interaction_history': interaction_history,
                'stats': self._calculate_user_stats(interaction_history)
            }
            
            return personalization_data
            
        except Exception as e:
            logger.error(f"Error getting anonymous personalization data: {e}")
            return None
    
    def _generate_default_preferences(self) -> Dict[str, Any]:
        """Generate default user preferences"""
        return {
            'recency_preference': 0.5,      # Neutral preference for recent content
            'complexity_preference': 0.5,   # Neutral complexity preference
            'availability_preference': 0.7, # Slight preference for available content
            'language_preference': ['english'],
            'result_type_preferences': {
                'article': 0.5,
                'book': 0.5,
                'expert': 0.5,
                'equipment': 0.5,
                'material': 0.5,
                'project': 0.5
            }
        }
    
    def _generate_anonymous_preferences(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate preferences for anonymous users based on interaction patterns"""
        preferences = self._generate_default_preferences()
        
        if not interactions:
            return preferences
        
        # Analyze interaction patterns
        search_queries = []
        feedback_patterns = {}
        
        for interaction in interactions:
            if interaction.get('action') == 'search':
                search_queries.append(interaction.get('query', ''))
            elif interaction.get('action') == 'feedback':
                rating = interaction.get('metadata', {}).get('rating', 3)
                result_type = interaction.get('metadata', {}).get('result_type', 'unknown')
                if result_type not in feedback_patterns:
                    feedback_patterns[result_type] = []
                feedback_patterns[result_type].append(rating)
        
        # Update preferences based on feedback patterns
        for result_type, ratings in feedback_patterns.items():
            if len(ratings) > 0:
                avg_rating = sum(ratings) / len(ratings)
                # Convert rating (1-5) to preference (0-1)
                preference_score = (avg_rating - 1) / 4
                preferences['result_type_preferences'][result_type] = preference_score
        
        # Estimate recency preference from query patterns
        recent_queries = [q for q in search_queries if any(word in q.lower() 
                         for word in ['recent', 'latest', 'new', '2023', '2024'])]
        if len(recent_queries) > len(search_queries) * 0.3:  # 30% threshold
            preferences['recency_preference'] = 0.8
        
        return preferences
    
    def _calculate_user_stats(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate user interaction statistics"""
        stats = {
            'total_interactions': len(interactions),
            'searches_count': 0,
            'feedback_count': 0,
            'avg_session_length': 0,
            'most_searched_terms': [],
            'preferred_types': []
        }
        
        search_queries = []
        feedback_entries = []
        session_lengths = []
        
        for interaction in interactions:
            action = interaction.get('action', '')
            
            if action == 'search':
                stats['searches_count'] += 1
                search_queries.append(interaction.get('query', ''))
            elif action == 'feedback':
                stats['feedback_count'] += 1
                feedback_entries.append(interaction)
        
        # Analyze search terms
        if search_queries:
            all_terms = []
            for query in search_queries:
                all_terms.extend(query.lower().split())
            
            from collections import Counter
            term_counts = Counter(all_terms)
            stats['most_searched_terms'] = [term for term, count in term_counts.most_common(5)]
        
        # Analyze preferred types from feedback
        type_ratings = {}
        for feedback in feedback_entries:
            result_type = feedback.get('metadata', {}).get('result_type', '')
            rating = feedback.get('metadata', {}).get('rating', 3)
            
            if result_type:
                if result_type not in type_ratings:
                    type_ratings[result_type] = []
                type_ratings[result_type].append(rating)
        
        # Calculate average ratings per type
        type_preferences = []
        for result_type, ratings in type_ratings.items():
            avg_rating = sum(ratings) / len(ratings)
            if avg_rating >= 4.0:  # High-rated types
                type_preferences.append(result_type)
        
        stats['preferred_types'] = type_preferences
        
        return stats 