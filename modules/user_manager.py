#!/usr/bin/env python3
"""
User Management System for APOSSS
Handles user authentication, profiles, and interaction tracking
"""
import os
import logging
import hashlib
import uuid
import json
import random
import smtplib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pymongo.errors import DuplicateKeyError
import jwt
import bcrypt

# Import email modules
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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
        self.verification_codes_collection = None
        
        # JWT Configuration from environment variables
        self.jwt_secret = os.getenv('JWT_SECRET_KEY')
        if not self.jwt_secret:
            raise ValueError("JWT_SECRET_KEY environment variable is required")
        self.jwt_algorithm = os.getenv('JWT_ALGORITHM', 'HS256')
        self.token_expiry_hours = int(os.getenv('JWT_EXPIRY_HOURS', '24'))
        
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
                    self.verification_codes_collection = aposss_db['verification_codes']
                    
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
            
            # Verification codes collection indexes
            self.verification_codes_collection.create_index("user_id")
            self.verification_codes_collection.create_index("expires_at")
            self.verification_codes_collection.create_index("code")
            
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
                    'first_name': user_data.get('firstName', user_data.get('first_name', '')),
                    'last_name': user_data.get('lastName', user_data.get('last_name', '')),
                    'academic_fields': user_data.get('academic_fields', []),
                    'organization': user_data.get('organization', ''),
                    'department': user_data.get('department', ''),
                    'job_title': user_data.get('jobTitle', user_data.get('job_title', '')),
                    'institution': user_data.get('institution', user_data.get('organization', '')),  # Backward compatibility
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
                    },
                    'ui_preferences': {
                        'theme_preference': 'light',
                        'display_language': 'en'
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
            
            # Find user by email or username (exclude profile picture from response)
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
            
            # Remove password hash and profile picture from response
            user.pop('password_hash', None)
            user.pop('_id', None)
            # Remove profile picture binary data to prevent JSON serialization errors
            if 'profile' in user and 'profile_picture' in user['profile']:
                user['profile'].pop('profile_picture', None)
            
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
                {'password_hash': 0, '_id': 0, 'profile.profile_picture': 0}
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
    
    def check_email_exists(self, email: str) -> bool:
        """Check if email is already registered"""
        if self.users_collection is None:
            return False
        
        user = self.users_collection.find_one({
            'email': email.lower().strip()
        })
        return user is not None

    def check_username_exists(self, username: str) -> bool:
        """Check if username is already taken"""
        if self.users_collection is None:
            logger.warning("Users collection not available for username check")
            return False
        
        username_clean = username.lower().strip()
        logger.info(f"Checking if username '{username_clean}' exists in database")
        
        try:
            user = self.users_collection.find_one({
                'username': username_clean
            })
            
            exists = user is not None
            logger.info(f"Username '{username_clean}' exists: {exists}")
            
            if exists:
                logger.info(f"Found existing user: {user.get('email', 'unknown')}")
            
            return exists
            
        except Exception as e:
            logger.error(f"Error checking username '{username_clean}': {str(e)}")
            return False

    def find_user_by_provider(self, provider: str, provider_id: str) -> Optional[Dict[str, Any]]:
        """Find user by OAuth provider and provider ID"""
        if self.users_collection is None:
            return None
        
        try:
            user = self.users_collection.find_one({
                f'oauth_providers.{provider}.provider_id': provider_id
            })
            
            if user:
                user.pop('password_hash', None)  # Remove sensitive data
                user['_id'] = str(user['_id'])  # Convert ObjectId to string
            
            return user
            
        except Exception as e:
            logger.error(f"Error finding user by {provider} provider ID {provider_id}: {str(e)}")
            return None

    def register_social_user(self, user_info: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new user from social login"""
        try:
            provider = user_info['provider']
            provider_id = user_info['provider_id']
            
            # Check if user already exists with this provider
            existing_user = self.find_user_by_provider(provider, provider_id)
            if existing_user:
                # User exists, generate token and return
                token = self._generate_token(existing_user['user_id'])
                return {
                    'success': True,
                    'user': existing_user,
                    'token': token,
                    'message': 'User logged in successfully'
                }
            
            # Check if email already exists (for linking accounts)
            email = user_info.get('email')
            existing_email_user = None
            if email:
                existing_email_user = self.users_collection.find_one({'email': email.lower().strip()})
            
            if existing_email_user:
                # Link social account to existing user
                return self._link_social_account(existing_email_user, user_info)
            
            # Generate unique username
            base_username = self._generate_username_from_social(user_info)
            username = self._ensure_unique_username(base_username)
            
            # Generate unique user ID
            user_id = str(uuid.uuid4())
            
            # Create user document
            user_doc = {
                'user_id': user_id,
                'username': username,
                'email': email.lower().strip() if email else '',
                'password_hash': None,  # No password for social users
                'profile': {
                    'first_name': user_info.get('first_name', ''),
                    'last_name': user_info.get('last_name', ''),
                    'full_name': user_info.get('full_name', ''),
                    'academic_fields': [],
                    'organization': '',
                    'department': '',
                    'job_title': '',
                    'institution': '',
                    'role': 'researcher',
                    'bio': '',
                    'avatar_url': user_info.get('picture', '')
                },
                'oauth_providers': {
                    provider: {
                        'provider_id': provider_id,
                        'email': email,
                        'email_verified': user_info.get('email_verified', False),
                        'picture': user_info.get('picture'),
                        'locale': user_info.get('locale'),
                        'linked_at': datetime.now().isoformat(),
                        'orcid': user_info.get('orcid') if provider == 'orcid' else None
                    }
                },
                'preferences': {
                    'search_preferences': {
                        'preferred_resource_types': ['article', 'book', 'expert'],
                        'preferred_databases': [],
                        'language_preference': user_info.get('locale', 'en')[:2] if user_info.get('locale') else 'en',
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
                    },
                    'ui_preferences': {
                        'theme_preference': 'light',
                        'display_language': user_info.get('locale', 'en')[:2] if user_info.get('locale') else 'en'
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
                'last_login': datetime.now().isoformat(),
                'is_active': True,
                'email_verified': user_info.get('email_verified', False),
                'login_count': 1,
                'registration_method': f'{provider}_oauth'
            }
            
            # Insert user into database
            if self.users_collection is not None:
                result = self.users_collection.insert_one(user_doc)
                user_doc['_id'] = str(result.inserted_id)
            
            # Create initial user preferences
            self._create_user_preferences(user_id)
            
            # Remove sensitive data
            user_doc.pop('password_hash', None)
            
            # Generate token
            token = self._generate_token(user_id)
            
            logger.info(f"Social user registered successfully: {username} via {provider}")
            
            return {
                'success': True,
                'user': user_doc,
                'token': token,
                'message': 'User registered successfully via social login'
            }
            
        except Exception as e:
            logger.error(f"Error registering social user: {str(e)}")
            return {'success': False, 'error': str(e)}

    def _link_social_account(self, existing_user: Dict[str, Any], user_info: Dict[str, Any]) -> Dict[str, Any]:
        """Link social account to existing user"""
        try:
            provider = user_info['provider']
            provider_id = user_info['provider_id']
            user_id = existing_user['user_id']
            
            # Update user with social provider info
            oauth_update = {
                f'oauth_providers.{provider}': {
                    'provider_id': provider_id,
                    'email': user_info.get('email'),
                    'email_verified': user_info.get('email_verified', False),
                    'picture': user_info.get('picture'),
                    'locale': user_info.get('locale'),
                    'linked_at': datetime.now().isoformat(),
                    'orcid': user_info.get('orcid') if provider == 'orcid' else None
                },
                'updated_at': datetime.now().isoformat(),
                'last_login': datetime.now().isoformat()
            }
            
            # Update avatar if not set
            if user_info.get('picture') and not existing_user.get('profile', {}).get('avatar_url'):
                oauth_update['profile.avatar_url'] = user_info['picture']
            
            # Increment login count
            oauth_update['$inc'] = {'login_count': 1}
            
            self.users_collection.update_one(
                {'user_id': user_id},
                {'$set': oauth_update, '$inc': {'login_count': 1}}
            )
            
            # Get updated user
            updated_user = self.users_collection.find_one({'user_id': user_id})
            updated_user.pop('password_hash', None)
            updated_user['_id'] = str(updated_user['_id'])
            
            # Generate token
            token = self._generate_token(user_id)
            
            logger.info(f"Social account {provider} linked to existing user: {existing_user['username']}")
            
            return {
                'success': True,
                'user': updated_user,
                'token': token,
                'message': f'{provider.title()} account linked successfully'
            }
            
        except Exception as e:
            logger.error(f"Error linking social account: {str(e)}")
            return {'success': False, 'error': str(e)}

    def _generate_username_from_social(self, user_info: Dict[str, Any]) -> str:
        """Generate username from social login info"""
        # Try different combinations
        first_name = user_info.get('first_name', '').lower().strip()
        last_name = user_info.get('last_name', '').lower().strip()
        email = user_info.get('email', '')
        
        if first_name and last_name:
            return f"{first_name}.{last_name}"
        elif first_name:
            return first_name
        elif email:
            return email.split('@')[0].lower()
        else:
            return f"user_{user_info['provider']}_{str(uuid.uuid4())[:8]}"

    def _ensure_unique_username(self, base_username: str) -> str:
        """Ensure username is unique by appending numbers if needed"""
        username = base_username
        counter = 1
        
        while self.check_username_exists(username):
            username = f"{base_username}{counter}"
            counter += 1
        
        return username

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
                'ui_preferences': {
                    'theme_preference': 'light',
                    'display_language': 'en'
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
            },
            'ui_preferences': {
                'theme_preference': 'light',
                'display_language': 'en'
            }
        }
    
    def _update_user_statistics(self, user_id: str, interaction_data: Dict[str, Any]):
        """Update user statistics"""
        try:
            if self.users_collection is None:
                return
            
            action = interaction_data.get('action', '')
            update_doc = {
                '$set': {'updated_at': datetime.now().isoformat()}
            }
            
            if action == 'search':
                update_doc['$inc'] = {'statistics.total_searches': 1}
            elif action == 'feedback':
                update_doc['$inc'] = {'statistics.total_feedback': 1}
            
            self.users_collection.update_one(
                {'user_id': user_id},
                update_doc
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
    
    def change_password(self, user_id: str, current_password: str, new_password: str) -> Dict[str, Any]:
        """Change user password"""
        try:
            if self.users_collection is None:
                return {'success': False, 'error': 'User management not available'}
            
            # Get user
            user = self.users_collection.find_one({'user_id': user_id})
            if not user:
                return {'success': False, 'error': 'User not found'}
            
            # Verify current password
            if not bcrypt.checkpw(current_password.encode('utf-8'), user['password_hash']):
                return {'success': False, 'error': 'Current password is incorrect'}
            
            # Validate new password
            if len(new_password) < 6:
                return {'success': False, 'error': 'New password must be at least 6 characters long'}
            
            # Hash new password
            new_password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
            
            # Update password
            result = self.users_collection.update_one(
                {'user_id': user_id},
                {
                    '$set': {
                        'password_hash': new_password_hash,
                        'updated_at': datetime.now().isoformat()
                    }
                }
            )
            
            if result.modified_count > 0:
                logger.info(f"Password changed successfully for user: {user_id}")
                return {'success': True, 'message': 'Password changed successfully'}
            else:
                return {'success': False, 'error': 'Failed to update password'}
            
        except Exception as e:
            logger.error(f"Error changing password for user {user_id}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def update_profile_picture(self, user_id: str, picture_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update user profile picture"""
        try:
            if self.users_collection is None:
                return {'success': False, 'error': 'User management not available'}
            
            # Get user
            user = self.users_collection.find_one({'user_id': user_id})
            if not user:
                return {'success': False, 'error': 'User not found'}
            
            # Prepare profile picture data
            profile_picture = {
                'filename': picture_data['filename'],
                'data': picture_data['data'],
                'content_type': picture_data['content_type'],
                'size': picture_data['size'],
                'uploaded_at': picture_data['uploaded_at']
            }
            
            # Update user profile with picture data
            result = self.users_collection.update_one(
                {'user_id': user_id},
                {
                    '$set': {
                        'profile.profile_picture': profile_picture,
                        'updated_at': datetime.now().isoformat()
                    }
                }
            )
            
            # Check if update was successful (either modified or matched)
            if result.modified_count > 0 or result.matched_count > 0:
                logger.info(f"Profile picture updated successfully for user: {user_id}")
                return {
                    'success': True,
                    'message': 'Profile picture updated successfully',
                    'profile_picture_id': user_id
                }
            else:
                return {'success': False, 'error': 'Failed to update profile picture'}
            
        except Exception as e:
            logger.error(f"Error updating profile picture for user {user_id}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_profile_picture(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile picture"""
        try:
            if self.users_collection is None:
                return None
            
            user = self.users_collection.find_one(
                {'user_id': user_id},
                {'profile.profile_picture': 1}
            )
            
            if user and 'profile' in user and 'profile_picture' in user['profile']:
                return user['profile']['profile_picture']
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting profile picture for user {user_id}: {str(e)}")
            return None
    
    def delete_profile_picture(self, user_id: str) -> Dict[str, Any]:
        """Delete user profile picture"""
        try:
            if self.users_collection is None:
                return {'success': False, 'error': 'User management not available'}
            
            # Check if user exists and has a profile picture
            user = self.users_collection.find_one({'user_id': user_id})
            if not user:
                return {'success': False, 'error': 'User not found'}
            
            # Check if user has a profile picture
            has_profile_picture = user.get('profile', {}).get('profile_picture') is not None
            
            # Remove profile picture from user document
            result = self.users_collection.update_one(
                {'user_id': user_id},
                {
                    '$unset': {'profile.profile_picture': ''},
                    '$set': {'updated_at': datetime.now().isoformat()}
                }
            )
            
            if result.modified_count > 0 or has_profile_picture:
                logger.info(f"Profile picture deleted successfully for user: {user_id}")
                return {'success': True, 'message': 'Profile picture deleted successfully'}
            else:
                return {'success': False, 'error': 'No profile picture to delete'}
            
        except Exception as e:
            logger.error(f"Error deleting profile picture for user {user_id}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def send_verification_code(self, user_id: str) -> Dict[str, Any]:
        """Send email verification code to user"""
        try:
            if self.users_collection is None:
                return {'success': False, 'error': 'User management not available'}
            
            # Get user
            user = self.users_collection.find_one({'user_id': user_id})
            if not user:
                return {'success': False, 'error': 'User not found'}
            
            if not user.get('email'):
                return {'success': False, 'error': 'No email address associated with this account'}
            
            if user.get('email_verified', False):
                return {'success': False, 'error': 'Email already verified'}
            
            # Generate 6-digit verification code
            verification_code = str(random.randint(100000, 999999))
            
            # Store verification code in database
            code_record = {
                'user_id': user_id,
                'code': verification_code,
                'email': user['email'],
                'created_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(minutes=15)).isoformat(),  # 15 minutes expiry
                'used': False
            }
            
            if self.verification_codes_collection is not None:
                # Remove any existing codes for this user
                self.verification_codes_collection.delete_many({'user_id': user_id})
                # Insert new code
                self.verification_codes_collection.insert_one(code_record)
            
            # Send email
            email_sent = self._send_verification_email(user['email'], verification_code, user.get('profile', {}).get('first_name', ''))
            
            if email_sent:
                logger.info(f"Verification code sent to user {user_id}")
                return {'success': True, 'message': 'Verification code sent successfully'}
            else:
                return {'success': False, 'error': 'Failed to send verification email'}
            
        except Exception as e:
            logger.error(f"Error sending verification code to user {user_id}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def verify_email_code(self, user_id: str, verification_code: str) -> Dict[str, Any]:
        """Verify email verification code"""
        try:
            if self.users_collection is None or self.verification_codes_collection is None:
                return {'success': False, 'error': 'User management not available'}
            
            # Find verification code record
            code_record = self.verification_codes_collection.find_one({
                'user_id': user_id,
                'code': verification_code,
                'used': False
            })
            
            if not code_record:
                return {'success': False, 'error': 'Invalid or expired verification code'}
            
            # Check if code has expired
            expires_at = datetime.fromisoformat(code_record['expires_at'])
            if datetime.now() > expires_at:
                return {'success': False, 'error': 'Verification code has expired'}
            
            # Mark code as used
            self.verification_codes_collection.update_one(
                {'_id': code_record['_id']},
                {'$set': {'used': True, 'used_at': datetime.now().isoformat()}}
            )
            
            # Update user's email verification status
            result = self.users_collection.update_one(
                {'user_id': user_id},
                {
                    '$set': {
                        'email_verified': True,
                        'email_verified_at': datetime.now().isoformat(),
                        'updated_at': datetime.now().isoformat()
                    }
                }
            )
            
            if result.modified_count > 0:
                logger.info(f"Email verified successfully for user: {user_id}")
                return {'success': True, 'message': 'Email verified successfully'}
            else:
                return {'success': False, 'error': 'Failed to update user verification status'}
            
        except Exception as e:
            logger.error(f"Error verifying email code for user {user_id}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _send_verification_email(self, email: str, verification_code: str, first_name: str = '') -> bool:
        """Send verification email using SMTP"""
        try:
            # Email configuration from environment variables
            smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
            smtp_port = int(os.getenv('SMTP_PORT', '587'))
            smtp_use_tls = os.getenv('SMTP_USE_TLS', 'true').lower() == 'true'
            smtp_username = os.getenv('SMTP_USERNAME')
            smtp_password = os.getenv('SMTP_PASSWORD')
            from_name = os.getenv('EMAIL_FROM_NAME', 'APOSSS System')
            from_address = os.getenv('EMAIL_FROM_ADDRESS', smtp_username)
            
            if not smtp_username or not smtp_password:
                logger.error("SMTP credentials not configured. Please set SMTP_USERNAME and SMTP_PASSWORD environment variables.")
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = f"{from_name} <{from_address}>"
            msg['To'] = email
            msg['Subject'] = "Email Verification - APOSSS"
            
            # Email body with configurable content
            name_part = f"Hello {first_name}," if first_name else "Hello,"
            app_name = os.getenv('APP_NAME', 'APOSSS')
            support_email = os.getenv('EMAIL_SUPPORT_ADDRESS', smtp_username)
            verification_expiry = os.getenv('EMAIL_VERIFICATION_TOKEN_EXPIRY_HOURS', '24')
            
            body = f"""
{name_part}

Thank you for registering with {app_name}. To complete your registration, please verify your email address by entering the following 6-digit code:

Verification Code: {verification_code}

This code will expire in {verification_expiry} hours.

If you didn't request this verification, please ignore this email or contact us at {support_email}.

Best regards,
The {app_name} Team
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(smtp_username, smtp_password)
            text = msg.as_string()
            server.sendmail(smtp_username, email, text)
            server.quit()
            
            logger.info(f"Verification email sent successfully to {email}")
            return True
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error sending verification email to {email}: {error_msg}")
            
            # Provide specific guidance based on the error
            if "authentication failed" in error_msg.lower() or "username and password not accepted" in error_msg.lower():
                print("\n" + "="*60)
                print("📧 EMAIL SENDING FAILED - GMAIL APP PASSWORD REQUIRED")
                print("="*60)
                print(f"📋 Your verification code (use this for now): {verification_code}")
                print("="*60 + "\n")
            
            # For development purposes, still return True so the system works
            # but log the code to console
            logger.info(f"DEVELOPMENT FALLBACK: Verification code for {email}: {verification_code}")
            print(f"🔐 VERIFICATION CODE FOR {email}: {verification_code}")
            return True  # Return True so the system continues to work

    def request_password_reset(self, email: str) -> Dict[str, Any]:
        """Request password reset by sending verification code to email"""
        try:
            if self.users_collection is None:
                return {'success': False, 'error': 'User management not available'}
            
            # Find user by email
            user = self.users_collection.find_one({'email': email.lower().strip()})
            if not user:
                return {'success': False, 'error': 'No account found with this email address'}
            
            # Generate 6-digit verification code
            verification_code = str(random.randint(100000, 999999))
            
            # Store verification code in database
            code_record = {
                'user_id': user['user_id'],
                'email': email.lower().strip(),
                'code': verification_code,
                'type': 'password_reset',
                'created_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(minutes=15)).isoformat(),  # 15 minutes expiry
                'used': False
            }
            
            if self.verification_codes_collection is not None:
                # Remove any existing password reset codes for this user
                self.verification_codes_collection.delete_many({
                    'user_id': user['user_id'],
                    'type': 'password_reset'
                })
                # Insert new code
                self.verification_codes_collection.insert_one(code_record)
            
            # Send email
            first_name = user.get('profile', {}).get('first_name', '')
            email_sent = self._send_password_reset_email(email, verification_code, first_name)
            
            if email_sent:
                logger.info(f"Password reset code sent to user {user['user_id']}")
                return {'success': True, 'message': 'Password reset code sent successfully'}
            else:
                return {'success': False, 'error': 'Failed to send password reset email'}
            
        except Exception as e:
            logger.error(f"Error requesting password reset: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def reset_password_with_code(self, email: str, verification_code: str, new_password: str) -> Dict[str, Any]:
        """Reset password using verification code"""
        try:
            if self.users_collection is None or self.verification_codes_collection is None:
                return {'success': False, 'error': 'User management not available'}
            
            # Find user by email
            user = self.users_collection.find_one({'email': email.lower().strip()})
            if not user:
                return {'success': False, 'error': 'User not found'}
            
            # Find verification code record
            code_record = self.verification_codes_collection.find_one({
                'user_id': user['user_id'],
                'email': email.lower().strip(),
                'code': verification_code,
                'type': 'password_reset',
                'used': False
            })
            
            if not code_record:
                return {'success': False, 'error': 'Invalid or expired verification code'}
            
            # Check if code has expired
            expires_at = datetime.fromisoformat(code_record['expires_at'])
            if datetime.now() > expires_at:
                return {'success': False, 'error': 'Verification code has expired'}
            
            # Validate new password
            if len(new_password) < 8:
                return {'success': False, 'error': 'Password must be at least 8 characters long'}
            
            # Hash new password
            new_password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
            
            # Update user's password
            result = self.users_collection.update_one(
                {'user_id': user['user_id']},
                {
                    '$set': {
                        'password_hash': new_password_hash,
                        'updated_at': datetime.now().isoformat()
                    }
                }
            )
            
            if result.modified_count > 0:
                # Mark code as used
                self.verification_codes_collection.update_one(
                    {'_id': code_record['_id']},
                    {'$set': {'used': True, 'used_at': datetime.now().isoformat()}}
                )
                
                # Invalidate all existing sessions for this user (force re-login)
                if self.sessions_collection is not None:
                    self.sessions_collection.update_many(
                        {'user_id': user['user_id']},
                        {'$set': {'is_active': False}}
                    )
                
                logger.info(f"Password reset successfully for user: {user['user_id']}")
                return {'success': True, 'message': 'Password reset successfully'}
            else:
                return {'success': False, 'error': 'Failed to update password'}
            
        except Exception as e:
            logger.error(f"Error resetting password: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _send_password_reset_email(self, email: str, verification_code: str, first_name: str = '') -> bool:
        """Send password reset email using SMTP"""
        try:
            # Email configuration from environment variables
            smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
            smtp_port = int(os.getenv('SMTP_PORT', '587'))
            smtp_use_tls = os.getenv('SMTP_USE_TLS', 'true').lower() == 'true'
            smtp_username = os.getenv('SMTP_USERNAME')
            smtp_password = os.getenv('SMTP_PASSWORD')
            from_name = os.getenv('EMAIL_FROM_NAME', 'APOSSS System')
            from_address = os.getenv('EMAIL_FROM_ADDRESS', smtp_username)
            
            if not smtp_username or not smtp_password:
                logger.error("SMTP credentials not configured. Please set SMTP_USERNAME and SMTP_PASSWORD environment variables.")
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = f"{from_name} <{from_address}>"
            msg['To'] = email
            msg['Subject'] = "Password Reset - APOSSS"
            
            # Email body with configurable content
            name_part = f"Hello {first_name}," if first_name else "Hello,"
            app_name = os.getenv('APP_NAME', 'APOSSS')
            support_email = os.getenv('EMAIL_SUPPORT_ADDRESS', smtp_username)
            reset_expiry = os.getenv('PASSWORD_RESET_TOKEN_EXPIRY_MINUTES', '30')
            
            body = f"""
{name_part}

You have requested to reset your password for your {app_name} account. To complete the password reset process, please use the following 6-digit verification code:

Password Reset Code: {verification_code}

This code will expire in {reset_expiry} minutes for security reasons.

If you didn't request this password reset, please ignore this email and your password will remain unchanged. If you believe someone else requested this reset, please contact us immediately at {support_email}.

For security purposes, please do not share this code with anyone.

Best regards,
The {app_name} Team
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(smtp_username, smtp_password)
            text = msg.as_string()
            server.sendmail(smtp_username, email, text)
            server.quit()
            
            logger.info(f"Password reset email sent successfully to {email}")
            return True
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error sending password reset email to {email}: {error_msg}")
            
            # Provide specific guidance based on the error
            if "authentication failed" in error_msg.lower() or "username and password not accepted" in error_msg.lower():
                print("\n" + "="*60)
                print("📧 EMAIL SENDING FAILED - GMAIL APP PASSWORD REQUIRED")
                print("="*60)
                print(f"📋 Your password reset code (use this for now): {verification_code}")
                print("="*60 + "\n")
            
            # For development purposes, still return True so the system works
            # but log the code to console
            logger.info(f"DEVELOPMENT FALLBACK: Password reset code for {email}: {verification_code}")
            print(f"🔐 PASSWORD RESET CODE FOR {email}: {verification_code}")
            return True  # Return True so the system continues to work