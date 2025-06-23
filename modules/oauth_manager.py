#!/usr/bin/env python3
"""
OAuth Manager for Social Authentication
Handles Google and ORCID OAuth flows
"""
import os
import logging
import requests
import secrets
from urllib.parse import urlencode
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class OAuthManager:
    """Manages OAuth authentication for Google and ORCID"""
    
    def __init__(self):
        """Initialize OAuth manager with provider configurations"""
        self.base_url = os.getenv('BASE_URL', 'http://localhost:5000')
        
        # Google OAuth Configuration
        self.google_config = {
            'client_id': os.getenv('GOOGLE_CLIENT_ID'),
            'client_secret': os.getenv('GOOGLE_CLIENT_SECRET'),
            'authorization_base_url': 'https://accounts.google.com/o/oauth2/v2/auth',
            'token_url': 'https://oauth2.googleapis.com/token',
            'userinfo_url': 'https://www.googleapis.com/oauth2/v2/userinfo',
            'redirect_uri': f"{self.base_url}/api/auth/google/callback",
            'scope': 'openid email profile'
        }
        
        # ORCID OAuth Configuration
        self.orcid_config = {
            'client_id': os.getenv('ORCID_CLIENT_ID'),
            'client_secret': os.getenv('ORCID_CLIENT_SECRET'),
            'authorization_base_url': 'https://orcid.org/oauth/authorize',
            'token_url': 'https://orcid.org/oauth/token',
            'userinfo_url': 'https://pub.orcid.org/v3.0',
            'redirect_uri': f"{self.base_url}/api/auth/orcid/callback",
            'scope': '/authenticate'
        }
        
        logger.info("OAuth Manager initialized")
    
    def is_configured(self, provider: str) -> bool:
        """Check if OAuth provider is properly configured"""
        if provider == 'google':
            return bool(self.google_config['client_id'] and self.google_config['client_secret'])
        elif provider == 'orcid':
            return bool(self.orcid_config['client_id'] and self.orcid_config['client_secret'])
        return False
    
    def get_authorization_url(self, provider: str, state: str = None) -> Dict[str, Any]:
        """Generate OAuth authorization URL"""
        if not state:
            state = secrets.token_urlsafe(32)
        
        try:
            if provider == 'google':
                config = self.google_config
                params = {
                    'client_id': config['client_id'],
                    'redirect_uri': config['redirect_uri'],
                    'scope': config['scope'],
                    'response_type': 'code',
                    'state': state,
                    'access_type': 'offline',
                    'prompt': 'consent'
                }
                
            elif provider == 'orcid':
                config = self.orcid_config
                params = {
                    'client_id': config['client_id'],
                    'redirect_uri': config['redirect_uri'],
                    'scope': config['scope'],
                    'response_type': 'code',
                    'state': state
                }
                
            else:
                return {'success': False, 'error': 'Unsupported provider'}
            
            auth_url = f"{config['authorization_base_url']}?{urlencode(params)}"
            
            return {
                'success': True,
                'authorization_url': auth_url,
                'state': state
            }
            
        except Exception as e:
            logger.error(f"Error generating {provider} authorization URL: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def exchange_code_for_token(self, provider: str, code: str, state: str = None) -> Dict[str, Any]:
        """Exchange authorization code for access token"""
        try:
            if provider == 'google':
                config = self.google_config
                data = {
                    'client_id': config['client_id'],
                    'client_secret': config['client_secret'],
                    'code': code,
                    'grant_type': 'authorization_code',
                    'redirect_uri': config['redirect_uri']
                }
                
            elif provider == 'orcid':
                config = self.orcid_config
                data = {
                    'client_id': config['client_id'],
                    'client_secret': config['client_secret'],
                    'code': code,
                    'grant_type': 'authorization_code',
                    'redirect_uri': config['redirect_uri']
                }
                
            else:
                return {'success': False, 'error': 'Unsupported provider'}
            
            headers = {
                'Accept': 'application/json',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            response = requests.post(config['token_url'], data=data, headers=headers)
            response.raise_for_status()
            
            token_data = response.json()
            
            return {
                'success': True,
                'access_token': token_data.get('access_token'),
                'token_type': token_data.get('token_type', 'Bearer'),
                'scope': token_data.get('scope'),
                'orcid': token_data.get('orcid') if provider == 'orcid' else None
            }
            
        except requests.RequestException as e:
            logger.error(f"Error exchanging {provider} code for token: {str(e)}")
            return {'success': False, 'error': f'Token exchange failed: {str(e)}'}
        except Exception as e:
            logger.error(f"Unexpected error during {provider} token exchange: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_user_info(self, provider: str, access_token: str, orcid: str = None) -> Dict[str, Any]:
        """Get user information from OAuth provider"""
        try:
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Accept': 'application/json'
            }
            
            if provider == 'google':
                response = requests.get(self.google_config['userinfo_url'], headers=headers)
                response.raise_for_status()
                user_data = response.json()
                
                return {
                    'success': True,
                    'user_info': {
                        'provider': 'google',
                        'provider_id': user_data.get('id'),
                        'email': user_data.get('email'),
                        'email_verified': user_data.get('verified_email', False),
                        'first_name': user_data.get('given_name', ''),
                        'last_name': user_data.get('family_name', ''),
                        'full_name': user_data.get('name', ''),
                        'picture': user_data.get('picture'),
                        'locale': user_data.get('locale')
                    }
                }
                
            elif provider == 'orcid':
                if not orcid:
                    return {'success': False, 'error': 'ORCID ID required'}
                
                # Get ORCID profile information
                profile_url = f"{self.orcid_config['userinfo_url']}/{orcid}/person"
                response = requests.get(profile_url, headers=headers)
                response.raise_for_status()
                profile_data = response.json()
                
                # Extract name information
                name_info = profile_data.get('name', {})
                given_names = name_info.get('given-names', {}).get('value', '') if name_info.get('given-names') else ''
                family_name = name_info.get('family-name', {}).get('value', '') if name_info.get('family-name') else ''
                
                # Get email information (if public)
                email_url = f"{self.orcid_config['userinfo_url']}/{orcid}/email"
                email_response = requests.get(email_url, headers=headers)
                email = None
                if email_response.status_code == 200:
                    email_data = email_response.json()
                    emails = email_data.get('email', [])
                    if emails:
                        # Get the primary email or the first one
                        primary_email = next((e for e in emails if e.get('primary')), emails[0])
                        email = primary_email.get('email')
                
                return {
                    'success': True,
                    'user_info': {
                        'provider': 'orcid',
                        'provider_id': orcid,
                        'orcid': orcid,
                        'email': email,
                        'email_verified': email is not None,
                        'first_name': given_names,
                        'last_name': family_name,
                        'full_name': f"{given_names} {family_name}".strip(),
                        'picture': None,
                        'locale': None
                    }
                }
                
            else:
                return {'success': False, 'error': 'Unsupported provider'}
                
        except requests.RequestException as e:
            logger.error(f"Error getting {provider} user info: {str(e)}")
            return {'success': False, 'error': f'Failed to get user info: {str(e)}'}
        except Exception as e:
            logger.error(f"Unexpected error getting {provider} user info: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def process_oauth_login(self, provider: str, code: str, state: str = None) -> Dict[str, Any]:
        """Complete OAuth login process"""
        try:
            # Exchange code for token
            token_result = self.exchange_code_for_token(provider, code, state)
            if not token_result['success']:
                return token_result
            
            access_token = token_result['access_token']
            orcid = token_result.get('orcid')
            
            # Get user information
            user_info_result = self.get_user_info(provider, access_token, orcid)
            if not user_info_result['success']:
                return user_info_result
            
            user_info = user_info_result['user_info']
            
            return {
                'success': True,
                'user_info': user_info,
                'access_token': access_token
            }
            
        except Exception as e:
            logger.error(f"Error processing {provider} OAuth login: {str(e)}")
            return {'success': False, 'error': str(e)} 