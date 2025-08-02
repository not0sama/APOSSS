import os
import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages connections to all four MongoDB databases"""
    
    def __init__(self):
        """Initialize database connections"""
        self.connections = {}
        self.databases = {}
        
        # Database configuration
        self.db_configs = {
            'academic_library': {
                'uri': os.getenv('MONGODB_URI_ACADEMIC_LIBRARY'),
                'collections': ['books', 'journals', 'projects']
            },
            'experts_system': {
                'uri': os.getenv('MONGODB_URI_EXPERTS_SYSTEM'),
                'collections': ['experts', 'certificates']
            },
            'research_papers': {
                'uri': os.getenv('MONGODB_URI_RESEARCH_PAPERS'),
                'collections': ['articles', 'conferences', 'theses']
            },
            'laboratories': {
                'uri': os.getenv('MONGODB_URI_LABORATORIES'),
                'collections': ['equipments', 'materials']
            },
            'funding': {
                'uri': os.getenv('MONGODB_URI_FUNDING'),
                'collections': ['research_projects', 'institutions', 'funding_records']
            },
            'aposss': {
                'uri': os.getenv('MONGODB_URI_APOSSS'),
                'collections': ['user_feedback', 'users', 'user_interactions', 'user_preferences', 'user_sessions', 'user_bookmarks', 'user_search_history']
            }
        }
        
        self._initialize_connections()
    
    def _initialize_connections(self):
        """Initialize connections to all databases"""
        for db_name, config in self.db_configs.items():
            try:
                # Create MongoDB client
                client = MongoClient(
                    config['uri'],
                    serverSelectionTimeoutMS=5000,  # 5 second timeout
                    connectTimeoutMS=5000,
                    socketTimeoutMS=5000
                )
                
                # Get database name from URI
                db_name_from_uri = config['uri'].split('/')[-1]
                database = client[db_name_from_uri]
                
                # Test connection
                client.admin.command('ping')
                
                # Store connections
                self.connections[db_name] = client
                self.databases[db_name] = database
                
                logger.info(f"Successfully connected to {db_name} database")
                
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                logger.warning(f"Failed to connect to {db_name} database: {str(e)}")
                self.connections[db_name] = None
                self.databases[db_name] = None
            except Exception as e:
                logger.error(f"Unexpected error connecting to {db_name}: {str(e)}")
                self.connections[db_name] = None
                self.databases[db_name] = None
    
    def test_connections(self) -> Dict[str, Any]:
        """Test all database connections and return status"""
        status = {}
        
        for db_name in self.db_configs.keys():
            if self.connections[db_name] is not None and self.databases[db_name] is not None:
                try:
                    # Test connection with ping
                    self.connections[db_name].admin.command('ping')
                    
                    # Get collection counts
                    collections_info = {}
                    for collection_name in self.db_configs[db_name]['collections']:
                        try:
                            count = self.databases[db_name][collection_name].count_documents({})
                            collections_info[collection_name] = count
                        except Exception as e:
                            collections_info[collection_name] = f"Error: {str(e)}"
                    
                    status[db_name] = {
                        'connected': True,
                        'collections': collections_info
                    }
                    
                except Exception as e:
                    status[db_name] = {
                        'connected': False,
                        'error': str(e)
                    }
            else:
                status[db_name] = {
                    'connected': False,
                    'error': 'Connection not established'
                }
        
        return status
    
    def get_database(self, db_name: str) -> Optional[Any]:
        """Get a specific database connection"""
        return self.databases.get(db_name)
    
    def get_collection(self, db_name: str, collection_name: str) -> Optional[Any]:
        """Get a specific collection from a database"""
        database = self.get_database(db_name)
        if database is not None:
            return database[collection_name]
        return None
    
    def close_connections(self):
        """Close all database connections"""
        for db_name, client in self.connections.items():
            if client is not None:
                try:
                    client.close()
                    logger.info(f"Closed connection to {db_name}")
                except Exception as e:
                    logger.error(f"Error closing connection to {db_name}: {str(e)}")
        
        self.connections.clear()
        self.databases.clear()
    
    def __del__(self):
        """Cleanup connections when object is destroyed"""
        self.close_connections() 