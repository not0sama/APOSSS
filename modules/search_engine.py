import logging
import re
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from .database_manager import DatabaseManager
from .query_processor import QueryProcessor

logger = logging.getLogger(__name__)

class SearchEngine:
    """Multi-database search engine for APOSSS"""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize with database manager"""
        self.db_manager = db_manager
        logger.info("Search engine initialized successfully")
    
    def search_all_databases(self, processed_query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search across all databases using processed query data
        
        Args:
            processed_query: The structured query data from LLM processing
            
        Returns:
            Aggregated search results from all databases
        """
        try:
            # Extract search parameters
            search_params = self._extract_search_parameters(processed_query)
            
            # Search each database
            academic_results = self._search_academic_library(search_params)
            experts_results = self._search_experts_system(search_params)
            papers_results = self._search_research_papers(search_params)
            labs_results = self._search_laboratories(search_params)
            
            # Aggregate results
            aggregated_results = self._aggregate_results({
                'academic_library': academic_results,
                'experts_system': experts_results,
                'research_papers': papers_results,
                'laboratories': labs_results
            }, processed_query)
            
            logger.info(f"Search completed. Found {aggregated_results['total_results']} total results")
            return aggregated_results
            
        except Exception as e:
            logger.error(f"Error in search_all_databases: {str(e)}")
            return self._create_empty_results(str(e))
    
    def _extract_search_parameters(self, processed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and prepare search parameters from processed query"""
        keywords = processed_query.get('keywords', {})
        entities = processed_query.get('entities', {})
        academic_fields = processed_query.get('academic_fields', {})
        intent = processed_query.get('intent', {})
        
        # Combine all search terms
        all_terms = []
        all_terms.extend(keywords.get('primary', []))
        all_terms.extend(keywords.get('secondary', []))
        all_terms.extend(keywords.get('technical_terms', []))
        all_terms.extend(entities.get('technologies', []))
        all_terms.extend(entities.get('concepts', []))
        
        # Create search parameters
        search_params = {
            'primary_terms': keywords.get('primary', []),
            'secondary_terms': keywords.get('secondary', []) + keywords.get('technical_terms', []),
            'all_terms': list(set(all_terms)),  # Remove duplicates
            'people': entities.get('people', []),
            'organizations': entities.get('organizations', []),
            'locations': entities.get('locations', []),
            'technologies': entities.get('technologies', []),
            'concepts': entities.get('concepts', []),
            'academic_fields': [academic_fields.get('primary_field', '')] + academic_fields.get('related_fields', []),
            'specializations': academic_fields.get('specializations', []),
            'intent': intent.get('primary_intent', 'general_search'),
            'corrected_query': processed_query.get('corrected_query', '')
        }
        
        # Remove empty strings and duplicates
        for key in search_params:
            if isinstance(search_params[key], list):
                search_params[key] = list(set(filter(None, search_params[key])))
        
        return search_params
    
    def _search_academic_library(self, search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search Academic Library database (books, journals, projects)"""
        results = []
        
        try:
            # Search books
            books_results = self._search_collection(
                'academic_library', 'books', search_params,
                search_fields=['title', 'author', 'description', 'abstract', 'keywords', 'category'],
                result_type='book'
            )
            results.extend(books_results)
            
            # Search journals
            journals_results = self._search_collection(
                'academic_library', 'journals', search_params,
                search_fields=['title', 'editor', 'description', 'abstract', 'keywords', 'category'],
                result_type='journal'
            )
            results.extend(journals_results)
            
            # Search projects
            projects_results = self._search_collection(
                'academic_library', 'projects', search_params,
                search_fields=['title', 'student_name', 'supervisor', 'description', 'abstract', 'keywords', 'category', 'department'],
                result_type='project'
            )
            results.extend(projects_results)
            
        except Exception as e:
            logger.error(f"Error searching academic library: {str(e)}")
        
        return results
    
    def _search_experts_system(self, search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search Experts System database (experts, certificates)"""
        results = []
        
        try:
            # Search experts
            experts_results = self._search_collection(
                'experts_system', 'experts', search_params,
                search_fields=['name', 'general_information.slogan', 'job_complete.role', 'general_information.locations'],
                result_type='expert'
            )
            results.extend(experts_results)
            
            # Search certificates
            certificates_results = self._search_collection(
                'experts_system', 'certificates', search_params,
                search_fields=['title', 'degree', 'institution', 'description'],
                result_type='certificate'
            )
            results.extend(certificates_results)
            
        except Exception as e:
            logger.error(f"Error searching experts system: {str(e)}")
        
        return results
    
    def _search_research_papers(self, search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search Research Papers database (articles, conferences, theses)"""
        results = []
        
        try:
            # Search articles
            articles_results = self._search_collection(
                'research_papers', 'articles', search_params,
                search_fields=['title', 'authors', 'abstract'],
                result_type='article'
            )
            results.extend(articles_results)
            
            # Search conferences
            conferences_results = self._search_collection(
                'research_papers', 'conferences', search_params,
                search_fields=['title', 'authors', 'summary', 'primary_category'],
                result_type='conference'
            )
            results.extend(conferences_results)
            
            # Search theses
            theses_results = self._search_collection(
                'research_papers', 'theses', search_params,
                search_fields=['title', 'student_name', 'supervisor', 'abstract', 'keywords', 'department', 'faculty'],
                result_type='thesis'
            )
            results.extend(theses_results)
            
        except Exception as e:
            logger.error(f"Error searching research papers: {str(e)}")
        
        return results
    
    def _search_laboratories(self, search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search Laboratories database (equipments, materials)"""
        results = []
        
        try:
            # Search equipments
            equipments_results = self._search_collection(
                'laboratories', 'equipments', search_params,
                search_fields=['equipment_name', 'description', 'model', 'specifications'],
                result_type='equipment'
            )
            results.extend(equipments_results)
            
            # Search materials
            materials_results = self._search_collection(
                'laboratories', 'materials', search_params,
                search_fields=['material_name', 'description', 'supplier.name'],
                result_type='material'
            )
            results.extend(materials_results)
            
        except Exception as e:
            logger.error(f"Error searching laboratories: {str(e)}")
        
        return results
    
    def _search_collection(self, db_name: str, collection_name: str, search_params: Dict[str, Any], 
                          search_fields: List[str], result_type: str) -> List[Dict[str, Any]]:
        """Generic method to search a specific collection"""
        results = []
        
        try:
            collection = self.db_manager.get_collection(db_name, collection_name)
            if collection is None:
                logger.warning(f"Collection {db_name}.{collection_name} not available")
                return results
            
            # Build MongoDB query
            query = self._build_mongodb_query(search_params, search_fields)
            
            if not query:  # If no valid query terms, return empty results
                return results
            
            # Execute query with limit
            cursor = collection.find(query).limit(50)  # Limit to 50 results per collection
            
            # Process results
            for doc in cursor:
                processed_result = self._process_search_result(doc, result_type, db_name, collection_name)
                if processed_result:
                    results.append(processed_result)
            
            logger.info(f"Found {len(results)} results in {db_name}.{collection_name}")
            
        except Exception as e:
            logger.error(f"Error searching {db_name}.{collection_name}: {str(e)}")
        
        return results
    
    def _build_mongodb_query(self, search_params: Dict[str, Any], search_fields: List[str]) -> Dict[str, Any]:
        """Build MongoDB query from search parameters"""
        query_conditions = []
        
        # Primary terms (higher priority)
        primary_terms = search_params.get('primary_terms', [])
        if primary_terms:
            primary_conditions = []
            for term in primary_terms:
                field_conditions = []
                for field in search_fields:
                    field_conditions.append({field: {"$regex": re.escape(term), "$options": "i"}})
                primary_conditions.append({"$or": field_conditions})
            
            if primary_conditions:
                query_conditions.append({"$or": primary_conditions})
        
        # Secondary terms
        secondary_terms = search_params.get('secondary_terms', [])
        if secondary_terms:
            secondary_conditions = []
            for term in secondary_terms:
                field_conditions = []
                for field in search_fields:
                    field_conditions.append({field: {"$regex": re.escape(term), "$options": "i"}})
                secondary_conditions.append({"$or": field_conditions})
            
            if secondary_conditions:
                query_conditions.append({"$or": secondary_conditions})
        
        # If no primary or secondary terms, use all terms
        if not query_conditions:
            all_terms = search_params.get('all_terms', [])
            if all_terms:
                all_conditions = []
                for term in all_terms[:10]:  # Limit to 10 terms to avoid overly complex queries
                    field_conditions = []
                    for field in search_fields:
                        field_conditions.append({field: {"$regex": re.escape(term), "$options": "i"}})
                    all_conditions.append({"$or": field_conditions})
                
                if all_conditions:
                    query_conditions.append({"$or": all_conditions})
        
        # Combine conditions
        if len(query_conditions) == 1:
            return query_conditions[0]
        elif len(query_conditions) > 1:
            return {"$or": query_conditions}
        else:
            return {}
    
    def _process_search_result(self, doc: Dict[str, Any], result_type: str, 
                              db_name: str, collection_name: str) -> Optional[Dict[str, Any]]:
        """Process and standardize a search result"""
        try:
            # Convert ObjectId to string
            doc_id = str(doc.get('_id', ''))
            
            # Extract common fields based on result type
            result = {
                'id': doc_id,
                'type': result_type,
                'database': db_name,
                'collection': collection_name,
                'title': '',
                'description': '',
                'author': '',
                'snippet': '',
                'metadata': {}
            }
            
            # Type-specific field mapping
            if result_type in ['book', 'journal']:
                result['title'] = doc.get('title', '')
                result['description'] = doc.get('description', '') or doc.get('abstract', '')
                result['author'] = doc.get('author', '') or doc.get('editor', '')
                result['metadata'] = {
                    'category': doc.get('category', ''),
                    'publication_date': str(doc.get('publication_date', '')),
                    'language': doc.get('language', ''),
                    'keywords': doc.get('keywords', [])
                }
            
            elif result_type == 'project':
                result['title'] = doc.get('title', '')
                result['description'] = doc.get('description', '') or doc.get('abstract', '')
                result['author'] = doc.get('student_name', '')
                result['metadata'] = {
                    'supervisor': doc.get('supervisor', ''),
                    'university': doc.get('university', ''),
                    'department': doc.get('department', ''),
                    'category': doc.get('category', ''),
                    'keywords': doc.get('keywords', [])
                }
            
            elif result_type == 'expert':
                result['title'] = doc.get('name', '')
                result['description'] = doc.get('general_information', {}).get('slogan', '')
                result['author'] = doc.get('name', '')
                result['metadata'] = {
                    'role': doc.get('job_complete', {}).get('role', ''),
                    'locations': doc.get('general_information', {}).get('locations', ''),
                    'languages': doc.get('general_information', {}).get('languages', []),
                    'email': doc.get('email', '')
                }
            
            elif result_type == 'certificate':
                result['title'] = doc.get('title', '')
                result['description'] = doc.get('description', '')
                result['metadata'] = {
                    'degree': doc.get('degree', ''),
                    'institution': doc.get('institution', ''),
                    'years': doc.get('years', ''),
                    'status': doc.get('status', '')
                }
            
            elif result_type in ['article', 'conference']:
                result['title'] = doc.get('title', '')
                result['description'] = doc.get('abstract', '') or doc.get('summary', '')
                result['author'] = str(doc.get('authors', ''))
                result['metadata'] = {
                    'year': str(doc.get('year', '')) or str(doc.get('published', '')),
                    'citations': doc.get('citations', 0),
                    'category': doc.get('primary_category', '')
                }
            
            elif result_type == 'thesis':
                result['title'] = doc.get('title', '')
                result['description'] = doc.get('abstract', '')
                result['author'] = doc.get('student_name', '')
                result['metadata'] = {
                    'supervisor': doc.get('supervisor', ''),
                    'university': doc.get('university', ''),
                    'faculty': doc.get('faculty', ''),
                    'department': doc.get('department', ''),
                    'degree_type': doc.get('degree_type', ''),
                    'keywords': doc.get('keywords', [])
                }
            
            elif result_type == 'equipment':
                result['title'] = doc.get('equipment_name', '')
                result['description'] = doc.get('description', '')
                result['metadata'] = {
                    'model': doc.get('model', ''),
                    'serial_number': doc.get('serial_number', ''),
                    'status': doc.get('status', ''),
                    'specifications': doc.get('specifications', '')
                }
            
            elif result_type == 'material':
                result['title'] = doc.get('material_name', '')
                result['description'] = doc.get('description', '')
                result['metadata'] = {
                    'quantity': str(doc.get('quantity', '')),
                    'unit': doc.get('unit', ''),
                    'status': doc.get('status', ''),
                    'supplier': doc.get('supplier', {}).get('name', '') if isinstance(doc.get('supplier'), dict) else str(doc.get('supplier', '')),
                    'storage_location': doc.get('storage_location', '')
                }
            
            # Generate snippet
            result['snippet'] = self._generate_snippet(result['description'], result['title'])
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing search result: {str(e)}")
            return None
    
    def _generate_snippet(self, description: str, title: str, max_length: int = 200) -> str:
        """Generate a snippet from description or title"""
        text = description or title or ''
        if len(text) <= max_length:
            return text
        
        # Try to cut at word boundary
        if len(text) > max_length:
            snippet = text[:max_length]
            last_space = snippet.rfind(' ')
            if last_space > max_length * 0.8:  # If we can cut at a word boundary reasonably close
                snippet = snippet[:last_space]
            snippet += '...'
            return snippet
        
        return text
    
    def _aggregate_results(self, db_results: Dict[str, List], processed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from all databases"""
        all_results = []
        
        # Flatten results from all databases
        for db_name, results in db_results.items():
            all_results.extend(results)
        
        # Count results by type and database
        result_counts = {
            'by_type': {},
            'by_database': {}
        }
        
        for result in all_results:
            result_type = result['type']
            db_name = result['database']
            
            result_counts['by_type'][result_type] = result_counts['by_type'].get(result_type, 0) + 1
            result_counts['by_database'][db_name] = result_counts['by_database'].get(db_name, 0) + 1
        
        return {
            'results': all_results,
            'total_results': len(all_results),
            'query_info': {
                'original_query': processed_query.get('_metadata', {}).get('original_query', ''),
                'corrected_query': processed_query.get('corrected_query', ''),
                'intent': processed_query.get('intent', {}).get('primary_intent', ''),
                'search_timestamp': datetime.now().isoformat()
            },
            'result_counts': result_counts,
            'search_metadata': {
                'databases_searched': list(db_results.keys()),
                'collections_searched': len([item for sublist in db_results.values() for item in sublist]),
                'success': True
            }
        }
    
    def _create_empty_results(self, error_msg: str = '') -> Dict[str, Any]:
        """Create empty results structure for error cases"""
        return {
            'results': [],
            'total_results': 0,
            'query_info': {
                'original_query': '',
                'corrected_query': '',
                'intent': 'general_search',
                'search_timestamp': datetime.now().isoformat()
            },
            'result_counts': {'by_type': {}, 'by_database': {}},
            'search_metadata': {
                'databases_searched': [],
                'collections_searched': 0,
                'success': False,
                'error': error_msg
            }
        } 