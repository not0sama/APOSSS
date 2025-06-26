import logging
import re
import os
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from .database_manager import DatabaseManager
from .query_processor import QueryProcessor

# Import embedding ranker for pre-built index
try:
    from .embedding_ranker import EmbeddingRanker
    PREINDEX_AVAILABLE = True
except ImportError:
    PREINDEX_AVAILABLE = False
    EmbeddingRanker = None

logger = logging.getLogger(__name__)

class SearchEngine:
    """Multi-database search engine for APOSSS with pre-indexing support"""
    
    def __init__(self, db_manager: DatabaseManager, use_preindex: bool = True):
        """Initialize with database manager and optional pre-indexing"""
        self.db_manager = db_manager
        self.use_preindex = use_preindex and PREINDEX_AVAILABLE
        self.preindex_ranker = None
        
        # Initialize pre-built index if available
        if self.use_preindex:
            try:
                # Check if production index exists
                production_cache_dir = 'production_index_cache'
                if os.path.exists(os.path.join(production_cache_dir, 'faiss_index.pkl')):
                    self.preindex_ranker = EmbeddingRanker(cache_dir=production_cache_dir)
                    stats = self.preindex_ranker.get_cache_stats()
                    logger.info(f"Pre-built index loaded: {stats['total_vectors']} documents indexed")
                else:
                    logger.info("No pre-built index found, using traditional search only")
                    self.use_preindex = False
            except Exception as e:
                logger.warning(f"Failed to load pre-built index: {e}")
                self.use_preindex = False
        
        logger.info(f"Search engine initialized successfully (pre-index: {'enabled' if self.use_preindex else 'disabled'})")
    
    def search_all_databases(self, processed_query: Dict[str, Any], 
                           hybrid_search: bool = True, database_filters: List[str] = None) -> Dict[str, Any]:
        """
        Search across all databases using processed query data
        
        Args:
            processed_query: The structured query data from LLM processing
            hybrid_search: Whether to combine traditional + pre-index search
            database_filters: Optional list of database names to filter by
            
        Returns:
            Aggregated search results from all databases
        """
        try:
            if self.use_preindex and hybrid_search:
                return self._hybrid_search(processed_query, database_filters)
            else:
                return self._traditional_search(processed_query, database_filters)
                
        except Exception as e:
            logger.error(f"Error in search_all_databases: {str(e)}")
            return self._create_empty_results(str(e))
    
    def _hybrid_search(self, processed_query: Dict[str, Any], database_filters: List[str] = None) -> Dict[str, Any]:
        """Combine pre-index semantic search with traditional keyword search"""
        try:
            # Get original query for semantic search
            original_query = processed_query.get('corrected_query', '')
            
            # Step 1: Fast semantic search using pre-built index
            logger.info("🧠 Performing semantic search using pre-built index...")
            semantic_results = self.preindex_ranker.search_similar_documents(
                original_query, k=100, processed_query=processed_query
            )
            
            # Step 2: Traditional keyword search for precision
            logger.info("🔍 Performing traditional keyword search...")
            traditional_results = self._traditional_search(processed_query, database_filters)
            
            # Step 3: Merge and deduplicate results
            logger.info("🔄 Merging semantic and keyword search results...")
            merged_results = self._merge_search_results(semantic_results, traditional_results, processed_query)
            
            logger.info(f"Hybrid search completed: {merged_results['total_results']} total results")
            return merged_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            # Fallback to traditional search
            return self._traditional_search(processed_query, database_filters)
    
    def _traditional_search(self, processed_query: Dict[str, Any], database_filters: List[str] = None) -> Dict[str, Any]:
        """Traditional MongoDB keyword search - Always searches all databases for frontend filtering"""
        # Extract search parameters
        search_params = self._extract_search_parameters(processed_query)
        
        # Always search all databases - frontend will handle filtering
        results_dict = {}
        
        # Search all databases regardless of database_filters
        results_dict['academic_library'] = self._search_academic_library(search_params)
        results_dict['experts_system'] = self._search_experts_system(search_params)
        results_dict['research_papers'] = self._search_research_papers(search_params)
        results_dict['laboratories'] = self._search_laboratories(search_params)
        results_dict['funding'] = self._search_funding_system(search_params)
        
        # Aggregate results
        aggregated_results = self._aggregate_results(results_dict, processed_query)
        
        return aggregated_results
    
    def _merge_search_results(self, semantic_results: List[Dict[str, Any]], 
                            traditional_results: Dict[str, Any], 
                            processed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Merge semantic and traditional search results"""
        try:
            # Get traditional results list
            traditional_list = traditional_results.get('results', [])
            
            # Create a mapping of semantic results by document ID
            semantic_map = {result['id']: result for result in semantic_results}
            traditional_map = {result['id']: result for result in traditional_list}
            
            # Merge results, prioritizing those found in both searches
            merged_results = []
            processed_ids = set()
            
            # Add semantic results that also appear in traditional search (highest priority)
            for sem_result in semantic_results:
                doc_id = sem_result['id']
                if doc_id in traditional_map and doc_id not in processed_ids:
                    # Merge the results
                    merged_result = traditional_map[doc_id].copy()
                    merged_result['semantic_similarity'] = sem_result.get('similarity_score', 0)
                    merged_result['search_source'] = 'hybrid'
                    merged_results.append(merged_result)
                    processed_ids.add(doc_id)
            
            # Add remaining semantic results (semantic-only)
            for sem_result in semantic_results:
                doc_id = sem_result['id']
                if doc_id not in processed_ids:
                    # Convert semantic result to full result format
                    full_result = self._convert_semantic_to_full_result(sem_result)
                    if full_result:
                        full_result['semantic_similarity'] = sem_result.get('similarity_score', 0)
                        full_result['search_source'] = 'semantic'
                        merged_results.append(full_result)
                        processed_ids.add(doc_id)
            
            # Add remaining traditional results (keyword-only)
            for trad_result in traditional_list:
                doc_id = trad_result['id']
                if doc_id not in processed_ids:
                    trad_result['semantic_similarity'] = 0
                    trad_result['search_source'] = 'traditional'
                    merged_results.append(trad_result)
                    processed_ids.add(doc_id)
            
            # Update result counts
            result_counts = traditional_results.get('result_counts', {})
            result_counts['semantic_results'] = len(semantic_results)
            result_counts['hybrid_results'] = len([r for r in merged_results if r['search_source'] == 'hybrid'])
            
            return {
                'results': merged_results,
                'total_results': len(merged_results),
                'result_counts': result_counts,
                'search_metadata': {
                    'search_type': 'hybrid',
                    'semantic_results': len(semantic_results),
                    'traditional_results': len(traditional_list),
                    'merged_results': len(merged_results),
                    'hybrid_matches': len([r for r in merged_results if r['search_source'] == 'hybrid']),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error merging search results: {e}")
            return traditional_results
    
    def _convert_semantic_to_full_result(self, semantic_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert a semantic search result to full result format by fetching from database"""
        try:
            doc_id = semantic_result['id']
            db_name = semantic_result.get('database', '')
            collection_name = semantic_result.get('collection', '')
            
            if not all([doc_id, db_name, collection_name]):
                return None
            
            # Fetch full document from database
            database = self.db_manager.get_database(db_name)
            if database is None:
                return None
            
            # Convert string ID back to ObjectId for MongoDB
            from bson import ObjectId
            try:
                mongo_id = ObjectId(doc_id)
            except:
                mongo_id = doc_id
            
            collection = database[collection_name]
            doc = collection.find_one({'_id': mongo_id})
            
            if doc:
                # Process the document using existing logic
                result_type = collection_name.rstrip('s')  # Remove plural 's'
                return self._process_search_result(doc, result_type, db_name, collection_name)
            
            return None
            
        except Exception as e:
            logger.error(f"Error converting semantic result: {e}")
            return None
    
    def get_preindex_stats(self) -> Dict[str, Any]:
        """Get statistics about the pre-built index"""
        if not self.use_preindex or not self.preindex_ranker:
            return {
                'enabled': False,
                'reason': 'Pre-indexing not available or not enabled'
            }
        
        try:
            stats = self.preindex_ranker.get_cache_stats()
            stats['enabled'] = True
            return stats
        except Exception as e:
            return {
                'enabled': False,
                'error': str(e)
            }
    
    def semantic_search_only(self, query: str, k: int = 50, 
                           processed_query: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Perform semantic search only using pre-built index"""
        if not self.use_preindex or not self.preindex_ranker:
            logger.warning("Semantic search not available - pre-index not loaded")
            return []
        
        try:
            return self.preindex_ranker.search_similar_documents(query, k, processed_query)
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def _extract_search_parameters(self, processed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and prepare search parameters from processed query"""
        keywords = processed_query.get('keywords', {})
        entities = processed_query.get('entities', {})
        academic_fields = processed_query.get('academic_fields', {})
        intent = processed_query.get('intent', {})
        
        # Handle both old and new formats for entities
        if isinstance(entities, list):
            entities = {}  # Convert list to empty dict for backward compatibility
        
        # Combine all search terms
        all_terms = []
        all_terms.extend(keywords.get('primary', []))
        all_terms.extend(keywords.get('secondary', []))
        all_terms.extend(keywords.get('technical_terms', []))
        all_terms.extend(entities.get('technologies', []))
        all_terms.extend(entities.get('concepts', []))
        
        # Also add basic terms from processed_query for simple queries
        all_terms.extend(processed_query.get('primary_terms', []))
        all_terms.extend(processed_query.get('secondary_terms', []))
        all_terms.extend(processed_query.get('all_terms', []))
        
        # Handle intent format
        if isinstance(intent, str):
            intent_value = intent
        else:
            intent_value = intent.get('primary_intent', 'general_search')
        
        # Create search parameters
        search_params = {
            'primary_terms': keywords.get('primary', []) + processed_query.get('primary_terms', []),
            'secondary_terms': keywords.get('secondary', []) + keywords.get('technical_terms', []) + processed_query.get('secondary_terms', []),
            'all_terms': list(set(all_terms)),  # Remove duplicates
            'people': entities.get('people', []),
            'organizations': entities.get('organizations', []),
            'locations': entities.get('locations', []),
            'technologies': entities.get('technologies', []),
            'concepts': entities.get('concepts', []),
            'academic_fields': [academic_fields.get('primary_field', '')] + academic_fields.get('related_fields', []),
            'specializations': academic_fields.get('specializations', []),
            'intent': intent_value,
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
    
    def _search_funding_system(self, search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search for funding institutions and research projects that match the query"""
        results = []
        
        try:
            # Step 1: Search research projects that match the user's query
            projects_collection = self.db_manager.get_collection('funding', 'research_projects')
            if projects_collection is None:
                logger.warning("Research projects collection not available")
                return results
            
            # Build query for research projects
            query = self._build_mongodb_query(search_params, [
                'title', 'background.problem', 'background.importance', 'background.hypotheses',
                'objectives', 'field_category', 'field_group', 'field_area'
            ])
            
            if not query:
                return results
            
            # Find matching research projects
            matching_projects = list(projects_collection.find(query))
            logger.info(f"Found {len(matching_projects)} research projects matching query")
            
            if not matching_projects:
                return results
            
            # Step 2: Get funding records for these projects to find institution IDs
            funding_records_collection = self.db_manager.get_collection('funding', 'funding_records')
            institutions_collection = self.db_manager.get_collection('funding', 'institutions')
            
            if funding_records_collection is None or institutions_collection is None:
                logger.warning("Funding records or institutions collection not available")
                return results
            
            # Collect project IDs
            project_ids = [project['_id'] for project in matching_projects]
            
            # Find funding records for these projects
            funding_records = list(funding_records_collection.find({
                'research_project_id': {'$in': project_ids}
            }))
            
            logger.info(f"Found {len(funding_records)} funding records for matching projects")
            
            # Step 3: Group by institution and collect institution details
            institution_funding_map = {}
            
            for record in funding_records:
                institution_id = record.get('institution_id')
                project_id = record.get('research_project_id')
                
                if institution_id and project_id:
                    if institution_id not in institution_funding_map:
                        institution_funding_map[institution_id] = {
                            'funding_records': [],
                            'related_projects': [],
                            'total_funding': 0
                        }
                    
                    # Add funding record
                    institution_funding_map[institution_id]['funding_records'].append(record)
                    institution_funding_map[institution_id]['total_funding'] += record.get('amount', 0)
                    
                    # Find the corresponding project details
                    matching_project = next((p for p in matching_projects if p['_id'] == project_id), None)
                    if matching_project:
                        institution_funding_map[institution_id]['related_projects'].append({
                            'project_id': str(project_id),
                            'title': matching_project.get('title', ''),
                            'field_category': matching_project.get('field_category', ''),
                            'field_group': matching_project.get('field_group', ''),
                            'field_area': matching_project.get('field_area', ''),
                            'status': matching_project.get('status', ''),
                            'budget_requested': matching_project.get('budget_requested', 0),
                            'background': matching_project.get('background', {}),
                            'objectives': matching_project.get('objectives', []),
                            'funding_amount': record.get('amount', 0),
                            'disbursed_on': str(record.get('disbursed_on', '')),
                            'funding_notes': record.get('notes', '')
                        })
            
            # Step 4: Create institution results
            for institution_id, funding_data in institution_funding_map.items():
                try:
                    # Get institution details (with fallback for broken IDs)
                    institution_doc = institutions_collection.find_one({'_id': institution_id})
                    
                    # If institution not found, try to fix broken ID pattern
                    if institution_doc is None:
                        # Convert broken ID to correct format: 665500000000000000100000 -> 665500000000000001000000
                        fixed_institution_id = self._fix_broken_institution_id(institution_id)
                        if fixed_institution_id:
                            institution_doc = institutions_collection.find_one({'_id': fixed_institution_id})
                            if institution_doc:
                                logger.info(f"Fixed broken institution ID: {institution_id} -> {fixed_institution_id}")
                                institution_id = fixed_institution_id  # Use the fixed ID for the result
                    
                    if institution_doc is None:
                        continue
                    
                    # Calculate relevance score based on number of related projects and funding amount
                    related_projects_count = len(funding_data['related_projects'])
                    total_funding = funding_data['total_funding']
                    
                    # Base relevance score (0.3-0.9 range)
                    relevance_score = min(0.9, 0.3 + (related_projects_count * 0.15) + min(0.3, total_funding / 1000000))
                    
                    # Create institution result
                    institution_result = {
                        'id': str(institution_id),
                        'type': 'funding_institution',
                        'database': 'funding',
                        'collection': 'institutions',
                        'title': institution_doc.get('name', 'Unknown Institution'),
                        'author': '',  # Not applicable for institutions
                        'ranking_score': relevance_score,
                        'metadata': {
                            'type': institution_doc.get('type', ''),
                            'country': institution_doc.get('country', ''),
                            'email': institution_doc.get('email', ''),
                            'tel_no': institution_doc.get('tel_no', ''),
                            'fax_no': institution_doc.get('fax_no', ''),
                            'related_projects_count': related_projects_count,
                            'total_funding_for_query': total_funding
                        },
                        'funding_info': {
                            'related_projects': funding_data['related_projects'],
                            'total_funding_amount': total_funding,
                            'funding_count': len(funding_data['funding_records'])
                        }
                    }
                    
                    # Create engaging description focused on query relevance
                    description_parts = []
                    description_parts.append(f"{institution_doc.get('type', 'Institution')} based in {institution_doc.get('country', 'Unknown')}")
                    description_parts.append(f"Funding {related_projects_count} project{'s' if related_projects_count != 1 else ''} related to your search")
                    description_parts.append(f"Total relevant funding: ${total_funding:,.0f}")
                    
                    # Add field focus if available
                    field_categories = set()
                    for project in funding_data['related_projects']:
                        if project.get('field_category'):
                            field_categories.add(project['field_category'])
                    
                    if field_categories:
                        description_parts.append(f"Focus areas: {', '.join(list(field_categories)[:2])}")
                    
                    institution_result['description'] = ' | '.join(description_parts)
                    institution_result['snippet'] = institution_result['description']
                    
                    results.append(institution_result)
                    
                except Exception as e:
                    logger.warning(f"Error processing institution {institution_id}: {e}")
                    continue
            
            # If no institutions found but we have matching projects, provide sample institutions
            if len(results) == 0 and len(matching_projects) > 0:
                logger.info("No valid funding records found, providing sample institutions for demonstration")
                
                # Get a few sample institutions to show funding capability exists
                sample_institutions = list(institutions_collection.find().limit(5))
                
                for institution in sample_institutions:
                    institution_result = self._process_search_result(
                        institution, 'funding_institution', 'funding', 'institutions'
                    )
                    if institution_result:
                        # Add context that this is related to the query through research projects
                        institution_result['ranking_score'] = 0.3  # Lower score since it's indirect
                        institution_result['description'] += f" (Related to {len(matching_projects)} research projects in this domain)"
                        
                        # Add 1-2 sample related projects for this institution
                        related_projects = []
                        for project in matching_projects[:2]:  # Limit to 2 projects
                            project_info = {
                                'id': str(project['_id']),
                                'title': project.get('title', 'Untitled Project'),
                                'objectives': project.get('objectives', [])[:2] if project.get('objectives') else [],  # First 2 objectives
                                'field_category': project.get('field_category', ''),
                                'budget_requested': project.get('budget_requested', 0)
                            }
                            related_projects.append(project_info)
                        
                        institution_result['related_projects'] = related_projects
                        results.append(institution_result)
            
            # Sort by relevance score (institutions with more relevant projects first)
            results.sort(key=lambda x: x.get('ranking_score', 0), reverse=True)
            
            logger.info(f"Returning {len(results)} funding institutions as results")
            
        except Exception as e:
            logger.error(f"Error in funding system search: {str(e)}")
        
        return results
    
    def _fix_broken_institution_id(self, broken_id):
        """
        Fix broken institution IDs by converting the pattern:
        665500000000000000100000 -> 665500000000000001000000
        (Insert '1' at position 15)
        """
        try:
            id_str = str(broken_id)
            if len(id_str) == 24 and id_str[14] == '0':  # Check if it matches broken pattern
                # Insert '1' at position 15 (after the 14th character)
                fixed_id_str = id_str[:14] + '1' + id_str[14:]
                from bson import ObjectId
                return ObjectId(fixed_id_str)
        except Exception as e:
            logger.debug(f"Could not fix institution ID {broken_id}: {e}")
        return None
    
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
            
            elif result_type == 'research_project':
                result['title'] = doc.get('title', '')
                background = doc.get('background', {})
                objectives = doc.get('objectives', [])
                
                # Create description from background and objectives
                desc_parts = []
                if background.get('problem'):
                    desc_parts.append(f"Problem: {background['problem']}")
                if background.get('importance'):
                    desc_parts.append(f"Importance: {background['importance']}")
                if objectives:
                    objectives_str = ', '.join(objectives) if isinstance(objectives, list) else str(objectives)
                    desc_parts.append(f"Objectives: {objectives_str}")
                
                result['description'] = ' | '.join(desc_parts)
                result['metadata'] = {
                    'status': doc.get('status', ''),
                    'field_category': doc.get('field_category', ''),
                    'field_group': doc.get('field_group', ''),
                    'field_area': doc.get('field_area', ''),
                    'budget_requested': doc.get('budget_requested', 0),
                    'submission_date': str(doc.get('submission_date', '')),
                    'methodology': doc.get('methodology', {})
                }
            
            elif result_type == 'funding_institution':
                result['title'] = doc.get('name', '')
                result['description'] = f"{doc.get('type', '')} based in {doc.get('country', '')}"
                result['metadata'] = {
                    'type': doc.get('type', ''),
                    'country': doc.get('country', ''),
                    'email': doc.get('email', ''),
                    'tel_no': doc.get('tel_no', ''),
                    'fax_no': doc.get('fax_no', '')
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
                'original_query': processed_query.get('_metadata', {}).get('original_query', '') or processed_query.get('original_query', ''),
                'corrected_query': processed_query.get('corrected_query', ''),
                'intent': processed_query.get('intent', 'general_search') if isinstance(processed_query.get('intent'), str) else processed_query.get('intent', {}).get('primary_intent', 'general_search'),
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