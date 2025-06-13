#!/usr/bin/env python3
"""
Advanced ranking engine for APOSSS with multiple ranking algorithms
"""
import logging
import math
import re
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter, defaultdict
from datetime import datetime
from .llm_processor import LLMProcessor

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from .embedding_ranker import EmbeddingRanker
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    EmbeddingRanker = None

try:
    from .ltr_ranker import LTRRanker
    LTR_AVAILABLE = True
except ImportError:
    LTR_AVAILABLE = False
    LTRRanker = None

try:
    from .knowledge_graph import KnowledgeGraph
    KNOWLEDGE_GRAPH_AVAILABLE = True
except ImportError:
    KNOWLEDGE_GRAPH_AVAILABLE = False
    KnowledgeGraph = None

try:
    from .llm_ranker import LLMRanker
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    LLMRanker = None

logger = logging.getLogger(__name__)

class RankingEngine:
    """
    Advanced ranking engine with multiple algorithms:
    - Heuristic ranking based on keyword matching and metadata
    - TF-IDF with cosine similarity
    - Intent alignment based on LLM analysis
    - Real-time embedding similarity
    - Learning-to-Rank (LTR) with XGBoost
    - Graph-based features from knowledge graph
    """
    
    def __init__(self, llm_processor: LLMProcessor, use_embedding: bool = True, use_ltr: bool = True):
        """Initialize the ranking engine with multiple rankers"""
        self.use_embedding = use_embedding and EMBEDDING_AVAILABLE
        self.use_ltr = use_ltr and LTR_AVAILABLE
        
        # Initialize knowledge graph
        try:
            self.knowledge_graph = KnowledgeGraph() if KNOWLEDGE_GRAPH_AVAILABLE else None
            if self.knowledge_graph:
                logger.info("Knowledge graph initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize knowledge graph: {e}")
            self.knowledge_graph = None
        
        # Initialize embedding ranker
        if self.use_embedding:
            try:
                self.embedding_ranker = EmbeddingRanker()
                logger.info("Embedding ranker initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize embedding ranker: {e}")
                self.use_embedding = False
                self.embedding_ranker = None
        else:
            self.embedding_ranker = None
        
        # Initialize LTR ranker
        if self.use_ltr:
            try:
                self.ltr_ranker = LTRRanker()
                logger.info("LTR ranker initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize LTR ranker: {e}")
                self.use_ltr = False
                self.ltr_ranker = None
        else:
            self.ltr_ranker = None
        
        # Initialize TF-IDF vectorizer
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=1000,
                ngram_range=(1, 2),
                lowercase=True
            )
        else:
            self.tfidf_vectorizer = None
        
        # Initialize LLM ranker
        if LLM_AVAILABLE:
            self.llm_ranker = LLMRanker(llm_processor)
            logger.info("LLM ranker initialized successfully")
        else:
            self.llm_ranker = None
        
        logger.info(f"Ranking engine initialized - Embedding: {self.use_embedding}, LTR: {self.use_ltr}, TF-IDF: {SKLEARN_AVAILABLE}")
    
    def rank_search_results(self, search_results: Dict[str, Any], 
                          processed_query: Dict[str, Any],
                          user_feedback_data: Dict[str, Any] = None,
                          ranking_mode: str = "hybrid",
                          user_personalization_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Rank search results using multiple algorithms
        
        Args:
            search_results: Raw search results from search engine
            processed_query: LLM-processed query with extracted features
            user_feedback_data: Historical user feedback data
            ranking_mode: "hybrid", "ltr_only", or "traditional"
            user_personalization_data: User preferences and interaction patterns
            
        Returns:
            Ranked search results with scores and explanations
        """
        try:
            results = search_results.get('results', [])
            original_query = processed_query.get('_metadata', {}).get('original_query', '')
            
            if not results:
                return search_results
            
            logger.info(f"Ranking {len(results)} search results using mode: {ranking_mode}")
            
            # Build knowledge graph from search results
            if self.knowledge_graph:
                for result in results:
                    result_type = result.get('type', '').lower()
                    result_id = result.get('id')
                    
                    if not result_id:
                        continue
                    
                    if result_type in ['paper', 'article', 'publication']:
                        self.knowledge_graph.add_paper(result_id, result)
                    elif result_type in ['expert', 'author', 'researcher']:
                        self.knowledge_graph.add_expert(result_id, result)
                    elif result_type in ['equipment', 'resource', 'facility']:
                        self.knowledge_graph.add_equipment(result_id, result)
                
                # Calculate PageRank scores
                self.knowledge_graph.calculate_pagerank()
                logger.info("Knowledge graph built from search results")
            
            # Calculate different types of scores
            heuristic_scores = self._calculate_heuristic_scores(results, processed_query)
            tfidf_scores = self._calculate_tfidf_scores(original_query, results)
            intent_scores = self._calculate_intent_scores(results, processed_query)
            
            # Calculate personalization scores
            personalization_scores = self._calculate_personalization_scores(
                results, user_personalization_data, processed_query
            )
            
            # Calculate embedding similarity scores (real-time)
            embedding_scores = []
            if self.use_embedding and self.embedding_ranker:
                try:
                    embedding_scores = self.embedding_ranker.calculate_realtime_similarity(
                        original_query, results, processed_query, use_cache=True
                    )
                    logger.info(f"Calculated real-time embedding scores for {len(embedding_scores)} results")
                except Exception as e:
                    logger.warning(f"Error calculating embedding scores: {e}")
                    embedding_scores = [0.0] * len(results)
            else:
                embedding_scores = [0.0] * len(results)
            
            # Prepare current scores for LTR
            current_scores = {
                'heuristic': heuristic_scores,
                'tfidf': tfidf_scores,
                'intent': intent_scores,
                'embedding': embedding_scores,
                'personalization': personalization_scores
            }
            
            # Apply ranking based on mode
            if ranking_mode == "ltr_only" and self.use_ltr and self.ltr_ranker and self.ltr_ranker.is_trained:
                # Use LTR only
                ranked_results = self.ltr_ranker.rank_results(
                    original_query, results, processed_query, current_scores, user_feedback_data
                )
                ranking_algorithm = "Learning-to-Rank (XGBoost)"
                score_components = ["LTR Score"]
                
            elif ranking_mode == "hybrid" and self.use_ltr and self.ltr_ranker and self.ltr_ranker.is_trained:
                # Hybrid: LTR + traditional weights
                ltr_results = self.ltr_ranker.rank_results(
                    original_query, results, processed_query, current_scores, user_feedback_data
                )
                
                # Combine LTR with traditional hybrid scoring (including personalization)
                for i, result in enumerate(ltr_results):
                    # Traditional hybrid score with personalization
                    if self.use_embedding:
                        traditional_score = (
                            0.20 * heuristic_scores[i] +
                            0.20 * tfidf_scores[i] +
                            0.20 * intent_scores[i] +
                            0.20 * embedding_scores[i] +
                            0.20 * personalization_scores[i]
                        )
                    else:
                        traditional_score = (
                            0.25 * heuristic_scores[i] +
                            0.25 * tfidf_scores[i] +
                            0.25 * intent_scores[i] +
                            0.25 * personalization_scores[i]
                        )
                    
                    # Combine with LTR score (weighted)
                    ltr_score = result.get('ltr_score', 0.5)
                    result['ranking_score'] = 0.7 * ltr_score + 0.3 * traditional_score
                    
                    # Store score breakdown
                    result['score_breakdown'] = {
                        'heuristic_score': heuristic_scores[i],
                        'tfidf_score': tfidf_scores[i],
                        'bm25_score': self._calculate_bm25_score(original_query, result),
                        'intent_score': intent_scores[i],
                        'embedding_score': embedding_scores[i],
                        'personalization_score': personalization_scores[i],
                        'ltr_score': ltr_score,
                        'traditional_score': traditional_score,
                        'graph_authority': self.knowledge_graph.get_authority_score(result['id']) if self.knowledge_graph else 0.0,
                        'graph_connection': self.knowledge_graph.get_connection_strength(result['id'], f"keyword_{original_query.lower()}") if self.knowledge_graph else 0.0,
                        'graph_pagerank': self.knowledge_graph.get_node_pagerank(result['id']) if self.knowledge_graph else 0.0
                    }
                
                # Re-sort by combined score
                ranked_results = sorted(ltr_results, key=lambda x: x['ranking_score'], reverse=True)
                ranking_algorithm = "Hybrid (LTR + Traditional + Personalization)"
                score_components = ["LTR", "Heuristic", "TF-IDF", "Intent", "Embedding", "Personalization"]
                
            else:
                # Traditional hybrid ranking with personalization
                for i, result in enumerate(results):
                    # Traditional weights (updated with embedding and personalization)
                    if self.use_embedding:
                        result['ranking_score'] = (
                            0.20 * heuristic_scores[i] +
                            0.20 * tfidf_scores[i] +
                            0.20 * intent_scores[i] +
                            0.20 * embedding_scores[i] +
                            0.20 * personalization_scores[i]
                        )
                    else:
                        result['ranking_score'] = (
                            0.25 * heuristic_scores[i] +
                            0.25 * tfidf_scores[i] +
                            0.25 * intent_scores[i] +
                            0.25 * personalization_scores[i]
                        )
                    
                    # Store score breakdown
                    result['score_breakdown'] = {
                        'heuristic_score': heuristic_scores[i],
                        'tfidf_score': tfidf_scores[i],
                        'bm25_score': self._calculate_bm25_score(original_query, result),
                        'intent_score': intent_scores[i],
                        'embedding_score': embedding_scores[i] if self.use_embedding else 0.0,
                        'personalization_score': personalization_scores[i],
                        'graph_authority': self.knowledge_graph.get_authority_score(result['id']) if self.knowledge_graph else 0.0,
                        'graph_connection': self.knowledge_graph.get_connection_strength(result['id'], f"keyword_{original_query.lower()}") if self.knowledge_graph else 0.0,
                        'graph_pagerank': self.knowledge_graph.get_node_pagerank(result['id']) if self.knowledge_graph else 0.0
                    }
                
                # Sort by combined score
                ranked_results = sorted(results, key=lambda x: x['ranking_score'], reverse=True)
                ranking_algorithm = f"Traditional Hybrid with Personalization ({'with' if self.use_embedding else 'without'} Embedding)"
                score_components = ["Heuristic", "TF-IDF", "Intent", "Personalization"] + (["Embedding"] if self.use_embedding else [])
            
            # Add final ranks
            for i, result in enumerate(ranked_results):
                result['rank'] = i + 1
            
            # Categorize results by relevance
            categorized_results = self._categorize_by_relevance(ranked_results)
            
            # Update search results
            search_results['results'] = ranked_results
            search_results['categorized_results'] = categorized_results
            search_results['ranking_metadata'] = {
                'ranking_algorithm': ranking_algorithm,
                'score_components': score_components,
                'total_ranked': len(ranked_results),
                'ltr_enabled': self.use_ltr and self.ltr_ranker and self.ltr_ranker.is_trained,
                'embedding_enabled': self.use_embedding,
                'personalization_enabled': user_personalization_data is not None,
                'ranking_mode': ranking_mode
            }
            
            logger.info(f"Ranking completed using {ranking_algorithm}")
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error in ranking search results: {str(e)}")
            # Return original results if ranking fails
            return search_results
    
    def train_ltr_model(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train the LTR model with user feedback data
        
        Args:
            training_data: List of training examples with relevance labels
            
        Returns:
            Training statistics
        """
        if not self.use_ltr or not self.ltr_ranker:
            raise ValueError("LTR ranker not available")
        
        return self.ltr_ranker.train_model(training_data)
    
    def get_ltr_stats(self) -> Dict[str, Any]:
        """Get LTR model statistics"""
        if not LTR_AVAILABLE:
            return {
                'ltr_available': False,
                'reason': 'XGBoost/LTR dependencies not available'
            }
        
        if not self.use_ltr or not self.ltr_ranker:
            return {
                'ltr_available': False,
                'reason': 'LTR ranker failed to initialize'
            }
        
        return self.ltr_ranker.get_model_stats()
    
    def clear_embedding_cache(self) -> bool:
        """Clear embedding cache"""
        if self.use_embedding and self.embedding_ranker:
            return self.embedding_ranker.clear_cache()
        return False
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding system statistics"""
        if not self.use_embedding or not self.embedding_ranker:
            return {
                'embedding_enabled': False,
                'reason': 'Embedding ranker not initialized'
            }
        
        return {
            'embedding_enabled': True,
            'model_name': getattr(self.embedding_ranker, 'model_name', 'unknown'),
            'embedding_dimension': getattr(self.embedding_ranker, 'embedding_dimension', 'unknown'),
            'cache_directory': getattr(self.embedding_ranker, 'cache_dir', 'unknown'),
            'total_vectors': len(getattr(self.embedding_ranker, 'document_cache', {})),
            'cache_size': len(getattr(self.embedding_ranker, 'embedding_cache', {}))
        }
    
    def warm_up_embedding_cache(self, sample_documents: List[Dict[str, Any]]) -> bool:
        """Warm up the embedding cache with sample documents"""
        if not self.use_embedding or not self.embedding_ranker:
            return False
        
        try:
            # Use the embedding ranker's warm_up_cache method
            success = self.embedding_ranker.warm_up_cache(sample_documents)
            
            if success:
                logger.info(f"Cache warmed up with {len(sample_documents)} documents")
            else:
                logger.warning(f"Cache warm-up failed for {len(sample_documents)} documents")
            
            return success
            
        except Exception as e:
            logger.error(f"Error warming up cache: {e}")
            return False
    
    def _calculate_heuristic_scores(self, results: List[Dict[str, Any]], 
                                  processed_query: Dict[str, Any]) -> List[float]:
        """Calculate heuristic-based relevance scores"""
        scores = []
        
        # Extract query components
        keywords = processed_query.get('keywords', {})
        primary_keywords = keywords.get('primary', [])
        secondary_keywords = keywords.get('secondary', [])
        
        # Combine all keywords
        all_keywords = primary_keywords + secondary_keywords
        
        for result in results:
            score = 0.0
            
            # Text fields to search
            title = result.get('title', '').lower()
            description = result.get('description', '').lower()
            author = result.get('author', '').lower()
            
            # Keyword matching with weights
            for keyword in primary_keywords:
                keyword_lower = keyword.lower()
                
                # Title matches (highest weight)
                if keyword_lower in title:
                    score += 0.4
                
                # Description matches
                if keyword_lower in description:
                    score += 0.3
                
                # Author matches
                if keyword_lower in author:
                    score += 0.2
            
            # Secondary keyword matching (lower weight)
            for keyword in secondary_keywords:
                keyword_lower = keyword.lower()
                
                if keyword_lower in title:
                    score += 0.2
                if keyword_lower in description:
                    score += 0.15
                if keyword_lower in author:
                    score += 0.1
            
            # Metadata bonuses
            metadata = result.get('metadata', {})
            
            # Recent publication bonus
            if 'publication_date' in metadata or 'year' in metadata:
                try:
                    year = metadata.get('year', 0)
                    if isinstance(year, str):
                        year = int(year) if year.isdigit() else 0
                    
                    current_year = datetime.now().year
                    if year >= current_year - 5:  # Recent publications
                        score += 0.1
                except (ValueError, TypeError):
                    pass
            
            # Availability bonus
            status = metadata.get('status', '').lower()
            if status in ['available', 'active', 'published']:
                score += 0.05
            
            # Normalize score
            max_possible_score = len(primary_keywords) * 0.9 + len(secondary_keywords) * 0.45 + 0.15
            normalized_score = min(score / max_possible_score, 1.0) if max_possible_score > 0 else 0.0
            
            scores.append(normalized_score)
        
        return scores
    
    def _calculate_tfidf_scores(self, query: str, results: List[Dict[str, Any]]) -> List[float]:
        """Calculate TF-IDF cosine similarity scores"""
        if not SKLEARN_AVAILABLE or not self.tfidf_vectorizer:
            return [0.0] * len(results)
        
        try:
            # Prepare documents
            documents = []
            for result in results:
                doc_text = f"{result.get('title', '')} {result.get('description', '')} {result.get('author', '')}"
                documents.append(doc_text)
            
            # Add query to documents for vectorization
            all_texts = [query] + documents
            
            # Fit and transform
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            
            # Calculate similarity between query and each document
            query_vector = tfidf_matrix[0:1]  # First row is the query
            document_vectors = tfidf_matrix[1:]  # Rest are documents
            
            similarities = cosine_similarity(query_vector, document_vectors)[0]
            
            return similarities.tolist()
            
        except Exception as e:
            logger.warning(f"Error calculating TF-IDF scores: {e}")
            return [0.0] * len(results)
    
    def _calculate_intent_scores(self, results: List[Dict[str, Any]], 
                               processed_query: Dict[str, Any]) -> List[float]:
        """Calculate intent alignment scores"""
        scores = []
        
        # Extract intent information
        intent = processed_query.get('intent', {})
        primary_intent = intent.get('primary_intent', '').lower()
        
        # Intent-to-resource type mapping
        intent_resource_mapping = {
            'find_expert': ['expert', 'certificate'],
            'find_literature': ['book', 'journal', 'article'],
            'find_research': ['article', 'thesis', 'project'],
            'find_equipment': ['equipment', 'material'],
            'find_data': ['article', 'project', 'thesis'],
            'general_search': ['book', 'journal', 'article', 'expert']
        }
        
        preferred_types = intent_resource_mapping.get(primary_intent, ['book', 'journal', 'article'])
        
        for result in results:
            score = 0.0
            result_type = result.get('type', '').lower()
            
            # Base intent alignment
            if result_type in preferred_types:
                score = 0.8
            else:
                score = 0.4
            
            # Academic field matching
            academic_fields = processed_query.get('academic_fields', {})
            primary_field = academic_fields.get('primary_field', '').lower()
            related_fields = [f.lower() for f in academic_fields.get('related_fields', [])]
            
            result_category = result.get('metadata', {}).get('category', '').lower()
            
            # Field matching bonus
            if primary_field and primary_field in result_category:
                score += 0.2
            elif any(field in result_category for field in related_fields):
                score += 0.1
            
            # Intent confidence weighting
            confidence = intent.get('confidence', 0.5)
            score = score * confidence + 0.5 * (1 - confidence)  # Blend with neutral score
            
            scores.append(min(score, 1.0))
        
        return scores
    
    def _calculate_personalization_scores(self, results: List[Dict[str, Any]], 
                                         user_personalization_data: Dict[str, Any],
                                         processed_query: Dict[str, Any]) -> List[float]:
        """Calculate personalization scores based on user preferences and interaction history"""
        scores = []
        
        # If no personalization data, return neutral scores
        if not user_personalization_data:
            return [0.5] * len(results)
        
        # Extract user data
        user_preferences = user_personalization_data.get('preferences', {})
        user_interactions = user_personalization_data.get('interaction_history', [])
        user_profile = user_personalization_data.get('profile', {})
        
        # Analyze user's historical preferences
        preferred_types = self._analyze_user_preferred_types(user_interactions)
        preferred_fields = self._analyze_user_preferred_fields(user_interactions, user_profile)
        preferred_authors = self._analyze_user_preferred_authors(user_interactions)
        
        for result in results:
            score = 0.5  # Start with neutral score
            
            # === TYPE PREFERENCE SCORING ===
            result_type = result.get('type', '').lower()
            if result_type in preferred_types:
                type_preference = preferred_types[result_type]
                score += 0.15 * type_preference  # Up to 15% boost
            
            # === FIELD/CATEGORY PREFERENCE SCORING ===
            result_category = result.get('metadata', {}).get('category', '').lower()
            for field, preference_score in preferred_fields.items():
                if field.lower() in result_category:
                    score += 0.15 * preference_score  # Up to 15% boost
                    break
            
            # === AUTHOR PREFERENCE SCORING ===
            result_author = result.get('author', '').lower()
            if result_author in preferred_authors:
                author_preference = preferred_authors[result_author]
                score += 0.10 * author_preference  # Up to 10% boost
            
            # === RECENCY PREFERENCE SCORING ===
            recency_preference = user_preferences.get('recency_preference', 0.5)
            if recency_preference > 0.6:  # User prefers recent content
                pub_date = result.get('metadata', {}).get('publication_date') or result.get('metadata', {}).get('year')
                if pub_date:
                    try:
                        if isinstance(pub_date, str) and len(pub_date) == 4:  # Year only
                            pub_year = int(pub_date)
                        else:
                            pub_year = int(str(pub_date)[:4])
                        
                        current_year = datetime.now().year
                        years_old = current_year - pub_year
                        
                        if years_old <= 2:  # Very recent
                            score += 0.10 * recency_preference
                        elif years_old <= 5:  # Recent
                            score += 0.05 * recency_preference
                    except (ValueError, TypeError):
                        pass
            
            # === LANGUAGE PREFERENCE SCORING ===
            preferred_languages = user_profile.get('languages', [])
            result_language = result.get('metadata', {}).get('language', 'english').lower()
            if preferred_languages and result_language in [lang.lower() for lang in preferred_languages]:
                score += 0.05  # 5% boost for preferred language
            
            # === INSTITUTION PREFERENCE SCORING ===
            user_institution = user_profile.get('institution', '').lower()
            result_institution = result.get('metadata', {}).get('institution', '').lower()
            if user_institution and user_institution in result_institution:
                score += 0.10  # 10% boost for same institution
            
            # === COMPLEXITY PREFERENCE SCORING ===
            complexity_preference = user_preferences.get('complexity_preference', 0.5)
            # Estimate complexity based on result type and metadata
            estimated_complexity = self._estimate_content_complexity(result)
            complexity_match = 1.0 - abs(complexity_preference - estimated_complexity)
            score += 0.05 * complexity_match  # Up to 5% boost for complexity match
            
            # === QUERY SIMILARITY TO PAST INTERACTIONS ===
            query_similarity_boost = self._calculate_query_similarity_boost(
                processed_query, user_interactions
            )
            score += 0.10 * query_similarity_boost  # Up to 10% boost
            
            # === AVAILABILITY PREFERENCE ===
            availability_preference = user_preferences.get('availability_preference', 0.5)
            result_status = result.get('metadata', {}).get('status', '').lower()
            if availability_preference > 0.7 and result_status in ['available', 'active', 'published']:
                score += 0.05  # 5% boost for available items
            
            # Normalize score to [0, 1] range
            scores.append(max(0.0, min(1.0, score)))
        
        return scores
    
    def _analyze_user_preferred_types(self, interactions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze user's preferred resource types from interaction history"""
        type_counts = Counter()
        positive_interactions = 0
        
        for interaction in interactions:
            if interaction.get('action') == 'feedback' and interaction.get('metadata', {}).get('rating', 0) >= 4:
                result_type = interaction.get('metadata', {}).get('result_type', '')
                if result_type:
                    type_counts[result_type.lower()] += 1
                    positive_interactions += 1
        
        # Convert counts to preferences (normalized)
        preferences = {}
        if positive_interactions > 0:
            for resource_type, count in type_counts.items():
                preferences[resource_type] = count / positive_interactions
        
        return preferences
    
    def _analyze_user_preferred_fields(self, interactions: List[Dict[str, Any]], 
                                     user_profile: Dict[str, Any]) -> Dict[str, float]:
        """Analyze user's preferred academic fields"""
        field_preferences = {}
        
        # Start with user's stated academic fields
        academic_fields = user_profile.get('academic_fields', [])
        for field in academic_fields:
            field_preferences[field.lower()] = 0.8  # High preference for stated fields
        
        # Analyze interaction history for implicit preferences
        field_counts = Counter()
        total_positive = 0
        
        for interaction in interactions:
            if interaction.get('action') == 'search':
                # Extract fields from query (simplified)
                query = interaction.get('query', '').lower()
                for field in academic_fields:
                    if field.lower() in query:
                        field_counts[field.lower()] += 1
                        total_positive += 1
        
        # Update preferences based on interaction patterns
        if total_positive > 0:
            for field, count in field_counts.items():
                interaction_preference = count / total_positive
                existing_preference = field_preferences.get(field, 0.0)
                # Weighted combination of stated and observed preferences
                field_preferences[field] = 0.6 * existing_preference + 0.4 * interaction_preference
        
        return field_preferences
    
    def _analyze_user_preferred_authors(self, interactions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze user's preferred authors from positive feedback"""
        author_counts = Counter()
        positive_interactions = 0
        
        for interaction in interactions:
            if (interaction.get('action') == 'feedback' and 
                interaction.get('metadata', {}).get('rating', 0) >= 4):
                
                # Extract author from interaction metadata (if available)
                result_data = interaction.get('metadata', {}).get('result_data', {})
                author = result_data.get('author', '')
                if author:
                    author_counts[author.lower()] += 1
                    positive_interactions += 1
        
        # Convert to preferences
        preferences = {}
        if positive_interactions > 0:
            for author, count in author_counts.items():
                preferences[author] = count / positive_interactions
        
        return preferences
    
    def _estimate_content_complexity(self, result: Dict[str, Any]) -> float:
        """Estimate content complexity based on result metadata"""
        complexity = 0.5  # Default neutral complexity
        
        result_type = result.get('type', '').lower()
        
        # Type-based complexity estimation
        type_complexity = {
            'article': 0.8,      # Research articles tend to be complex
            'thesis': 0.9,       # Theses are typically very complex
            'book': 0.6,         # Books vary but generally accessible
            'journal': 0.7,      # Journal articles are moderately complex
            'expert': 0.4,       # Expert profiles are relatively simple
            'equipment': 0.3,    # Equipment specs are straightforward
            'material': 0.3,     # Material info is straightforward
            'project': 0.6       # Projects vary in complexity
        }
        
        complexity = type_complexity.get(result_type, 0.5)
        
        # Adjust based on description length (longer = more complex)
        description = result.get('description', '')
        if len(description) > 500:
            complexity = min(1.0, complexity + 0.1)
        elif len(description) < 100:
            complexity = max(0.0, complexity - 0.1)
        
        return complexity
    
    def _calculate_query_similarity_boost(self, processed_query: Dict[str, Any], 
                                        interactions: List[Dict[str, Any]]) -> float:
        """Calculate boost based on similarity to past successful queries"""
        current_keywords = set()
        
        # Extract keywords from current query
        keywords = processed_query.get('keywords', {})
        for keyword_list in keywords.values():
            if isinstance(keyword_list, list):
                current_keywords.update([kw.lower() for kw in keyword_list])
        
        if not current_keywords:
            return 0.0
        
        # Analyze past successful queries
        similarity_scores = []
        
        for interaction in interactions:
            if (interaction.get('action') == 'search' and 
                len(interaction.get('query', '')) > 0):
                
                # Simple keyword overlap calculation
                past_query = interaction.get('query', '').lower()
                past_keywords = set(past_query.split())
                
                if past_keywords:
                    overlap = len(current_keywords.intersection(past_keywords))
                    similarity = overlap / len(current_keywords.union(past_keywords))
                    similarity_scores.append(similarity)
        
        # Return average similarity (capped at reasonable boost)
        if similarity_scores:
            avg_similarity = sum(similarity_scores) / len(similarity_scores)
            return min(0.5, avg_similarity)  # Cap at 50% boost
        
        return 0.0
    
    def _categorize_by_relevance(self, ranked_results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize results by relevance level"""
        high_relevance = []
        medium_relevance = []
        low_relevance = []
        
        for result in ranked_results:
            score = result.get('ranking_score', 0.0)
            
            if score >= 0.7:
                high_relevance.append(result)
            elif score >= 0.4:
                medium_relevance.append(result)
            else:
                low_relevance.append(result)
        
        return {
            'high_relevance': high_relevance,
            'medium_relevance': medium_relevance,
            'low_relevance': low_relevance
        }
    
    def _calculate_bm25_score(self, query: str, result: Dict[str, Any]) -> float:
        """Calculate BM25 score for a result"""
        try:
            # Get text content
            title = result.get('title', '')
            description = result.get('description', '')
            content = f"{title} {description}"
            
            # Tokenize
            query_tokens = query.lower().split()
            content_tokens = content.lower().split()
            
            # BM25 parameters
            k1, b = 1.2, 0.75
            avg_doc_len = 100  # Assumed average document length
            
            # Calculate BM25 score
            doc_len = len(content_tokens)
            doc_freq = Counter(content_tokens)
            score = 0.0
            
            for term in query_tokens:
                tf = doc_freq.get(term, 0)
                if tf > 0:
                    idf = math.log(1000 / (1 + 1))  # Simplified IDF
                    score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
            
            # Normalize score to [0,1] range
            return min(1.0, score / 10.0)  # Assuming max score around 10
            
        except Exception as e:
            logger.warning(f"Error calculating BM25 score: {e}")
            return 0.0 