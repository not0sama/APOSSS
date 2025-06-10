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

logger = logging.getLogger(__name__)

class RankingEngine:
    """
    Advanced ranking engine with multiple algorithms:
    - Heuristic ranking based on keyword matching and metadata
    - TF-IDF with cosine similarity
    - Intent alignment based on LLM analysis
    - Real-time embedding similarity
    - Learning-to-Rank (LTR) with XGBoost
    """
    
    def __init__(self, use_embedding: bool = True, use_ltr: bool = True):
        """Initialize the ranking engine with multiple rankers"""
        self.use_embedding = use_embedding and EMBEDDING_AVAILABLE
        self.use_ltr = use_ltr and LTR_AVAILABLE
        
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
        
        logger.info(f"Ranking engine initialized - Embedding: {self.use_embedding}, LTR: {self.use_ltr}, TF-IDF: {SKLEARN_AVAILABLE}")
    
    def rank_search_results(self, search_results: Dict[str, Any], 
                          processed_query: Dict[str, Any],
                          user_feedback_data: Dict[str, Any] = None,
                          ranking_mode: str = "hybrid") -> Dict[str, Any]:
        """
        Rank search results using multiple algorithms
        
        Args:
            search_results: Raw search results from search engine
            processed_query: LLM-processed query with extracted features
            user_feedback_data: Historical user feedback data
            ranking_mode: "hybrid", "ltr_only", or "traditional"
            
        Returns:
            Ranked search results with scores and explanations
        """
        try:
            results = search_results.get('results', [])
            original_query = processed_query.get('_metadata', {}).get('original_query', '')
            
            if not results:
                return search_results
            
            logger.info(f"Ranking {len(results)} search results using mode: {ranking_mode}")
            
            # Calculate different types of scores
            heuristic_scores = self._calculate_heuristic_scores(results, processed_query)
            tfidf_scores = self._calculate_tfidf_scores(original_query, results)
            intent_scores = self._calculate_intent_scores(results, processed_query)
            
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
                'embedding': embedding_scores
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
                
                # Combine LTR with traditional hybrid scoring
                for i, result in enumerate(ltr_results):
                    # Traditional hybrid score
                    traditional_score = (
                        0.25 * heuristic_scores[i] +
                        0.25 * tfidf_scores[i] +
                        0.25 * intent_scores[i] +
                        0.25 * embedding_scores[i]
                    )
                    
                    # Combine with LTR score (weighted)
                    ltr_score = result.get('ltr_score', 0.5)
                    result['ranking_score'] = 0.7 * ltr_score + 0.3 * traditional_score
                    
                    # Store score breakdown
                    result['score_breakdown'] = {
                        'heuristic_score': heuristic_scores[i],
                        'tfidf_score': tfidf_scores[i],
                        'intent_score': intent_scores[i],
                        'embedding_score': embedding_scores[i],
                        'ltr_score': ltr_score,
                        'traditional_score': traditional_score
                    }
                
                # Re-sort by combined score
                ranked_results = sorted(ltr_results, key=lambda x: x['ranking_score'], reverse=True)
                ranking_algorithm = "Hybrid (LTR + Traditional)"
                score_components = ["LTR", "Heuristic", "TF-IDF", "Intent", "Embedding"]
                
            else:
                # Traditional hybrid ranking
                for i, result in enumerate(results):
                    # Traditional weights (updated with embedding)
                    if self.use_embedding:
                        result['ranking_score'] = (
                            0.25 * heuristic_scores[i] +
                            0.25 * tfidf_scores[i] +
                            0.25 * intent_scores[i] +
                            0.25 * embedding_scores[i]
                        )
                    else:
                        result['ranking_score'] = (
                            0.35 * heuristic_scores[i] +
                            0.35 * tfidf_scores[i] +
                            0.30 * intent_scores[i]
                        )
                    
                    # Store score breakdown
                    result['score_breakdown'] = {
                        'heuristic_score': heuristic_scores[i],
                        'tfidf_score': tfidf_scores[i],
                        'intent_score': intent_scores[i],
                        'embedding_score': embedding_scores[i] if self.use_embedding else 0.0
                    }
                
                # Sort by combined score
                ranked_results = sorted(results, key=lambda x: x['ranking_score'], reverse=True)
                ranking_algorithm = f"Traditional Hybrid ({'with' if self.use_embedding else 'without'} Embedding)"
                score_components = ["Heuristic", "TF-IDF", "Intent"] + (["Embedding"] if self.use_embedding else [])
            
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