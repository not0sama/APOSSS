import logging
import math
import re
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import the new embedding ranker
try:
    from .embedding_ranker import EmbeddingRanker
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    EmbeddingRanker = None

logger = logging.getLogger(__name__)

class RankingEngine:
    """AI-powered ranking engine for APOSSS search results"""
    
    def __init__(self, use_embeddings: bool = True):
        """Initialize the ranking engine"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        # Initialize embedding ranker if available and requested
        self.embedding_ranker = None
        self.use_embeddings = use_embeddings and EMBEDDING_AVAILABLE
        
        if self.use_embeddings:
            try:
                self.embedding_ranker = EmbeddingRanker()
                logger.info("Embedding ranker initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize embedding ranker: {str(e)}")
                self.use_embeddings = False
        
        logger.info(f"Ranking engine initialized successfully (embeddings: {'enabled' if self.use_embeddings else 'disabled'})")
    
    def rank_search_results(self, search_results: Dict[str, Any], 
                          processed_query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rank search results using multiple scoring strategies
        
        Args:
            search_results: Raw search results from search engine
            processed_query: LLM-processed query with extracted features
            
        Returns:
            Ranked search results with scores and explanations
        """
        try:
            results = search_results.get('results', [])
            if not results:
                return search_results
            
            logger.info(f"Ranking {len(results)} search results")
            
            # Calculate different types of scores
            heuristic_scores = self._calculate_heuristic_scores(results, processed_query)
            tfidf_scores = self._calculate_tfidf_scores(results, processed_query)
            intent_scores = self._calculate_intent_scores(results, processed_query)
            
            # Calculate embedding scores if available
            embedding_scores = []
            if self.use_embeddings:
                try:
                    original_query = processed_query.get('corrected_query', '') or processed_query.get('_metadata', {}).get('original_query', '')
                    embedding_scores = self.embedding_ranker.calculate_embedding_similarity(
                        original_query, results, processed_query
                    )
                    logger.info("Embedding similarity scores calculated successfully")
                except Exception as e:
                    logger.warning(f"Failed to calculate embedding scores: {str(e)}")
                    embedding_scores = [0.0] * len(results)
            else:
                embedding_scores = [0.0] * len(results)
            
            # Combine scores and rank results
            ranked_results = self._combine_scores_and_rank(
                results, heuristic_scores, tfidf_scores, intent_scores, embedding_scores, processed_query
            )
            
            # Update search results with ranking information
            score_components = ['heuristic', 'tfidf', 'intent_alignment']
            if self.use_embeddings:
                score_components.append('embedding_similarity')
            
            search_results['results'] = ranked_results
            search_results['ranking_metadata'] = {
                'ranking_algorithm': 'hybrid_heuristic_tfidf_embedding' if self.use_embeddings else 'hybrid_heuristic_tfidf',
                'total_ranked': len(ranked_results),
                'ranking_timestamp': datetime.now().isoformat(),
                'score_components': score_components,
                'embedding_enabled': self.use_embeddings
            }
            
            logger.info("Search results ranked successfully")
            return search_results
            
        except Exception as e:
            logger.error(f"Error ranking search results: {str(e)}")
            # Return original results if ranking fails
            return search_results
    
    def _calculate_heuristic_scores(self, results: List[Dict[str, Any]], 
                                  processed_query: Dict[str, Any]) -> List[float]:
        """Calculate heuristic-based scores"""
        scores = []
        
        # Extract query terms
        keywords = processed_query.get('keywords', {})
        primary_terms = [term.lower() for term in keywords.get('primary', [])]
        secondary_terms = [term.lower() for term in keywords.get('secondary', [])]
        all_terms = primary_terms + secondary_terms
        
        for result in results:
            score = 0.0
            
            # Get text content for analysis
            title = result.get('title', '').lower()
            description = result.get('description', '').lower()
            author = result.get('author', '').lower()
            
            # Title matches (highest weight)
            for term in primary_terms:
                if term in title:
                    score += 3.0
            for term in secondary_terms:
                if term in title:
                    score += 1.5
            
            # Description matches (medium weight)
            for term in primary_terms:
                score += description.count(term) * 1.0
            for term in secondary_terms:
                score += description.count(term) * 0.5
            
            # Author matches (lower weight)
            for term in all_terms:
                if term in author:
                    score += 0.3
            
            # Keyword density bonus
            if all_terms:
                combined_text = f"{title} {description}"
                if combined_text:
                    keyword_density = sum(combined_text.count(term) for term in all_terms) / len(combined_text.split())
                    score += keyword_density * 2.0
            
            # Resource type bonus based on intent
            intent = processed_query.get('intent', {}).get('primary_intent', '')
            score += self._get_resource_type_bonus(result.get('type', ''), intent)
            
            scores.append(score)
        
        return scores
    
    def _calculate_tfidf_scores(self, results: List[Dict[str, Any]], 
                              processed_query: Dict[str, Any]) -> List[float]:
        """Calculate TF-IDF cosine similarity scores"""
        try:
            # Prepare documents for TF-IDF
            documents = []
            query_text = self._build_expanded_query(processed_query)
            
            # Add query as first document
            documents.append(query_text)
            
            # Add result documents
            for result in results:
                doc_text = self._extract_document_text(result)
                documents.append(doc_text)
            
            if len(documents) < 2:  # Need at least query + 1 document
                return [0.0] * len(results)
            
            # Calculate TF-IDF matrix
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
            
            # Calculate cosine similarity between query and each document
            query_vector = tfidf_matrix[0:1]  # First document is the query
            doc_vectors = tfidf_matrix[1:]    # Rest are the documents
            
            similarities = cosine_similarity(query_vector, doc_vectors)[0]
            
            return similarities.tolist()
            
        except Exception as e:
            logger.error(f"Error calculating TF-IDF scores: {str(e)}")
            return [0.0] * len(results)
    
    def _calculate_intent_scores(self, results: List[Dict[str, Any]], 
                               processed_query: Dict[str, Any]) -> List[float]:
        """Calculate scores based on intent alignment"""
        scores = []
        intent = processed_query.get('intent', {}).get('primary_intent', 'general_search')
        
        # Intent-based scoring weights
        intent_weights = {
            'find_research': {'article': 2.0, 'conference': 2.0, 'thesis': 1.5, 'book': 1.0},
            'find_expert': {'expert': 3.0, 'certificate': 1.5},
            'find_equipment': {'equipment': 3.0, 'material': 1.0},
            'find_materials': {'material': 3.0, 'equipment': 1.0},
            'find_literature': {'book': 2.0, 'journal': 2.0, 'article': 1.5},
            'general_search': {}  # No specific bias
        }
        
        weights = intent_weights.get(intent, {})
        
        for result in results:
            result_type = result.get('type', '')
            score = weights.get(result_type, 1.0)  # Default weight of 1.0
            scores.append(score)
        
        return scores
    
    def _combine_scores_and_rank(self, results: List[Dict[str, Any]], 
                               heuristic_scores: List[float],
                               tfidf_scores: List[float], 
                               intent_scores: List[float],
                               embedding_scores: List[float],
                               processed_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Combine different score types and rank results"""
        
        # Normalize scores to 0-1 range
        def normalize_scores(scores):
            if not scores or max(scores) == 0:
                return [0.0] * len(scores)
            max_score = max(scores)
            return [score / max_score for score in scores]
        
        norm_heuristic = normalize_scores(heuristic_scores)
        norm_tfidf = normalize_scores(tfidf_scores)
        norm_intent = normalize_scores(intent_scores)
        norm_embedding = normalize_scores(embedding_scores)
        
        # Combine scores with weights
        if self.use_embeddings:
            # Adjusted weights when embeddings are available
            heuristic_weight = 0.3
            tfidf_weight = 0.3
            intent_weight = 0.2
            embedding_weight = 0.2
        else:
            # Original weights when embeddings not available
            heuristic_weight = 0.4
            tfidf_weight = 0.4
            intent_weight = 0.2
            embedding_weight = 0.0
        
        combined_scores = []
        for i in range(len(results)):
            combined_score = (
                norm_heuristic[i] * heuristic_weight +
                norm_tfidf[i] * tfidf_weight +
                norm_intent[i] * intent_weight +
                norm_embedding[i] * embedding_weight
            )
            combined_scores.append(combined_score)
        
        # Create result tuples with scores and sort
        result_tuples = []
        for i, result in enumerate(results):
            # Add scoring information to result
            enhanced_result = result.copy()
            enhanced_result['ranking_score'] = combined_scores[i]
            enhanced_result['score_breakdown'] = {
                'heuristic_score': norm_heuristic[i],
                'tfidf_score': norm_tfidf[i],
                'intent_score': norm_intent[i],
                'embedding_score': norm_embedding[i],
                'combined_score': combined_scores[i]
            }
            result_tuples.append((combined_scores[i], enhanced_result))
        
        # Sort by combined score (descending)
        result_tuples.sort(key=lambda x: x[0], reverse=True)
        
        # Extract sorted results and add rank information
        ranked_results = []
        for rank, (score, result) in enumerate(result_tuples, 1):
            result['rank'] = rank
            ranked_results.append(result)
        
        return ranked_results
    
    def _get_resource_type_bonus(self, resource_type: str, intent: str) -> float:
        """Get bonus score based on resource type and query intent"""
        bonuses = {
            'find_research': {
                'article': 1.0, 'conference': 1.0, 'thesis': 0.8, 
                'book': 0.5, 'journal': 0.5
            },
            'find_expert': {
                'expert': 2.0, 'certificate': 0.5
            },
            'find_equipment': {
                'equipment': 2.0, 'material': 0.3
            },
            'find_materials': {
                'material': 2.0, 'equipment': 0.3
            },
            'find_literature': {
                'book': 1.5, 'journal': 1.5, 'article': 1.0
            }
        }
        
        return bonuses.get(intent, {}).get(resource_type, 0.0)
    
    def _build_expanded_query(self, processed_query: Dict[str, Any]) -> str:
        """Build expanded query text from processed query components"""
        components = []
        
        # Add original query
        components.append(processed_query.get('corrected_query', ''))
        
        # Add keywords
        keywords = processed_query.get('keywords', {})
        components.extend(keywords.get('primary', []))
        components.extend(keywords.get('secondary', []))
        components.extend(keywords.get('technical_terms', []))
        
        # Add entities
        entities = processed_query.get('entities', {})
        components.extend(entities.get('technologies', []))
        components.extend(entities.get('concepts', []))
        
        # Add synonyms
        synonyms = processed_query.get('synonyms_and_related', {})
        components.extend(synonyms.get('synonyms', []))
        components.extend(synonyms.get('related_terms', []))
        
        return ' '.join(filter(None, components))
    
    def _extract_document_text(self, result: Dict[str, Any]) -> str:
        """Extract searchable text from a result document"""
        text_components = []
        
        # Add title (with higher weight by repeating)
        title = result.get('title', '')
        if title:
            text_components.extend([title] * 3)  # Triple weight for title
        
        # Add description
        description = result.get('description', '')
        if description:
            text_components.append(description)
        
        # Add author
        author = result.get('author', '')
        if author:
            text_components.append(author)
        
        # Add metadata keywords if available
        metadata = result.get('metadata', {})
        keywords = metadata.get('keywords', [])
        if keywords:
            text_components.extend(keywords)
        
        # Add category information
        category = metadata.get('category', '')
        if category:
            text_components.append(category)
        
        return ' '.join(filter(None, text_components))

    def categorize_results(self, ranked_results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize ranked results by relevance tiers"""
        if not ranked_results:
            return {'high_relevance': [], 'medium_relevance': [], 'low_relevance': []}
        
        # Calculate score thresholds
        scores = [result.get('ranking_score', 0) for result in ranked_results]
        max_score = max(scores) if scores else 0
        
        if max_score == 0:
            return {'low_relevance': ranked_results}
        
        high_threshold = max_score * 0.7
        medium_threshold = max_score * 0.3
        
        categorized = {
            'high_relevance': [],
            'medium_relevance': [],
            'low_relevance': []
        }
        
        for result in ranked_results:
            score = result.get('ranking_score', 0)
            if score >= high_threshold:
                categorized['high_relevance'].append(result)
            elif score >= medium_threshold:
                categorized['medium_relevance'].append(result)
            else:
                categorized['low_relevance'].append(result)
        
        return categorized 
    
    def build_embedding_index(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Build FAISS index for all documents (useful for pre-indexing)
        
        Args:
            documents: List of all documents to index
            
        Returns:
            True if successful, False otherwise
        """
        if not self.use_embeddings:
            logger.warning("Embeddings not available, cannot build index")
            return False
        
        try:
            self.embedding_ranker.build_document_index(documents)
            logger.info(f"Successfully built embedding index for {len(documents)} documents")
            return True
        except Exception as e:
            logger.error(f"Failed to build embedding index: {str(e)}")
            return False
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding system"""
        if not self.use_embeddings or not self.embedding_ranker:
            return {
                'embedding_enabled': False,
                'reason': 'Embeddings not available or not initialized'
            }
        
        try:
            stats = self.embedding_ranker.get_cache_stats()
            stats['embedding_enabled'] = True
            return stats
        except Exception as e:
            return {
                'embedding_enabled': False,
                'error': str(e)
            }
    
    def clear_embedding_cache(self) -> bool:
        """Clear the embedding cache"""
        if not self.use_embeddings or not self.embedding_ranker:
            return False
        
        try:
            self.embedding_ranker.clear_cache()
            logger.info("Embedding cache cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to clear embedding cache: {str(e)}")
            return False
    
    def search_similar_documents(self, query: str, processed_query: Dict[str, Any] = None, 
                                k: int = 50) -> List[Dict[str, Any]]:
        """
        Search for similar documents using only embeddings (useful for testing)
        
        Args:
            query: Search query
            processed_query: LLM-processed query
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        if not self.use_embeddings:
            logger.warning("Embeddings not available for similarity search")
            return []
        
        try:
            return self.embedding_ranker.search_similar_documents(query, k, processed_query)
        except Exception as e:
            logger.error(f"Failed to search similar documents: {str(e)}")
            return [] 