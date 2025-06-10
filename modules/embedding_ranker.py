#!/usr/bin/env python3
"""
Embedding-based semantic similarity ranker using sentence transformers and FAISS
"""
import logging
import os
import pickle
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class EmbeddingRanker:
    """Semantic similarity ranker using sentence transformers and FAISS"""
    
    def __init__(self, cache_dir: str = 'embedding_cache'):
        """Initialize the embedding ranker with multilingual sentence transformer model"""
        self.cache_dir = cache_dir
        self.model_name = 'paraphrase-multilingual-mpnet-base-v2'
        self.model = None
        self.embedding_dimension = 768  # Updated dimension for multilingual mpnet model
        self.faiss_index = None
        self.document_cache = {}  # Store document metadata with embeddings
        self.embedding_cache = {}  # Store cached embeddings
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize model
        self._initialize_model()
        
        # Try to load existing index
        self._load_cache()
    
    def _load_or_create_faiss_index(self):
        """Load existing FAISS index or create a new one"""
        index_path = os.path.join(self.cache_dir, 'faiss_index.pkl')
        cache_path = os.path.join(self.cache_dir, 'document_cache.pkl')
        
        try:
            if os.path.exists(index_path) and os.path.exists(cache_path):
                logger.info("Loading existing FAISS index and document cache")
                
                # Load FAISS index
                with open(index_path, 'rb') as f:
                    index_data = pickle.load(f)
                    self.faiss_index = faiss.deserialize_index(index_data)
                
                # Load document cache
                with open(cache_path, 'rb') as f:
                    self.document_cache = pickle.load(f)
                
                logger.info(f"Loaded index with {self.faiss_index.ntotal} vectors")
            else:
                logger.info("Creating new FAISS index")
                self._create_new_faiss_index()
                
        except Exception as e:
            logger.warning(f"Failed to load existing index, creating new one: {str(e)}")
            self._create_new_faiss_index()
    
    def _create_new_faiss_index(self):
        """Create a new FAISS index for similarity search"""
        # Use IndexFlatIP for cosine similarity (after L2 normalization)
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)
        self.document_cache = {}
        logger.info("Created new FAISS index")
    
    def calculate_embedding_similarity(self, query: str, documents: List[Dict[str, Any]], 
                                     processed_query: Dict[str, Any] = None) -> List[float]:
        """
        Calculate semantic similarity scores using embeddings
        
        Args:
            query: Original search query
            documents: List of search result documents
            processed_query: LLM-processed query with additional context
            
        Returns:
            List of similarity scores for each document
        """
        try:
            # Build enhanced query if LLM processing is available
            enhanced_query = self._build_enhanced_query(query, processed_query)
            
            # Get query embedding
            query_embedding = self._encode_text([enhanced_query])[0]
            
            # Extract and encode document texts
            document_texts = [self._extract_document_text(doc) for doc in documents]
            document_embeddings = self._encode_text(document_texts)
            
            # Calculate cosine similarity
            query_embedding = query_embedding.reshape(1, -1)
            similarities = cosine_similarity(query_embedding, document_embeddings)[0]
            
            return similarities.tolist()
            
        except Exception as e:
            logger.error(f"Error calculating embedding similarity: {str(e)}")
            return [0.0] * len(documents)
    
    def build_document_index(self, documents: List[Dict[str, Any]], batch_size: int = 100):
        """
        Build FAISS index from a collection of documents
        
        Args:
            documents: List of all documents to index
            batch_size: Number of documents to process in each batch
        """
        try:
            logger.info(f"Building FAISS index for {len(documents)} documents")
            
            # Clear existing index
            self._create_new_faiss_index()
            
            # Process documents in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
                
                # Extract texts and generate embeddings
                batch_texts = [self._extract_document_text(doc) for doc in batch]
                batch_embeddings = self._encode_text(batch_texts)
                
                # Add to FAISS index
                self.faiss_index.add(batch_embeddings)
                
                # Cache document metadata
                for j, doc in enumerate(batch):
                    doc_id = doc.get('id', f"doc_{i+j}")
                    self.document_cache[i + j] = {
                        'id': doc_id,
                        'title': doc.get('title', ''),
                        'type': doc.get('type', ''),
                        'database': doc.get('database', ''),
                        'collection': doc.get('collection', '')
                    }
            
            # Save index and cache
            self._save_faiss_index()
            logger.info(f"Successfully built index with {self.faiss_index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error building document index: {str(e)}")
            raise
    
    def search_similar_documents(self, query: str, k: int = 100, 
                                processed_query: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using FAISS index
        
        Args:
            query: Search query
            k: Number of similar documents to retrieve
            processed_query: LLM-processed query for enhancement
            
        Returns:
            List of similar documents with similarity scores
        """
        try:
            if self.faiss_index.ntotal == 0:
                logger.warning("FAISS index is empty, no documents to search")
                return []
            
            # Build enhanced query
            enhanced_query = self._build_enhanced_query(query, processed_query)
            
            # Get query embedding
            query_embedding = self._encode_text([enhanced_query])[0]
            query_embedding = query_embedding.reshape(1, -1)
            
            # Search using FAISS
            similarities, indices = self.faiss_index.search(query_embedding, min(k, self.faiss_index.ntotal))
            
            # Build results
            results = []
            for sim, idx in zip(similarities[0], indices[0]):
                if idx in self.document_cache:
                    doc_info = self.document_cache[idx].copy()
                    doc_info['similarity_score'] = float(sim)
                    doc_info['rank'] = len(results) + 1
                    results.append(doc_info)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {str(e)}")
            return []
    
    def _encode_text(self, texts: List[str]) -> np.ndarray:
        """Encode texts into embeddings"""
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            embeddings = embeddings / norms
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding texts: {str(e)}")
            return np.zeros((len(texts), self.embedding_dimension))
    
    def _extract_document_text(self, document: Dict[str, Any]) -> str:
        """Extract searchable text from a document"""
        text_parts = []
        
        # Title (high importance)
        title = document.get('title', '')
        if title:
            text_parts.extend([title] * 3)  # Triple weight for title
        
        # Description/abstract
        description = document.get('description', '')
        if description:
            text_parts.append(description)
        
        # Additional content fields
        for field in ['abstract', 'summary', 'content']:
            content = document.get(field, '')
            if content:
                text_parts.append(content)
        
        # Author information
        author = document.get('author', '')
        if author:
            text_parts.append(author)
        
        # Keywords from metadata
        metadata = document.get('metadata', {})
        keywords = metadata.get('keywords', [])
        if keywords:
            if isinstance(keywords, list):
                text_parts.extend(keywords)
            else:
                text_parts.append(str(keywords))
        
        # Category information
        category = metadata.get('category', '')
        if category:
            text_parts.append(category)
        
        return ' '.join(filter(None, text_parts))
    
    def _build_enhanced_query(self, original_query: str, processed_query: Dict[str, Any] = None) -> str:
        """Build enhanced query from original + LLM processing with multilingual support"""
        components = []
        
        # Always start with the original query for multilingual matching
        components.append(original_query)
        
        if processed_query:
            # Add translated query if available and different from original
            translation = processed_query.get('translation', {})
            translated_query = translation.get('translated_query', '')
            if translated_query and translated_query != original_query:
                components.append(translated_query)
            
            # Add corrected query
            corrected = processed_query.get('corrected_query', '')
            if corrected and corrected not in components:
                components.append(corrected)
            
            # Add keywords
            keywords = processed_query.get('keywords', {})
            for key in ['primary', 'secondary', 'technical_terms']:
                terms = keywords.get(key, [])
                if terms:
                    components.extend(terms)
            
            # Add entities
            entities = processed_query.get('entities', {})
            for key in ['technologies', 'concepts', 'organizations']:
                entity_list = entities.get(key, [])
                if entity_list:
                    components.extend(entity_list)
            
            # Add synonyms and related terms
            synonyms = processed_query.get('synonyms_and_related', {})
            for key in ['synonyms', 'related_terms']:
                syn_list = synonyms.get(key, [])
                if syn_list:
                    components.extend(syn_list[:5])  # Limit to avoid too long queries
        
        return ' '.join(filter(None, components))
    
    def _save_faiss_index(self):
        """Save FAISS index and document cache to disk"""
        try:
            index_path = os.path.join(self.cache_dir, 'faiss_index.pkl')
            cache_path = os.path.join(self.cache_dir, 'document_cache.pkl')
            
            # Save FAISS index
            with open(index_path, 'wb') as f:
                index_data = faiss.serialize_index(self.faiss_index)
                pickle.dump(index_data, f)
            
            # Save document cache
            with open(cache_path, 'wb') as f:
                pickle.dump(self.document_cache, f)
            
            logger.info("FAISS index and cache saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving FAISS index: {str(e)}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding cache with multilingual support info"""
        return {
            'model_name': self.model_name,
            'model_type': 'multilingual',
            'supported_languages': 'English, French, Spanish, German, Italian, Dutch, Portuguese, Russian, Chinese, Japanese, Korean, Arabic, and 50+ more',
            'embedding_dimension': self.embedding_dimension,
            'total_vectors': self.faiss_index.ntotal if self.faiss_index else 0,
            'cache_size': len(self.document_cache),
            'cache_directory': self.cache_dir,
            'multilingual_capable': True
        }
    
    def clear_cache(self):
        """Clear the embedding cache and FAISS index"""
        try:
            self._create_new_faiss_index()
            
            # Remove cache files
            index_path = os.path.join(self.cache_dir, 'faiss_index.pkl')
            cache_path = os.path.join(self.cache_dir, 'document_cache.pkl')
            
            for path in [index_path, cache_path]:
                if os.path.exists(path):
                    os.remove(path)
            
            logger.info("Embedding cache cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
    
    def calculate_realtime_similarity(self, query: str, results: List[Dict[str, Any]], 
                                    processed_query: Dict[str, Any] = None,
                                    use_cache: bool = True) -> List[float]:
        """
        Calculate embedding similarity scores in real-time for search results
        
        Args:
            query: Original search query
            results: List of search results to calculate similarity for
            processed_query: LLM-processed query for enhanced similarity
            use_cache: Whether to use cached embeddings for documents
            
        Returns:
            List of similarity scores (0-1) for each result
        """
        try:
            if not results:
                return []
            
            logger.info(f"Calculating real-time embedding similarity for {len(results)} results")
            
            # Step 1: Get query embedding
            query_embedding = self._get_query_embedding(query, processed_query)
            if query_embedding is None:
                logger.warning("Failed to generate query embedding")
                return [0.0] * len(results)
            
            # Step 2: Get document embeddings (with caching)
            doc_embeddings = self._get_document_embeddings_realtime(results, use_cache)
            
            # Step 3: Calculate similarities
            similarities = self._calculate_cosine_similarities(query_embedding, doc_embeddings)
            
            logger.info(f"Real-time similarity calculation completed for {len(results)} results")
            return similarities.tolist() if hasattr(similarities, 'tolist') else similarities
            
        except Exception as e:
            logger.error(f"Error in real-time similarity calculation: {str(e)}")
            return [0.0] * len(results)
    
    def _get_query_embedding(self, query: str, processed_query: Dict[str, Any] = None) -> Optional[np.ndarray]:
        """Generate embedding for search query with enhancement from processed query"""
        try:
            # Build enhanced query text
            enhanced_query = self._build_enhanced_query_text(query, processed_query)
            
            # Check cache first
            cache_key = f"query:{hash(enhanced_query)}"
            if cache_key in self.embedding_cache:
                return self.embedding_cache[cache_key]
            
            # Generate embedding
            embedding = self.model.encode([enhanced_query], convert_to_numpy=True)[0]
            
            # Cache the result
            self.embedding_cache[cache_key] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            return None
    
    def _get_document_embeddings_realtime(self, results: List[Dict[str, Any]], 
                                        use_cache: bool = True) -> List[np.ndarray]:
        """Get embeddings for documents in real-time with intelligent caching"""
        embeddings = []
        uncached_results = []
        uncached_indices = []
        
        # Step 1: Check cache for existing embeddings
        for i, result in enumerate(results):
            doc_id = result.get('id', '')
            cache_key = f"doc:{doc_id}"
            
            if use_cache and cache_key in self.embedding_cache:
                embeddings.append(self.embedding_cache[cache_key])
            else:
                # Mark for batch processing
                embeddings.append(None)  # Placeholder
                uncached_results.append(result)
                uncached_indices.append(i)
        
        # Step 2: Batch process uncached documents
        if uncached_results:
            logger.info(f"Processing {len(uncached_results)} uncached document embeddings")
            
            # Extract text for batch processing
            texts = []
            for result in uncached_results:
                doc_text = self._extract_document_text_for_embedding(result)
                texts.append(doc_text)
            
            # Generate embeddings in batch for efficiency
            try:
                batch_embeddings = self.model.encode(texts, convert_to_numpy=True, 
                                                   batch_size=32, show_progress_bar=False)
                
                # Store in cache and update embeddings list
                for i, (result, embedding) in enumerate(zip(uncached_results, batch_embeddings)):
                    doc_id = result.get('id', '')
                    cache_key = f"doc:{doc_id}"
                    
                    # Cache the embedding
                    self.embedding_cache[cache_key] = embedding
                    
                    # Update the embeddings list
                    original_index = uncached_indices[i]
                    embeddings[original_index] = embedding
                    
            except Exception as e:
                logger.error(f"Error in batch embedding generation: {str(e)}")
                # Fill with zero embeddings as fallback
                zero_embedding = np.zeros(self.embedding_dimension)
                for idx in uncached_indices:
                    embeddings[idx] = zero_embedding
        
        # Step 3: Handle any remaining None values
        zero_embedding = np.zeros(self.embedding_dimension)
        for i in range(len(embeddings)):
            if embeddings[i] is None:
                embeddings[i] = zero_embedding
        
        return embeddings
    
    def _extract_document_text_for_embedding(self, result: Dict[str, Any]) -> str:
        """Extract and prepare text from document for embedding generation"""
        text_parts = []
        
        # Title (most important - repeat for emphasis)
        title = result.get('title', '').strip()
        if title:
            text_parts.extend([title] * 2)  # Give title extra weight
        
        # Description/Abstract
        description = result.get('description', '').strip()
        if description:
            text_parts.append(description)
        
        # Author information
        author = result.get('author', '').strip()
        if author:
            text_parts.append(f"Author: {author}")
        
        # Keywords from metadata
        metadata = result.get('metadata', {})
        keywords = metadata.get('keywords', [])
        if keywords and isinstance(keywords, list):
            text_parts.append(f"Keywords: {' '.join(keywords)}")
        
        # Category information
        category = metadata.get('category', '').strip()
        if category:
            text_parts.append(f"Category: {category}")
        
        # Department/Institution context
        institution = metadata.get('institution', '') or metadata.get('university', '')
        if institution:
            text_parts.append(f"Institution: {institution}")
        
        department = metadata.get('department', '')
        if department:
            text_parts.append(f"Department: {department}")
        
        # Combine all parts
        combined_text = ' '.join(filter(None, text_parts))
        
        # Ensure we have some text
        if not combined_text:
            combined_text = result.get('type', 'document')
        
        return combined_text
    
    def _build_enhanced_query_text(self, original_query: str, processed_query: Dict[str, Any] = None) -> str:
        """Build enhanced query text using LLM processing results with multilingual support"""
        text_parts = [original_query]
        
        if processed_query:
            # Add translated query if available and different from original  
            translation = processed_query.get('translation', {})
            translated_query = translation.get('translated_query', '')
            if translated_query and translated_query != original_query:
                text_parts.append(translated_query)
            
            # Add corrected query if different
            corrected = processed_query.get('corrected_query', '')
            if corrected and corrected not in text_parts:
                text_parts.append(corrected)
            
            # Add primary keywords
            keywords = processed_query.get('keywords', {})
            primary_keywords = keywords.get('primary', [])
            if primary_keywords:
                text_parts.extend(primary_keywords)
            
            # Add technical terms
            technical_terms = keywords.get('technical_terms', [])
            if technical_terms:
                text_parts.extend(technical_terms)
            
            # Add key entities
            entities = processed_query.get('entities', {})
            technologies = entities.get('technologies', [])
            concepts = entities.get('concepts', [])
            if technologies:
                text_parts.extend(technologies)
            if concepts:
                text_parts.extend(concepts)
            
            # Add related terms
            synonyms = processed_query.get('synonyms_and_related', {})
            related_terms = synonyms.get('related_terms', [])
            if related_terms:
                text_parts.extend(related_terms[:3])  # Limit to top 3
        
        return ' '.join(filter(None, text_parts))
    
    def _calculate_cosine_similarities(self, query_embedding: np.ndarray, 
                                     doc_embeddings: List[np.ndarray]) -> np.ndarray:
        """Calculate cosine similarity between query and document embeddings"""
        try:
            # Stack document embeddings
            doc_matrix = np.stack(doc_embeddings)
            
            # Reshape query embedding for matrix operations
            query_vector = query_embedding.reshape(1, -1)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, doc_matrix)[0]
            
            # Ensure values are in [0, 1] range (cosine similarity can be [-1, 1])
            similarities = np.clip((similarities + 1) / 2, 0, 1)
            
            return similarities
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarities: {str(e)}")
            return np.zeros(len(doc_embeddings))
    
    def calculate_pairwise_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        try:
            embeddings = self.model.encode([text1, text2], convert_to_numpy=True)
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            # Normalize to [0, 1] range
            return max(0, (similarity + 1) / 2)
        except Exception as e:
            logger.error(f"Error calculating pairwise similarity: {str(e)}")
            return 0.0
    
    def get_realtime_stats(self) -> Dict[str, Any]:
        """Get statistics about real-time embedding calculations"""
        try:
            cache_stats = self.get_cache_stats()
            
            # Add real-time specific stats
            query_cache_count = len([k for k in self.embedding_cache.keys() if k.startswith('query:')])
            doc_cache_count = len([k for k in self.embedding_cache.keys() if k.startswith('doc:')])
            
            realtime_stats = {
                'realtime_enabled': True,
                'query_cache_count': query_cache_count,
                'document_cache_count': doc_cache_count,
                'total_cache_entries': len(self.embedding_cache),
                'model_ready': self.model is not None,
                'embedding_dimension': self.embedding_dimension,
                'cache_hit_efficiency': f"{doc_cache_count}/{doc_cache_count + query_cache_count}" if (doc_cache_count + query_cache_count) > 0 else "0/0"
            }
            
            # Combine with base stats
            cache_stats.update(realtime_stats)
            return cache_stats
            
        except Exception as e:
            logger.error(f"Error getting real-time stats: {str(e)}")
            return {'realtime_enabled': False, 'error': str(e)}
    
    def warm_up_cache(self, sample_results: List[Dict[str, Any]]) -> bool:
        """Pre-warm the cache with a sample of documents for better performance"""
        try:
            logger.info(f"Warming up cache with {len(sample_results)} sample documents")
            
            # Use the real-time embedding calculation to populate cache
            self._get_document_embeddings_realtime(sample_results, use_cache=True)
            
            logger.info("Cache warm-up completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error warming up cache: {str(e)}")
            return False

    def _initialize_model(self):
        """Initialize the multilingual sentence transformer model"""
        try:
            logger.info(f"Loading multilingual sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Multilingual model loaded successfully. Embedding dimension: {self.embedding_dimension}")
            logger.info("Model supports multiple languages including English, French, Spanish, German, Italian, Dutch, Portuguese, Russian, Chinese, Japanese, Korean, Arabic, and many others")
        except Exception as e:
            logger.error(f"Failed to load multilingual sentence transformer model: {str(e)}")
            raise

    def _load_cache(self):
        """Load existing FAISS index and document cache"""
        self._load_or_create_faiss_index() 