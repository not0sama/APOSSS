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
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_dir: str = 'embeddings_cache'):
        """
        Initialize the embedding ranker
        
        Args:
            model_name: Sentence transformer model to use
            cache_dir: Directory to cache embeddings and FAISS index
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.embedding_dim = None
        self.faiss_index = None
        self.document_cache = {}  # Store document metadata with embeddings
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize sentence transformer
        try:
            logger.info(f"Loading sentence transformer model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model: {str(e)}")
            raise
        
        # Load or create FAISS index
        self._load_or_create_faiss_index()
    
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
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
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
            return np.zeros((len(texts), self.embedding_dim))
    
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
        """Build enhanced query from original + LLM processing"""
        components = [original_query]
        
        if processed_query:
            # Add corrected query
            corrected = processed_query.get('corrected_query', '')
            if corrected and corrected != original_query:
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
        """Get statistics about the embedding cache"""
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'total_vectors': self.faiss_index.ntotal if self.faiss_index else 0,
            'cache_size': len(self.document_cache),
            'cache_directory': self.cache_dir
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