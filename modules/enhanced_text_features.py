import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from rank_bm25 import BM25Okapi
import textstat
from typing import List, Dict, Any, Tuple
import numpy as np
from collections import defaultdict

class EnhancedTextFeatures:
    def __init__(self):
        """Initialize the enhanced text features extractor"""
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Initialize BM25
        self.bm25 = None
        self.corpus = []
        self.corpus_tokens = []
    
    def extract_all_features(self, query: str, result: Dict[str, Any], 
                           processed_query: Dict[str, Any] = None) -> Dict[str, float]:
        """Extract all enhanced textual features"""
        features = {}
        
        # Get text content from result
        title = result.get('title', '')
        description = result.get('description', '')
        content = f"{title} {description}"
        
        # Extract features
        features.update(self._extract_bm25_scores(query, content))
        features.update(self._extract_ngram_features(query, content))
        features.update(self._extract_proximity_features(query, content))
        features.update(self._extract_complexity_features(content, processed_query))
        
        return features
    
    def _extract_bm25_scores(self, query: str, content: str) -> Dict[str, float]:
        """Extract BM25 scores for different parts of the content"""
        # Tokenize
        query_tokens = word_tokenize(query.lower())
        content_tokens = word_tokenize(content.lower())
        
        # Initialize BM25 if not done
        if self.bm25 is None or content not in self.corpus:
            self.corpus.append(content)
            self.corpus_tokens.append(content_tokens)
            self.bm25 = BM25Okapi(self.corpus_tokens)
        
        # Get BM25 scores
        doc_scores = self.bm25.get_scores(query_tokens)
        max_score = max(doc_scores) if doc_scores else 0
        
        return {
            'bm25_score': float(max_score),
            'bm25_normalized': float(max_score / (max_score + 1))  # Normalize to [0,1]
        }
    
    def _extract_ngram_features(self, query: str, content: str) -> Dict[str, float]:
        """Extract n-gram overlap features"""
        features = {}
        
        # Tokenize
        query_tokens = word_tokenize(query.lower())
        content_tokens = word_tokenize(content.lower())
        
        # Calculate n-gram overlaps for n=1,2,3
        for n in range(1, 4):
            # Generate n-grams
            query_ngrams = set(ngrams(query_tokens, n))
            content_ngrams = set(ngrams(content_tokens, n))
            
            # Calculate overlap
            if query_ngrams and content_ngrams:
                overlap = len(query_ngrams.intersection(content_ngrams))
                total = len(query_ngrams.union(content_ngrams))
                features[f'ngram_{n}_overlap'] = float(overlap / total)
            else:
                features[f'ngram_{n}_overlap'] = 0.0
        
        return features
    
    def _extract_proximity_features(self, query: str, content: str) -> Dict[str, float]:
        """Extract query term proximity features"""
        features = {}
        
        # Tokenize
        query_tokens = word_tokenize(query.lower())
        content_tokens = word_tokenize(content.lower())
        
        # Find positions of query terms
        term_positions = defaultdict(list)
        for i, token in enumerate(content_tokens):
            if token in query_tokens:
                term_positions[token].append(i)
        
        if term_positions:
            # Calculate minimum distance between any two query terms
            min_distances = []
            for term1 in term_positions:
                for term2 in term_positions:
                    if term1 != term2:
                        for pos1 in term_positions[term1]:
                            for pos2 in term_positions[term2]:
                                min_distances.append(abs(pos1 - pos2))
            
            if min_distances:
                features['min_term_distance'] = float(min(min_distances))
                features['avg_term_distance'] = float(sum(min_distances) / len(min_distances))
                features['proximity_score'] = float(1 / (1 + min(min_distances)))  # Normalized score
            else:
                features['min_term_distance'] = float('inf')
                features['avg_term_distance'] = 0.0
                features['proximity_score'] = 0.0
        else:
            features['min_term_distance'] = float('inf')
            features['avg_term_distance'] = 0.0
            features['proximity_score'] = 0.0
        
        return features
    
    def _extract_complexity_features(self, content: str, processed_query: Dict[str, Any] = None) -> Dict[str, float]:
        """Extract text complexity features and match with user expertise"""
        features = {}
        
        # Calculate text complexity metrics
        features['flesch_reading_ease'] = textstat.flesch_reading_ease(content)
        features['smog_index'] = textstat.smog_index(content)
        features['coleman_liau_index'] = textstat.coleman_liau_index(content)
        features['automated_readability_index'] = textstat.automated_readability_index(content)
        
        # Normalize complexity scores to [0,1]
        max_complexity = 100  # Maximum expected complexity
        features['complexity_score'] = min(1.0, features['smog_index'] / max_complexity)
        
        # Match with user expertise if available
        if processed_query and 'user_expertise' in processed_query:
            user_expertise = processed_query['user_expertise']
            complexity_diff = abs(features['complexity_score'] - user_expertise)
            features['expertise_match'] = float(1 / (1 + complexity_diff))  # Higher score for better match
        
        return features 