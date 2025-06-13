#!/usr/bin/env python3
"""
Learning-to-Rank (LTR) module for APOSSS using XGBoost
Advanced ranking with feature engineering and machine learning
"""
import logging
import os
import pickle
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import Counter
import re
import math
from modules.enhanced_text_features import EnhancedTextFeatures

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import ndcg_score
    from scipy.stats import pearsonr
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

logger = logging.getLogger(__name__)

class LTRRanker:
    """Learning-to-Rank ranker using XGBoost with comprehensive feature engineering"""
    
    def __init__(self, model_dir: str = 'ltr_models'):
        """Initialize the LTR ranker"""
        self.model_dir = model_dir
        self.model = None
        self.feature_names = []
        self.is_trained = False
        self.training_stats = {}
        
        # Feature engineering components
        self.feature_extractor = FeatureExtractor()
        self.enhanced_text_features = EnhancedTextFeatures()  # Add enhanced text features
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, LTR functionality disabled")
            return
        
        # Try to load existing model
        self._load_model()
        
        logger.info(f"LTR Ranker initialized (XGBoost available: {XGBOOST_AVAILABLE})")
    
    def extract_features(self, query: str, results: List[Dict[str, Any]], 
                        processed_query: Dict[str, Any] = None,
                        current_scores: Dict[str, List[float]] = None,
                        user_feedback_data: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Extract comprehensive features for LTR
        
        Args:
            query: Original search query
            results: List of search results
            processed_query: LLM-processed query
            current_scores: Current ranking scores (heuristic, tfidf, intent, embedding)
            user_feedback_data: Historical user feedback data
            
        Returns:
            DataFrame with features for each query-result pair
        """
        if not results:
            return pd.DataFrame()
        
        logger.info(f"Extracting LTR features for {len(results)} results")
        
        # Initialize feature matrix
        features_list = []
        
        for i, result in enumerate(results):
            feature_row = {}
            
            # Basic identifiers
            feature_row['query_id'] = hash(query)
            feature_row['result_id'] = result.get('id', f'result_{i}')
            feature_row['result_index'] = i
            
            # === CURRENT RANKING FEATURES ===
            if current_scores:
                feature_row['heuristic_score'] = current_scores.get('heuristic', [0.0])[i] if i < len(current_scores.get('heuristic', [])) else 0.0
                feature_row['tfidf_score'] = current_scores.get('tfidf', [0.0])[i] if i < len(current_scores.get('tfidf', [])) else 0.0
                feature_row['intent_score'] = current_scores.get('intent', [0.0])[i] if i < len(current_scores.get('intent', [])) else 0.0
                feature_row['embedding_score'] = current_scores.get('embedding', [0.0])[i] if i < len(current_scores.get('embedding', [])) else 0.0
            else:
                feature_row['heuristic_score'] = 0.0
                feature_row['tfidf_score'] = 0.0
                feature_row['intent_score'] = 0.0
                feature_row['embedding_score'] = 0.0
            
            # === NEW TEXTUAL FEATURES ===
            feature_row.update(self.feature_extractor.extract_textual_features(query, result, processed_query))
            
            # === METADATA FEATURES ===
            feature_row.update(self.feature_extractor.extract_metadata_features(result))
            
            # === LLM-DERIVED FEATURES ===
            feature_row.update(self.feature_extractor.extract_llm_features(processed_query, result))
            
            # === USER INTERACTION FEATURES ===
            feature_row.update(self.feature_extractor.extract_user_features(result, user_feedback_data))
            
            # Add enhanced textual features
            enhanced_features = self.enhanced_text_features.extract_all_features(query, result, processed_query)
            feature_row.update(enhanced_features)
            
            # Add current scores if available
            if current_scores:
                for score_type, scores in current_scores.items():
                    if len(scores) > i:
                        feature_row[f'current_{score_type}'] = scores[i]
            
            features_list.append(feature_row)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Fill NaN values
        features_df = features_df.fillna(0.0)
        
        logger.info(f"Extracted {len(features_df.columns)} features for {len(features_df)} results")
        
        return features_df
    
    def train_model(self, training_data: List[Dict[str, Any]], 
                   validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train XGBoost LTR model
        
        Args:
            training_data: List of training examples with features and relevance labels
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training statistics and metrics
        """
        if not XGBOOST_AVAILABLE:
            raise ValueError("XGBoost not available for training")
        
        if not training_data:
            raise ValueError("No training data provided")
        
        logger.info(f"Training LTR model with {len(training_data)} examples")
        
        # Convert training data to DataFrame
        df = pd.DataFrame(training_data)
        
        # Separate features and labels
        feature_columns = [col for col in df.columns if col not in ['relevance_label', 'query_id', 'result_id']]
        self.feature_names = feature_columns
        
        X = df[feature_columns].values
        y = df['relevance_label'].values
        groups = df.groupby('query_id').size().values  # Group sizes for ranking
        
        # Train-validation split
        if validation_split > 0:
            # Split by queries to maintain group structure
            unique_queries = df['query_id'].unique()
            train_queries, val_queries = train_test_split(unique_queries, test_size=validation_split, random_state=42)
            
            train_mask = df['query_id'].isin(train_queries)
            val_mask = df['query_id'].isin(val_queries)
            
            X_train, X_val = X[train_mask], X[val_mask]
            y_train, y_val = y[train_mask], y[val_mask]
            train_groups = df[train_mask].groupby('query_id').size().values
            val_groups = df[val_mask].groupby('query_id').size().values
        else:
            X_train, X_val = X, None
            y_train, y_val = y, None
            train_groups, val_groups = groups, None
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtrain.set_group(train_groups)
        
        if X_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            dval.set_group(val_groups)
            evallist = [(dtrain, 'train'), (dval, 'eval')]
        else:
            evallist = [(dtrain, 'train')]
        
        # XGBoost parameters for ranking
        params = {
            'objective': 'rank:pairwise',
            'eta': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'ndcg@10',
            'seed': 42,
            'silent': 1
        }
        
        # Train model
        num_rounds = 100
        self.model = xgb.train(
            params, dtrain, num_rounds,
            evals=evallist,
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        self.is_trained = True
        
        # Calculate training statistics
        train_pred = self.model.predict(dtrain)
        train_ndcg = self._calculate_ndcg(y_train, train_pred, train_groups)
        
        val_ndcg = None
        if X_val is not None:
            val_pred = self.model.predict(dval)
            val_ndcg = self._calculate_ndcg(y_val, val_pred, val_groups)
        
        # Feature importance
        feature_importance = self.model.get_fscore()
        
        self.training_stats = {
            'training_samples': len(X_train),
            'validation_samples': len(X_val) if X_val is not None else 0,
            'num_features': len(feature_columns),
            'num_queries': len(train_groups),
            'train_ndcg': train_ndcg,
            'val_ndcg': val_ndcg,
            'feature_importance': feature_importance,
            'training_date': datetime.now().isoformat(),
            'model_params': params
        }
        
        # Save model
        self._save_model()
        
        # Format validation NDCG properly
        val_ndcg_str = f"{val_ndcg:.4f}" if val_ndcg is not None else "N/A"
        logger.info(f"LTR model trained successfully. Train NDCG: {train_ndcg:.4f}, Val NDCG: {val_ndcg_str}")
        
        return self.training_stats
    
    def predict_scores(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Predict relevance scores using trained model
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            Array of predicted relevance scores
        """
        if not self.is_trained or self.model is None:
            logger.warning("LTR model not trained, returning zeros")
            return np.zeros(len(features_df))
        
        if features_df.empty:
            return np.array([])
        
        # Ensure feature order matches training
        feature_columns = [col for col in self.feature_names if col in features_df.columns]
        missing_features = set(self.feature_names) - set(feature_columns)
        
        if missing_features:
            logger.warning(f"Missing features for prediction: {missing_features}")
            # Add missing features with zero values
            for feature in missing_features:
                features_df[feature] = 0.0
        
        # Select and order features
        X = features_df[self.feature_names].values
        
        # Create DMatrix and predict
        dmatrix = xgb.DMatrix(X)
        scores = self.model.predict(dmatrix)
        
        return scores
    
    def rank_results(self, query: str, results: List[Dict[str, Any]], 
                    processed_query: Dict[str, Any] = None,
                    current_scores: Dict[str, List[float]] = None,
                    user_feedback_data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Rank search results using LTR model
        
        Args:
            query: Search query
            results: List of search results
            processed_query: LLM-processed query
            current_scores: Current ranking scores
            user_feedback_data: User feedback data
            
        Returns:
            Ranked list of results with LTR scores
        """
        if not results:
            return results
        
        if not self.is_trained:
            logger.warning("LTR model not trained, returning original order")
            # Add placeholder LTR scores
            for i, result in enumerate(results):
                result['ltr_score'] = 0.5  # Neutral score
                result['ltr_rank'] = i + 1
            return results
        
        # Extract features
        features_df = self.extract_features(
            query, results, processed_query, current_scores, user_feedback_data
        )
        
        # Predict scores
        ltr_scores = self.predict_scores(features_df)
        
        # Add scores to results and sort
        for i, (result, score) in enumerate(zip(results, ltr_scores)):
            result['ltr_score'] = float(score)
        
        # Sort by LTR score (descending)
        ranked_results = sorted(results, key=lambda x: x['ltr_score'], reverse=True)
        
        # Add LTR ranks
        for i, result in enumerate(ranked_results):
            result['ltr_rank'] = i + 1
        
        logger.info(f"LTR ranking completed for {len(results)} results")
        
        return ranked_results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if not self.is_trained or self.model is None:
            return {}
        
        importance = self.model.get_fscore()
        # Normalize importance scores
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}
        
        return importance
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get comprehensive model statistics"""
        stats = {
            'ltr_available': XGBOOST_AVAILABLE,
            'model_trained': self.is_trained,
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'model_dir': self.model_dir
        }
        
        # Add reason if LTR is not available
        if not XGBOOST_AVAILABLE:
            stats['reason'] = 'XGBoost dependencies not available'
        elif not self.is_trained:
            stats['reason'] = 'Model not yet trained (requires feedback data)'
        
        if self.is_trained:
            stats.update(self.training_stats)
            stats['feature_importance'] = self.get_feature_importance()
        
        return stats
    
    def _calculate_ndcg(self, y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> float:
        """Calculate NDCG@10 for ranking evaluation"""
        try:
            # Split predictions by groups
            start_idx = 0
            ndcg_scores = []
            
            for group_size in groups:
                end_idx = start_idx + group_size
                
                group_true = y_true[start_idx:end_idx]
                group_pred = y_pred[start_idx:end_idx]
                
                # Calculate NDCG for this group
                if len(group_true) > 1:
                    # Reshape for sklearn
                    group_true = group_true.reshape(1, -1)
                    group_pred = group_pred.reshape(1, -1)
                    
                    ndcg = ndcg_score(group_true, group_pred, k=min(10, len(group_true[0])))
                    ndcg_scores.append(ndcg)
                
                start_idx = end_idx
            
            return np.mean(ndcg_scores) if ndcg_scores else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating NDCG: {e}")
            return 0.0
    
    def _save_model(self):
        """Save trained model and metadata"""
        if not self.is_trained or self.model is None:
            return
        
        try:
            # Save XGBoost model
            model_path = os.path.join(self.model_dir, 'ltr_model.json')
            self.model.save_model(model_path)
            
            # Save metadata
            metadata = {
                'feature_names': self.feature_names,
                'training_stats': self.training_stats,
                'is_trained': self.is_trained
            }
            
            metadata_path = os.path.join(self.model_dir, 'model_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("LTR model saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving LTR model: {e}")
    
    def _load_model(self):
        """Load existing trained model"""
        model_path = os.path.join(self.model_dir, 'ltr_model.json')
        metadata_path = os.path.join(self.model_dir, 'model_metadata.json')
        
        try:
            if os.path.exists(model_path) and os.path.exists(metadata_path):
                # Load XGBoost model
                self.model = xgb.Booster()
                self.model.load_model(model_path)
                
                # Load metadata
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.feature_names = metadata.get('feature_names', [])
                self.training_stats = metadata.get('training_stats', {})
                self.is_trained = metadata.get('is_trained', False)
                
                logger.info(f"LTR model loaded successfully with {len(self.feature_names)} features")
            
        except Exception as e:
            logger.warning(f"Failed to load existing LTR model: {e}")
            self.model = None
            self.is_trained = False

    def build_knowledge_graph(self, search_results: List[Dict[str, Any]]) -> None:
        """Build the knowledge graph from search results"""
        if not self.knowledge_graph:
            return
        
        logger.info("Building knowledge graph from search results...")
        
        for result in search_results:
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
        logger.info("Knowledge graph built successfully")


class FeatureExtractor:
    """Advanced feature extraction for LTR"""
    
    def extract_textual_features(self, query: str, result: Dict[str, Any], 
                                processed_query: Dict[str, Any] = None) -> Dict[str, float]:
        """Extract advanced textual features"""
        features = {}
        
        # Document text fields
        title = result.get('title', '')
        description = result.get('description', '')
        author = result.get('author', '')
        
        # Combined text for analysis
        full_text = f"{title} {description} {author}".lower()
        query_lower = query.lower()
        
        # === BM25-style features ===
        features.update(self._calculate_bm25_features(query_lower, title.lower(), description.lower()))
        
        # === N-gram overlap features ===
        features.update(self._calculate_ngram_features(query_lower, full_text))
        
        # === Query term proximity ===
        features['query_term_proximity'] = self._calculate_term_proximity(query_lower, full_text)
        
        # === Text length features ===
        features['title_length'] = len(title.split())
        features['description_length'] = len(description.split())
        features['total_text_length'] = len(full_text.split())
        
        # === Query coverage ===
        query_terms = set(query_lower.split())
        text_terms = set(full_text.split())
        features['query_coverage'] = len(query_terms.intersection(text_terms)) / len(query_terms) if query_terms else 0
        
        # === Graph-based features ===
        if hasattr(self, 'knowledge_graph'):
            result_id = result.get('id')
            if result_id:
                # Get graph features
                graph_features = self.knowledge_graph.extract_graph_features(result_id)
                features.update(graph_features)
                
                # Get authority score
                features['authority_score'] = self.knowledge_graph.get_authority_score(result_id)
                
                # Get connection strength to query terms
                query_keywords = processed_query.get('keywords', {}).get('primary', []) if processed_query else []
                connection_strengths = []
                for keyword in query_keywords:
                    keyword_id = f"keyword_{keyword.lower()}"
                    strength = self.knowledge_graph.get_connection_strength(result_id, keyword_id)
                    connection_strengths.append(strength)
                features['query_connection_strength'] = max(connection_strengths) if connection_strengths else 0.0
        
        return features
    
    def extract_metadata_features(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Extract metadata-based features"""
        features = {}
        metadata = result.get('metadata', {})
        
        # === Recency features ===
        features.update(self._calculate_recency_features(result, metadata))
        
        # === Authority/Impact features ===
        features.update(self._calculate_authority_features(result, metadata))
        
        # === Availability features ===
        features.update(self._calculate_availability_features(result, metadata))
        
        # === Type-specific features ===
        features.update(self._calculate_type_features(result))
        
        return features
    
    def extract_llm_features(self, processed_query: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, float]:
        """Extract LLM-derived features"""
        features = {}
        
        if not processed_query:
            # Default values when no LLM processing
            features['intent_confidence'] = 0.5
            features['field_match_score'] = 0.0
            features['entity_match_score'] = 0.0
            return features
        
        # === Intent confidence ===
        intent = processed_query.get('intent', {})
        features['intent_confidence'] = intent.get('confidence', 0.5)
        
        # === Field matching ===
        academic_fields = processed_query.get('academic_fields', {})
        result_category = result.get('metadata', {}).get('category', '')
        features['field_match_score'] = self._calculate_field_match(academic_fields, result_category)
        
        # === Entity matching ===
        entities = processed_query.get('entities', {})
        features['entity_match_score'] = self._calculate_entity_match(entities, result)
        
        return features
    
    def extract_user_features(self, result: Dict[str, Any], 
                            feedback_data: Dict[str, Any] = None) -> Dict[str, float]:
        """Extract user interaction features"""
        features = {}
        
        if not feedback_data:
            # Default values when no feedback data
            features['avg_rating'] = 2.5  # Neutral rating
            features['feedback_count'] = 0
            features['positive_feedback_ratio'] = 0.5
            return features
        
        result_id = result.get('id', '')
        result_feedback = feedback_data.get(result_id, {})
        
        # === Average rating ===
        ratings = result_feedback.get('ratings', [])
        features['avg_rating'] = np.mean(ratings) if ratings else 2.5
        
        # === Feedback count ===
        features['feedback_count'] = len(ratings)
        
        # === Positive feedback ratio ===
        positive_ratings = [r for r in ratings if r >= 4]  # 4-5 stars considered positive
        features['positive_feedback_ratio'] = len(positive_ratings) / len(ratings) if ratings else 0.5
        
        return features
    
    def _calculate_bm25_features(self, query: str, title: str, description: str) -> Dict[str, float]:
        """Calculate BM25-style features"""
        features = {}
        
        # BM25 parameters
        k1, b = 1.2, 0.75
        
        query_terms = query.split()
        
        # Calculate for title
        features['bm25_title'] = self._bm25_score(query_terms, title.split(), k1, b)
        
        # Calculate for description
        features['bm25_description'] = self._bm25_score(query_terms, description.split(), k1, b)
        
        return features
    
    def _bm25_score(self, query_terms: List[str], doc_terms: List[str], k1: float, b: float) -> float:
        """Calculate BM25 score for document"""
        if not query_terms or not doc_terms:
            return 0.0
        
        doc_len = len(doc_terms)
        avg_doc_len = 100  # Assumed average document length
        
        doc_freq = Counter(doc_terms)
        score = 0.0
        
        for term in query_terms:
            tf = doc_freq.get(term, 0)
            if tf > 0:
                idf = math.log(1000 / (1 + 1))  # Simplified IDF
                score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
        
        return score
    
    def _calculate_ngram_features(self, query: str, text: str) -> Dict[str, float]:
        """Calculate n-gram overlap features"""
        features = {}
        
        query_words = query.split()
        text_words = text.split()
        
        # Unigram overlap
        query_unigrams = set(query_words)
        text_unigrams = set(text_words)
        features['unigram_overlap'] = len(query_unigrams.intersection(text_unigrams)) / len(query_unigrams) if query_unigrams else 0
        
        # Bigram overlap
        query_bigrams = set(zip(query_words[:-1], query_words[1:]))
        text_bigrams = set(zip(text_words[:-1], text_words[1:]))
        features['bigram_overlap'] = len(query_bigrams.intersection(text_bigrams)) / len(query_bigrams) if query_bigrams else 0
        
        return features
    
    def _calculate_term_proximity(self, query: str, text: str) -> float:
        """Calculate average proximity of query terms in text"""
        query_terms = query.split()
        text_words = text.split()
        
        if len(query_terms) < 2:
            return 1.0
        
        term_positions = {}
        for i, word in enumerate(text_words):
            if word in query_terms:
                if word not in term_positions:
                    term_positions[word] = []
                term_positions[word].append(i)
        
        if len(term_positions) < 2:
            return 0.0
        
        # Calculate minimum distance between any two query terms
        min_distance = float('inf')
        positions_list = list(term_positions.values())
        
        for i in range(len(positions_list)):
            for j in range(i + 1, len(positions_list)):
                for pos1 in positions_list[i]:
                    for pos2 in positions_list[j]:
                        distance = abs(pos1 - pos2)
                        min_distance = min(min_distance, distance)
        
        # Convert to proximity score (closer = higher score)
        return 1.0 / (1.0 + min_distance) if min_distance != float('inf') else 0.0
    
    def _calculate_recency_features(self, result: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, float]:
        """Calculate recency-based features"""
        features = {}
        
        # Get publication date
        pub_date_str = metadata.get('publication_date', '') or str(metadata.get('year', ''))
        
        try:
            if pub_date_str and pub_date_str != 'nan':
                # Try to parse date
                if len(pub_date_str) == 4:  # Year only
                    pub_date = datetime(int(pub_date_str), 1, 1)
                else:
                    pub_date = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                
                # Calculate days since publication
                days_since = (datetime.now() - pub_date).days
                
                # Recency score (more recent = higher score)
                features['days_since_publication'] = days_since
                features['recency_score'] = 1.0 / (1.0 + days_since / 365.0)  # Decay over years
            else:
                features['days_since_publication'] = 3650  # 10 years default
                features['recency_score'] = 0.1
                
        except (ValueError, TypeError):
            features['days_since_publication'] = 3650
            features['recency_score'] = 0.1
        
        return features
    
    def _calculate_authority_features(self, result: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, float]:
        """Calculate authority/impact features"""
        features = {}
        
        # Citations (for research papers)
        citations = metadata.get('citations', 0)
        features['citation_count'] = citations
        features['citation_score'] = math.log(1 + citations)  # Log scale for citations
        
        # Author/Expert authority (simplified)
        author = result.get('author', '')
        features['author_name_length'] = len(author.split())  # More words might indicate academic titles
        
        # Institution prestige (simplified by checking for common prestigious terms)
        institution = metadata.get('institution', '') or metadata.get('university', '')
        prestige_keywords = ['university', 'institute', 'research', 'center', 'college']
        features['institution_prestige'] = sum(1 for keyword in prestige_keywords if keyword.lower() in institution.lower())
        
        return features
    
    def _calculate_availability_features(self, result: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, float]:
        """Calculate availability/status features"""
        features = {}
        
        status = metadata.get('status', '').lower()
        
        # Availability score
        if status in ['available', 'active', 'published']:
            features['availability_score'] = 1.0
        elif status in ['pending', 'under_review']:
            features['availability_score'] = 0.5
        elif status in ['unavailable', 'out_of_service', 'rejected']:
            features['availability_score'] = 0.0
        else:
            features['availability_score'] = 0.7  # Default for unknown status
        
        return features
    
    def _calculate_type_features(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate resource type features"""
        features = {}
        
        result_type = result.get('type', '').lower()
        
        # One-hot encoding for result types
        type_mapping = {
            'article': 1.0, 'book': 0.9, 'journal': 0.8, 'thesis': 0.7,
            'expert': 0.6, 'equipment': 0.5, 'material': 0.4, 'project': 0.8
        }
        
        features['type_importance'] = type_mapping.get(result_type, 0.5)
        
        # Type-specific binary features
        for t in ['article', 'book', 'expert', 'equipment']:
            features[f'is_{t}'] = 1.0 if result_type == t else 0.0
        
        return features
    
    def _calculate_field_match(self, academic_fields: Dict[str, Any], result_category: str) -> float:
        """Calculate how well result matches academic fields from LLM"""
        if not academic_fields or not result_category:
            return 0.0
        
        primary_field = academic_fields.get('primary_field', '').lower()
        related_fields = [f.lower() for f in academic_fields.get('related_fields', [])]
        
        result_category = result_category.lower()
        
        # Check for exact match
        if primary_field in result_category or result_category in primary_field:
            return 1.0
        
        # Check for partial matches in related fields
        for field in related_fields:
            if field in result_category or result_category in field:
                return 0.7
        
        return 0.0
    
    def _calculate_entity_match(self, entities: Dict[str, Any], result: Dict[str, Any]) -> float:
        """Calculate how well result matches entities from LLM"""
        if not entities:
            return 0.0
        
        # Get all text from result
        result_text = f"{result.get('title', '')} {result.get('description', '')}".lower()
        
        match_score = 0.0
        total_entities = 0
        
        # Check different entity types
        for entity_type in ['technologies', 'concepts', 'organizations']:
            entity_list = entities.get(entity_type, [])
            total_entities += len(entity_list)
            
            for entity in entity_list:
                if entity.lower() in result_text:
                    match_score += 1.0
        
        return match_score / total_entities if total_entities > 0 else 0.0 