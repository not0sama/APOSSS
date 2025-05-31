# Feedback-Based Learning Strategy for APOSSS Ranking Improvement

## 📊 Current Feedback Data Structure

### Stored Feedback Format
```json
{
  "query_id": "uuid-generated-query-id",
  "result_id": "result-document-id", 
  "rating": 5,  // 1-5 scale (1=👎 not helpful, 5=👍 helpful)
  "feedback_type": "thumbs_up",  // "thumbs_up" or "thumbs_down"
  "user_session": "web_session_timestamp",
  "submitted_at": "2024-01-15T14:30:00.000Z",
  "feedback_version": "1.0"
}
```

### Associated Data Available
- **Query Analysis**: LLM-processed query with keywords, intent, entities
- **Result Metadata**: Type, content, source database, original ranking scores
- **Ranking Components**: Heuristic, TF-IDF, and intent scores for each result

## 🎯 Learning Algorithms to Implement

### 1. **Learning-to-Rank (LTR) Model**

#### Approach
- **Training Data**: Query-result pairs with feedback as labels
- **Features**: Current ranking components (heuristic, TF-IDF, intent) + metadata
- **Algorithm**: RankNet, LambdaMART, or XGBoost ranking

#### Implementation Steps
```python
# Phase 4 Implementation Example
class FeedbackLearningEngine:
    def prepare_training_data(self):
        """Convert feedback to training features"""
        training_data = []
        
        # Get all feedback with associated query and result data
        for feedback in self.feedback_system.get_all_feedback():
            query_data = self.get_query_analysis(feedback['query_id'])
            result_data = self.get_result_metadata(feedback['result_id'])
            
            features = {
                'heuristic_score': result_data['score_breakdown']['heuristic_score'],
                'tfidf_score': result_data['score_breakdown']['tfidf_score'],
                'intent_score': result_data['score_breakdown']['intent_score'],
                'result_type': self.encode_result_type(result_data['type']),
                'query_intent': self.encode_intent(query_data['intent']['primary_intent']),
                'keyword_match_count': self.count_keyword_matches(query_data, result_data),
                'title_match': self.has_title_match(query_data, result_data),
                'author_relevance': self.calculate_author_relevance(query_data, result_data)
            }
            
            training_data.append({
                'features': features,
                'label': 1 if feedback['rating'] >= 4 else 0,  # Binary relevance
                'query_id': feedback['query_id']
            })
        
        return training_data
```

### 2. **Pairwise Learning**

#### Approach
- **Concept**: Learn relative preferences between result pairs
- **Training**: For each query, create pairs where feedback differs
- **Benefit**: More robust than pointwise scoring

#### Implementation
```python
def create_pairwise_training_data(self):
    """Create pairwise preference data from feedback"""
    pairwise_data = []
    
    # Group feedback by query
    feedback_by_query = self.group_feedback_by_query()
    
    for query_id, query_feedback in feedback_by_query.items():
        # Find positive and negative examples
        positive_results = [f for f in query_feedback if f['rating'] >= 4]
        negative_results = [f for f in query_feedback if f['rating'] <= 2]
        
        # Create preference pairs (positive should rank higher than negative)
        for pos_result in positive_results:
            for neg_result in negative_results:
                pairwise_data.append({
                    'query_id': query_id,
                    'preferred_result': pos_result['result_id'],
                    'non_preferred_result': neg_result['result_id'],
                    'preference_strength': pos_result['rating'] - neg_result['rating']
                })
    
    return pairwise_data
```

### 3. **Weight Adjustment Strategy**

#### Simple Approach for Quick Improvement
```python
class AdaptiveWeightRanking:
    def __init__(self):
        # Current weights: heuristic=0.4, tfidf=0.4, intent=0.2
        self.component_weights = {
            'heuristic': 0.4,
            'tfidf': 0.4, 
            'intent': 0.2
        }
    
    def adjust_weights_from_feedback(self):
        """Adjust component weights based on feedback correlation"""
        
        # Calculate correlation between each component and user satisfaction
        feedback_data = self.get_feedback_with_scores()
        
        correlations = {}
        for component in ['heuristic', 'tfidf', 'intent']:
            component_scores = [f['scores'][component] for f in feedback_data]
            user_ratings = [f['rating'] for f in feedback_data]
            correlations[component] = self.calculate_correlation(component_scores, user_ratings)
        
        # Adjust weights based on correlation strength
        total_correlation = sum(abs(c) for c in correlations.values())
        
        for component, correlation in correlations.items():
            if total_correlation > 0:
                self.component_weights[component] = abs(correlation) / total_correlation
        
        return self.component_weights
```

## 🔄 Continuous Learning Implementation

### 1. **Batch Learning Updates**
```python
class ContinuousLearning:
    def __init__(self, update_frequency='weekly'):
        self.update_frequency = update_frequency
        self.min_feedback_threshold = 50  # Minimum feedback needed for retraining
    
    def should_update_model(self):
        """Check if enough new feedback collected for retraining"""
        new_feedback_count = self.feedback_system.get_feedback_count_since_last_update()
        return new_feedback_count >= self.min_feedback_threshold
    
    def update_ranking_model(self):
        """Retrain ranking model with new feedback data"""
        if self.should_update_model():
            # Prepare training data
            training_data = self.prepare_training_data()
            
            # Train new model
            new_model = self.train_ranking_model(training_data)
            
            # A/B test new model vs current model
            if self.validate_model_improvement(new_model):
                self.deploy_new_model(new_model)
                logger.info("Ranking model updated successfully")
            else:
                logger.info("New model did not show improvement, keeping current model")
```

### 2. **Online Learning Approach**
```python
class OnlineLearning:
    def update_model_incrementally(self, new_feedback):
        """Update model weights with each new feedback"""
        
        # Get prediction for this query-result pair
        current_prediction = self.predict_relevance(new_feedback)
        actual_relevance = 1 if new_feedback['rating'] >= 4 else 0
        
        # Calculate prediction error
        error = actual_relevance - current_prediction
        
        # Update model weights using gradient descent
        learning_rate = 0.01
        self.update_weights_with_gradient(error, learning_rate, new_feedback)
```

## 📈 Performance Metrics & Validation

### 1. **Ranking Quality Metrics**
```python
def calculate_ranking_metrics(self, test_feedback):
    """Calculate standard IR metrics using feedback as ground truth"""
    
    metrics = {}
    
    # Group by query for metric calculation
    feedback_by_query = self.group_feedback_by_query(test_feedback)
    
    ndcg_scores = []
    map_scores = []
    
    for query_id, query_feedback in feedback_by_query.items():
        # Get current ranking for this query
        current_ranking = self.get_current_ranking(query_id)
        
        # Calculate NDCG (Normalized Discounted Cumulative Gain)
        ndcg = self.calculate_ndcg(current_ranking, query_feedback)
        ndcg_scores.append(ndcg)
        
        # Calculate MAP (Mean Average Precision)
        map_score = self.calculate_map(current_ranking, query_feedback)
        map_scores.append(map_score)
    
    metrics['ndcg'] = np.mean(ndcg_scores)
    metrics['map'] = np.mean(map_scores)
    metrics['user_satisfaction'] = self.calculate_user_satisfaction_score(test_feedback)
    
    return metrics
```

### 2. **A/B Testing Framework**
```python
class ABTestingFramework:
    def run_ab_test(self, new_model, test_duration_days=14):
        """Run A/B test comparing current vs new ranking model"""
        
        # Split users randomly
        test_group_ratio = 0.2  # 20% get new model
        
        # Track metrics for both groups
        control_metrics = []
        test_metrics = []
        
        # After test period, compare results
        if self.test_group_performs_better(control_metrics, test_metrics):
            return True  # Deploy new model
        else:
            return False  # Keep current model
```

## 🎯 Specific Improvement Strategies

### 1. **Intent-Based Learning**
- **Observation**: Track which resource types users prefer for different query intents
- **Adaptation**: Adjust intent weights based on actual user preferences
- **Example**: If users consistently rate equipment higher for "find_equipment" queries, increase equipment bias

### 2. **Query-Specific Learning**
- **Observation**: Some queries may benefit from different scoring approaches
- **Adaptation**: Learn query-specific weight adjustments
- **Implementation**: Cluster similar queries and learn optimal weights per cluster

### 3. **Temporal Learning**
- **Observation**: User preferences may change over time
- **Adaptation**: Give more weight to recent feedback
- **Implementation**: Time-decay weights in learning algorithm

### 4. **User Personalization**
- **Observation**: Different users may have different preferences
- **Adaptation**: Learn user-specific ranking preferences
- **Implementation**: Track user sessions and adapt rankings

## 🛠️ Implementation Roadmap

### Phase 4: Basic Learning
1. **Weight Adjustment**: Implement simple correlation-based weight adjustment
2. **Feedback Analysis**: Build analytics dashboard for feedback insights
3. **A/B Testing**: Implement framework for testing ranking changes

### Phase 5: Advanced Learning
1. **LTR Model**: Implement full learning-to-rank system
2. **Continuous Learning**: Automated model updates based on feedback
3. **Personalization**: User-specific ranking adaptations
4. **Query Clustering**: Learn query-type specific ranking strategies

## 📊 Data Requirements

### Minimum Viable Learning
- **Feedback Count**: 100+ feedback entries for initial learning
- **Query Diversity**: Feedback across different query types and intents
- **Rating Distribution**: Both positive and negative feedback for meaningful learning

### Optimal Learning Dataset
- **Feedback Count**: 1000+ feedback entries
- **Temporal Distribution**: Feedback collected over time to detect trends
- **User Diversity**: Feedback from multiple users/sessions

## 🔧 Technical Implementation

### Database Schema Extensions
```python
# Additional collections for learning
{
    "learning_models": {
        "model_id": "string",
        "model_type": "string",  # "weight_adjustment", "ltr", etc.
        "weights": {"heuristic": 0.4, "tfidf": 0.4, "intent": 0.2},
        "performance_metrics": {"ndcg": 0.85, "map": 0.78},
        "created_at": "datetime",
        "is_active": "boolean"
    },
    
    "model_performance": {
        "model_id": "string",
        "test_date": "datetime", 
        "metrics": {"ndcg": 0.85, "user_satisfaction": 4.2},
        "feedback_count": 150
    }
}
```

### Learning Pipeline
```python
class LearningPipeline:
    def run_learning_cycle(self):
        """Complete learning cycle"""
        
        # 1. Data Preparation
        training_data = self.prepare_training_data()
        
        # 2. Model Training
        candidate_models = self.train_candidate_models(training_data)
        
        # 3. Model Validation
        best_model = self.select_best_model(candidate_models)
        
        # 4. A/B Testing
        if self.ab_test_passes(best_model):
            # 5. Model Deployment
            self.deploy_model(best_model)
            
        # 6. Performance Monitoring
        self.monitor_model_performance()
```

This feedback learning strategy provides a comprehensive roadmap for using the collected user feedback to continuously improve the APOSSS ranking algorithm, making it more accurate and user-centric over time. 