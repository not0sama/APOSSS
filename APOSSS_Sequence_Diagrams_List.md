# APOSSS System - Comprehensive Sequence Diagrams List
## AI-Powered Open-Science Semantic Search System

### Overview
This document provides a comprehensive list of all possible sequence diagrams that can be created for the APOSSS system. These diagrams show the temporal flow of interactions between different system components, external services, and users.

---

## 1. User Authentication & Session Management (12 Diagrams)

### 1.1 User Registration Flow
**Participants**: User, Frontend, Flask API, UserManager, DatabaseManager, MongoDB, EmailService
**Key Interactions**:
- Email/username availability checking
- Password validation and hashing
- User profile creation
- Email verification code generation
- Welcome email sending

### 1.2 User Login Flow
**Participants**: User, Frontend, Flask API, UserManager, DatabaseManager, MongoDB, JWT Service
**Key Interactions**:
- Credential validation
- Password verification (bcrypt)
- JWT token generation
- Session creation
- Profile data retrieval

### 1.3 Google OAuth Authentication
**Participants**: User, Frontend, Flask API, OAuthManager, Google OAuth Service, UserManager, MongoDB
**Key Interactions**:
- Authorization URL generation
- OAuth redirect handling
- Authorization code exchange
- Google user info retrieval
- Account creation/linking

### 1.4 ORCID OAuth Authentication
**Participants**: User, Frontend, Flask API, OAuthManager, ORCID API, UserManager, MongoDB
**Key Interactions**:
- ORCID authorization flow
- Academic profile retrieval
- Research credentials validation
- Account integration

### 1.5 Email Verification Process
**Participants**: User, Frontend, Flask API, UserManager, EmailService, DatabaseManager, MongoDB
**Key Interactions**:
- Verification code generation
- Email sending with SMTP
- Code validation
- Account activation
- Status updates

### 1.6 Forgot Password Flow
**Participants**: User, Frontend, Flask API, UserManager, EmailService, DatabaseManager, MongoDB
**Key Interactions**:
- Password reset request
- Verification code generation
- Secure email sending
- Code validation
- Password update with bcrypt

### 1.7 Password Change Flow
**Participants**: User, Frontend, Flask API, UserManager, DatabaseManager, MongoDB
**Key Interactions**:
- Current password verification
- New password validation
- Password hashing
- Database update
- Session invalidation

### 1.8 JWT Token Verification
**Participants**: Frontend, Flask API, UserManager, JWT Service
**Key Interactions**:
- Token extraction from headers
- Token signature verification
- Expiry validation
- User session validation

### 1.9 Profile Picture Upload
**Participants**: User, Frontend, Flask API, UserManager, File System, DatabaseManager, MongoDB
**Key Interactions**:
- Image file validation
- File size checking
- Image processing/resizing
- File system storage
- Database metadata update

### 1.10 User Profile Management
**Participants**: User, Frontend, Flask API, UserManager, DatabaseManager, MongoDB
**Key Interactions**:
- Profile data retrieval
- Field validation
- Academic information updates
- Preferences storage

### 1.11 User Preferences Management
**Participants**: User, Frontend, Flask API, UserManager, DatabaseManager, MongoDB
**Key Interactions**:
- Preference categories handling
- Search settings configuration
- Notification preferences
- Privacy settings updates

### 1.12 Session Logout Process
**Participants**: User, Frontend, Flask API, UserManager, JWT Service
**Key Interactions**:
- Session invalidation
- Token blacklisting
- Client-side cleanup
- Redirect handling

---

## 2. Core Search & Query Processing (15 Diagrams)

### 2.1 Complete Search Process (Main Flow)
**Participants**: User, Frontend, Flask API, QueryProcessor, LLMProcessor, Gemini API, SearchEngine, RankingEngine, DatabaseManager, 6 MongoDB Databases, FAISS Index
**Key Interactions**:
- Query submission and validation
- LLM processing with Gemini
- Multi-database search execution
- Result aggregation and ranking
- Response formatting and delivery

### 2.2 Query Processing with LLM
**Participants**: QueryProcessor, LLMProcessor, Gemini API
**Key Interactions**:
- Query cleaning and validation
- Language detection
- Translation (if needed)
- Intent analysis
- Entity extraction
- Keyword analysis
- Semantic expansion

### 2.3 Multi-Database Search Execution
**Participants**: SearchEngine, DatabaseManager, Academic Library DB, Experts System DB, Research Papers DB, Laboratories DB, Funding DB, APOSSS DB
**Key Interactions**:
- Database connection management
- Parallel query execution
- Collection-specific searches
- Result standardization
- Error handling and fallbacks

### 2.4 Hybrid Search (Semantic + Traditional)
**Participants**: SearchEngine, EmbeddingRanker, FAISS Index, DatabaseManager, MongoDB Collections
**Key Interactions**:
- Semantic search via FAISS
- Traditional keyword search
- Result merging and deduplication
- Score normalization

### 2.5 FAISS Index Search Process
**Participants**: SearchEngine, EmbeddingRanker, SentenceTransformers, FAISS Index, Embedding Cache
**Key Interactions**:
- Query embedding generation
- FAISS similarity search
- Result filtering and ranking
- Cache management

### 2.6 Real-time Embedding Calculation
**Participants**: EmbeddingRanker, SentenceTransformers, Embedding Cache, File System
**Key Interactions**:
- Text preprocessing
- Model inference
- Vector calculation
- Cache storage and retrieval

### 2.7 Multi-Algorithm Ranking Process
**Participants**: RankingEngine, LTRRanker, EmbeddingRanker, LLMRanker, KnowledgeGraph
**Key Interactions**:
- Heuristic scoring
- TF-IDF calculation
- BM25 scoring
- Intent alignment scoring
- Embedding similarity
- LTR model prediction
- Score combination and normalization

### 2.8 Personalized Search Results
**Participants**: RankingEngine, UserManager, DatabaseManager, User Interaction History, User Preferences
**Key Interactions**:
- User behavior analysis
- Preference extraction
- Personalization scoring
- Historical pattern recognition
- Adaptive ranking adjustment

### 2.9 Learning-to-Rank (LTR) Inference
**Participants**: LTRRanker, XGBoost Model, FeatureExtractor, Model Cache
**Key Interactions**:
- Feature extraction (50+ features)
- Model inference
- Score prediction
- Ranking adjustment

### 2.10 Knowledge Graph Ranking
**Participants**: KnowledgeGraph, NetworkX, PageRank Algorithm, DatabaseManager
**Key Interactions**:
- Graph construction from results
- Authority calculation
- Connection strength analysis
- PageRank computation

### 2.11 LLM-based Result Re-ranking
**Participants**: LLMRanker, LLMProcessor, Gemini API
**Key Interactions**:
- Result analysis prompt creation
- LLM evaluation
- Relevance scoring
- Quality assessment

### 2.12 Search Result Categorization
**Participants**: RankingEngine, SearchResults
**Key Interactions**:
- Relevance threshold calculation
- High/Medium/Low categorization
- Result distribution analysis

### 2.13 Database Health Monitoring
**Participants**: DatabaseManager, MongoDB Instances, Health Check Service
**Key Interactions**:
- Connection testing
- Performance monitoring
- Error detection and logging
- Failover mechanisms

### 2.14 Search Error Handling & Fallbacks
**Participants**: All Search Components, Error Handler, Logger
**Key Interactions**:
- Component failure detection
- Graceful degradation
- Fallback strategy execution
- Error reporting

### 2.15 Search Session Tracking
**Participants**: Flask API, UserManager, DatabaseManager, Interaction Tracker
**Key Interactions**:
- Session ID generation
- Query logging
- Interaction recording
- Analytics data collection

---

## 3. Feedback & Learning System (8 Diagrams)

### 3.1 User Feedback Submission
**Participants**: User, Frontend, Flask API, FeedbackSystem, DatabaseManager, MongoDB, JSON File Storage
**Key Interactions**:
- Feedback validation
- Rating processing
- Comment storage
- Dual storage (MongoDB + file backup)
- Confirmation response

### 3.2 Feedback Analytics & Statistics
**Participants**: FeedbackSystem, DatabaseManager, MongoDB, Analytics Engine
**Key Interactions**:
- Feedback aggregation
- Statistical calculations
- Trend analysis
- Report generation

### 3.3 LTR Model Training Process
**Participants**: LTRRanker, FeedbackSystem, FeatureExtractor, XGBoost, Model Storage
**Key Interactions**:
- Training data preparation
- Feature extraction
- Model training
- Validation and evaluation
- Model persistence

### 3.4 Feedback-based Model Improvement
**Participants**: LTRRanker, FeedbackSystem, Training Scheduler, Model Evaluator
**Key Interactions**:
- Feedback collection
- Training data generation
- Model retraining
- Performance evaluation
- Model deployment

### 3.5 Real-time Feedback Processing
**Participants**: FeedbackSystem, DatabaseManager, Real-time Processor, Analytics Engine
**Key Interactions**:
- Immediate feedback storage
- Real-time statistics update
- Trend detection
- Alert generation

### 3.6 Feedback Quality Assessment
**Participants**: FeedbackSystem, Quality Analyzer, DatabaseManager
**Key Interactions**:
- Feedback validation
- Quality scoring
- Spam detection
- Reliability assessment

### 3.7 User Feedback History Retrieval
**Participants**: User, Frontend, Flask API, FeedbackSystem, DatabaseManager
**Key Interactions**:
- User authentication
- Feedback query execution
- History compilation
- Privacy filtering

### 3.8 Batch Feedback Processing
**Participants**: FeedbackSystem, Batch Processor, DatabaseManager, LTRRanker
**Key Interactions**:
- Scheduled batch execution
- Bulk feedback processing
- Model update triggering
- Performance monitoring

---

## 4. User Interaction & Personalization (10 Diagrams)

### 4.1 User Search History Management
**Participants**: User, Frontend, Flask API, UserManager, DatabaseManager, MongoDB
**Key Interactions**:
- Search query logging
- History retrieval
- Privacy filtering
- History clearing

### 4.2 Bookmark Management System
**Participants**: User, Frontend, Flask API, UserManager, DatabaseManager, MongoDB
**Key Interactions**:
- Bookmark creation/deletion
- Bookmark categorization
- Synchronization across devices
- Export functionality

### 4.3 User Preference Learning
**Participants**: UserManager, InteractionTracker, PreferenceAnalyzer, DatabaseManager
**Key Interactions**:
- Interaction pattern analysis
- Preference extraction
- Behavior modeling
- Recommendation generation

### 4.4 Personalization Engine Process
**Participants**: PersonalizationEngine, UserManager, InteractionHistory, PreferenceModel
**Key Interactions**:
- User profile analysis
- Preference scoring
- Content filtering
- Personalized ranking

### 4.5 User Statistics Dashboard
**Participants**: User, Frontend, Flask API, UserManager, AnalyticsEngine, DatabaseManager
**Key Interactions**:
- Usage statistics calculation
- Activity timeline generation
- Performance metrics compilation
- Visual dashboard rendering

### 4.6 Interaction Tracking System
**Participants**: Frontend, Flask API, InteractionTracker, DatabaseManager, MongoDB
**Key Interactions**:
- User action capture
- Event logging
- Session tracking
- Behavior analysis

### 4.7 Search Recommendation Engine
**Participants**: RecommendationEngine, UserManager, SearchHistory, TrendAnalyzer
**Key Interactions**:
- User behavior analysis
- Similar user identification
- Recommendation generation
- Suggestion ranking

### 4.8 User Onboarding Flow
**Participants**: User, Frontend, Flask API, UserManager, OnboardingService
**Key Interactions**:
- Welcome sequence
- Preference setup
- Feature introduction
- Initial recommendations

### 4.9 Privacy Settings Management
**Participants**: User, Frontend, Flask API, UserManager, PrivacyController, DatabaseManager
**Key Interactions**:
- Privacy preference setup
- Data visibility control
- Consent management
- Data deletion requests

### 4.10 Cross-session Continuity
**Participants**: Frontend, Flask API, UserManager, SessionManager, DatabaseManager
**Key Interactions**:
- Session restoration
- State persistence
- Cross-device synchronization
- Continuous experience

---

## 5. External Service Integration (7 Diagrams)

### 5.1 Google Gemini LLM Integration
**Participants**: LLMProcessor, Google Gemini API, API Rate Limiter, Error Handler
**Key Interactions**:
- API authentication
- Request formatting
- Response processing
- Error handling and retries
- Rate limit management

### 5.2 Email Service Integration (SMTP)
**Participants**: EmailService, SMTP Server, UserManager, Template Engine
**Key Interactions**:
- Email template rendering
- SMTP connection management
- Message sending
- Delivery confirmation
- Error handling

### 5.3 OAuth Provider Integration (Google)
**Participants**: OAuthManager, Google OAuth API, State Manager, Security Handler
**Key Interactions**:
- Authorization flow initiation
- State token management
- Code exchange
- User info retrieval
- Security validation

### 5.4 OAuth Provider Integration (ORCID)
**Participants**: OAuthManager, ORCID API, Academic Profile Handler
**Key Interactions**:
- Academic authentication
- Research profile retrieval
- Credential validation
- Institution verification

### 5.5 External Database Connectivity
**Participants**: DatabaseManager, MongoDB Instances, Connection Pool, Health Monitor
**Key Interactions**:
- Connection establishment
- Pool management
- Health monitoring
- Failover handling
- Performance optimization

### 5.6 File System Integration
**Participants**: FileManager, Operating System, Storage Handler, Backup Service
**Key Interactions**:
- File operations
- Storage management
- Backup creation
- Cleanup processes

### 5.7 Third-party API Error Handling
**Participants**: API Clients, Error Handler, Fallback Service, Logger, Monitoring
**Key Interactions**:
- Error detection
- Retry mechanisms
- Fallback activation
- Error reporting
- Performance monitoring

---

## 6. Data Processing & ML Pipelines (12 Diagrams)

### 6.1 FAISS Index Building Process
**Participants**: IndexBuilder, EmbeddingRanker, SentenceTransformers, DatabaseManager, FAISS Library, File System
**Key Interactions**:
- Document retrieval
- Batch processing
- Embedding generation
- Index construction
- Index optimization and storage

### 6.2 Embedding Cache Management
**Participants**: EmbeddingRanker, CacheManager, File System, Memory Manager
**Key Interactions**:
- Cache hit/miss handling
- Memory optimization
- Persistent storage
- Cache invalidation
- Performance monitoring

### 6.3 Document Preprocessing Pipeline
**Participants**: DocumentProcessor, TextCleaner, Tokenizer, LanguageDetector
**Key Interactions**:
- Text normalization
- Language detection
- Content extraction
- Metadata processing
- Quality validation

### 6.4 Feature Extraction for LTR
**Participants**: FeatureExtractor, LTRRanker, TextAnalyzer, MetadataProcessor
**Key Interactions**:
- Text feature calculation
- Metadata feature extraction
- User behavior features
- Graph-based features
- Feature normalization

### 6.5 Model Training Pipeline (XGBoost)
**Participants**: LTRRanker, XGBoost, TrainingDataProcessor, ModelValidator, ModelStorage
**Key Interactions**:
- Data preparation
- Feature engineering
- Model training
- Hyperparameter tuning
- Model validation and storage

### 6.6 Knowledge Graph Construction
**Participants**: KnowledgeGraph, NetworkX, DatabaseManager, GraphAnalyzer
**Key Interactions**:
- Entity extraction
- Relationship identification
- Graph building
- Authority calculation
- Graph optimization

### 6.7 Semantic Similarity Calculation
**Participants**: EmbeddingRanker, SentenceTransformers, SimilarityCalculator, VectorDatabase
**Key Interactions**:
- Vector embedding
- Similarity computation
- Result ranking
- Cache management

### 6.8 Real-time Model Inference
**Participants**: MLInferenceEngine, TrainedModels, FeatureProcessor, PredictionCache
**Key Interactions**:
- Feature preparation
- Model loading
- Inference execution
- Result processing
- Cache updates

### 6.9 Data Quality Assessment
**Participants**: QualityAssessment, DataValidator, StatisticsCalculator, ReportGenerator
**Key Interactions**:
- Data validation
- Quality metrics calculation
- Anomaly detection
- Quality reporting

### 6.10 Batch Processing Workflows
**Participants**: BatchProcessor, TaskScheduler, ResourceManager, ProgressTracker
**Key Interactions**:
- Job scheduling
- Resource allocation
- Progress monitoring
- Error handling
- Completion notification

### 6.11 Model Performance Monitoring
**Participants**: ModelMonitor, PerformanceTracker, MetricsCalculator, AlertSystem
**Key Interactions**:
- Performance measurement
- Metric tracking
- Threshold monitoring
- Alert generation
- Report creation

### 6.12 Continuous Learning Loop
**Participants**: LearningOrchestrator, FeedbackSystem, ModelTrainer, PerformanceEvaluator
**Key Interactions**:
- Feedback collection
- Performance evaluation
- Retraining triggers
- Model updates
- Performance validation

---

## 7. System Administration & Monitoring (8 Diagrams)

### 7.1 System Health Monitoring
**Participants**: HealthMonitor, ComponentChecker, DatabaseMonitor, APIMonitor, AlertSystem
**Key Interactions**:
- Component status checking
- Health metrics collection
- Threshold monitoring
- Alert generation
- Dashboard updates

### 7.2 API Endpoint Monitoring
**Participants**: APIMonitor, EndpointTracker, PerformanceAnalyzer, ErrorDetector
**Key Interactions**:
- Response time monitoring
- Error rate tracking
- Performance analysis
- Anomaly detection

### 7.3 Database Performance Monitoring
**Participants**: DatabaseMonitor, MongoDB Instances, QueryAnalyzer, PerformanceTracker
**Key Interactions**:
- Query performance tracking
- Connection monitoring
- Resource usage analysis
- Optimization recommendations

### 7.4 Log Management System
**Participants**: LogManager, Logger, LogAggregator, LogAnalyzer, Storage
**Key Interactions**:
- Log collection
- Log aggregation
- Analysis and filtering
- Storage management
- Search and retrieval

### 7.5 Error Handling & Recovery
**Participants**: ErrorHandler, RecoveryManager, NotificationService, SystemController
**Key Interactions**:
- Error detection
- Error classification
- Recovery strategy execution
- Notification sending
- System restoration

### 7.6 Resource Usage Monitoring
**Participants**: ResourceMonitor, SystemProfiler, UsageAnalyzer, OptimizationEngine
**Key Interactions**:
- Resource utilization tracking
- Performance profiling
- Usage analysis
- Optimization recommendations

### 7.7 Security Monitoring
**Participants**: SecurityMonitor, AuthenticationTracker, AccessController, ThreatDetector
**Key Interactions**:
- Access pattern monitoring
- Threat detection
- Security alert generation
- Access control enforcement

### 7.8 Backup & Recovery Operations
**Participants**: BackupManager, DatabaseManager, FileSystemManager, RecoveryService
**Key Interactions**:
- Scheduled backup execution
- Data integrity verification
- Recovery process execution
- Backup validation

---

## 8. Frontend-Backend Communication (10 Diagrams)

### 8.1 Search Interface Interactions
**Participants**: SearchUI, SearchAPI, ResultRenderer, InteractionTracker
**Key Interactions**:
- Query input handling
- Real-time suggestions
- Result display
- User interaction tracking

### 8.2 Authentication Form Handling
**Participants**: AuthUI, AuthAPI, FormValidator, SecurityHandler
**Key Interactions**:
- Form submission
- Client-side validation
- Server-side processing
- Error handling and feedback

### 8.3 Profile Management Interface
**Participants**: ProfileUI, ProfileAPI, FormHandler, ImageUploader
**Key Interactions**:
- Profile data loading
- Form updates
- Image upload handling
- Validation and feedback

### 8.4 Dashboard Data Loading
**Participants**: DashboardUI, StatisticsAPI, ChartRenderer, DataProcessor
**Key Interactions**:
- Dashboard initialization
- Data fetching
- Chart rendering
- Real-time updates

### 8.5 Feedback Form Processing
**Participants**: FeedbackUI, FeedbackAPI, FormProcessor, ConfirmationHandler
**Key Interactions**:
- Feedback form display
- Rating submission
- Comment processing
- Confirmation feedback

### 8.6 Search Filter Management
**Participants**: FilterUI, SearchAPI, PreferenceManager, StateManager
**Key Interactions**:
- Filter option display
- Filter application
- State persistence
- Result updates

### 8.7 Bookmark Interface Operations
**Participants**: BookmarkUI, BookmarkAPI, StateManager, SyncService
**Key Interactions**:
- Bookmark toggle
- List display
- Organization features
- Synchronization

### 8.8 History Management Interface
**Participants**: HistoryUI, HistoryAPI, TimelineRenderer, SearchHandler
**Key Interactions**:
- History display
- Search replay
- History management
- Privacy controls

### 8.9 Real-time Notifications
**Participants**: NotificationUI, NotificationAPI, EventHandler, WebSocket
**Key Interactions**:
- Event subscription
- Notification delivery
- User interaction
- State synchronization

### 8.10 Progressive Web App Features
**Participants**: PWA Controller, ServiceWorker, CacheManager, OfflineHandler
**Key Interactions**:
- Offline capability
- Cache management
- Background sync
- Push notifications

---

## 9. API Integration & Data Flow (6 Diagrams)

### 9.1 RESTful API Request Processing
**Participants**: API Gateway, Request Handler, Authentication Middleware, Response Formatter
**Key Interactions**:
- Request routing
- Authentication verification
- Processing execution
- Response formatting

### 9.2 Rate Limiting & Throttling
**Participants**: RateLimiter, RequestTracker, ThrottleController, ErrorHandler
**Key Interactions**:
- Request counting
- Limit enforcement
- Throttling application
- Error responses

### 9.3 API Documentation Generation
**Participants**: DocumentationGenerator, APIInspector, TemplateEngine, PublishingService
**Key Interactions**:
- API endpoint discovery
- Documentation generation
- Template rendering
- Publication workflow

### 9.4 Cross-Origin Resource Sharing (CORS)
**Participants**: CORS Handler, Origin Validator, Security Controller, Response Processor
**Key Interactions**:
- Origin validation
- Header processing
- Permission checking
- Response modification

### 9.5 Data Serialization & Validation
**Participants**: Serializer, Validator, Schema Manager, Error Handler
**Key Interactions**:
- Data validation
- Format conversion
- Schema validation
- Error handling

### 9.6 API Versioning Management
**Participants**: VersionController, RequestRouter, CompatibilityChecker, ResponseAdapter
**Key Interactions**:
- Version detection
- Route selection
- Compatibility checking
- Response adaptation

---

## 10. Performance & Optimization (5 Diagrams)

### 10.1 Query Performance Optimization
**Participants**: QueryOptimizer, PerformanceAnalyzer, CacheManager, IndexOptimizer
**Key Interactions**:
- Query analysis
- Optimization strategy
- Cache utilization
- Index optimization

### 10.2 Memory Management & Caching
**Participants**: MemoryManager, CacheController, GarbageCollector, PerformanceMonitor
**Key Interactions**:
- Memory allocation
- Cache management
- Cleanup operations
- Performance monitoring

### 10.3 Database Query Optimization
**Participants**: QueryOptimizer, IndexManager, StatisticsCollector, PerformanceAnalyzer
**Key Interactions**:
- Query plan analysis
- Index optimization
- Statistics collection
- Performance improvement

### 10.4 Load Balancing & Scaling
**Participants**: LoadBalancer, ServerManager, HealthChecker, TrafficRouter
**Key Interactions**:
- Load distribution
- Server health monitoring
- Traffic routing
- Scale-up/down decisions

### 10.5 Asynchronous Processing
**Participants**: AsyncProcessor, TaskQueue, WorkerManager, ResultCollector
**Key Interactions**:
- Task queuing
- Worker allocation
- Asynchronous execution
- Result collection

---

## Summary

This comprehensive list includes **113 sequence diagrams** covering all major system interactions in APOSSS:

- **Authentication & Session Management**: 12 diagrams
- **Search & Query Processing**: 15 diagrams  
- **Feedback & Learning**: 8 diagrams
- **User Interaction & Personalization**: 10 diagrams
- **External Service Integration**: 7 diagrams
- **Data Processing & ML Pipelines**: 12 diagrams
- **System Administration & Monitoring**: 8 diagrams
- **Frontend-Backend Communication**: 10 diagrams
- **API Integration & Data Flow**: 6 diagrams
- **Performance & Optimization**: 5 diagrams

Each diagram would show the temporal sequence of messages between different actors (users, services, databases, external APIs) and provide valuable insights for:

- **Developers**: Understanding system architecture and integration points
- **System Administrators**: Monitoring and troubleshooting workflows
- **Quality Assurance**: Testing scenarios and edge cases
- **Business Analysts**: Understanding user journeys and system capabilities
- **Security Teams**: Analyzing authentication and authorization flows
- **Performance Engineers**: Identifying optimization opportunities

These sequence diagrams would be particularly valuable for documenting the complex AI/ML workflows, multi-database search processes, and real-time user interaction patterns that make APOSSS a sophisticated academic search system. 