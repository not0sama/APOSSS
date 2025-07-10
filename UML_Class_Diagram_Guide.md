# APOSSS Backend UML Class Diagram Guide

## Overview

The **`APOSSS_Backend_UML_Class_Diagram.drawio`** file contains a comprehensive UML class diagram that visualizes the backend architecture of the AI-Powered Open-Science Semantic Search System (APOSSS).

## How to Open the Diagram

1. **Online**: Go to [draw.io](https://app.diagrams.net/) and open the `.drawio` file
2. **Offline**: Download the draw.io desktop application and open the file
3. **VS Code**: Install the "Draw.io Integration" extension and open the file directly

## Diagram Components

### 🏛️ **Core Architecture Classes**

#### **APOSSSApp** (Main Flask Application)
- **Type**: Flask Application Controller
- **Purpose**: Main entry point and API endpoint manager
- **Key Methods**: All HTTP route handlers (`home()`, `search()`, `login()`, etc.)
- **Dependencies**: All other system components

#### **DatabaseManager**
- **Type**: Data Access Layer
- **Purpose**: Manages connections to 6 MongoDB databases
- **Key Methods**: `get_database()`, `test_connections()`
- **Databases**: academic_library, experts_system, research_papers, laboratories, funding, aposss

### 👤 **User Management Classes**

#### **UserManager**
- **Type**: User Service Layer
- **Purpose**: Complete user lifecycle management
- **Key Methods**: `register_user()`, `authenticate_user()`, `verify_token()`, `update_profile()`
- **Features**: JWT authentication, profile management, interaction tracking

#### **OAuthManager**
- **Type**: Authentication Service
- **Purpose**: Social authentication (Google, ORCID)
- **Key Methods**: `get_authorization_url()`, `exchange_code_for_token()`, `process_oauth_login()`

### 🧠 **AI & Processing Classes**

#### **LLMProcessor**
- **Type**: AI Processing Service
- **Purpose**: Query understanding using Gemini-2.0-flash
- **Key Methods**: `process_query()`, `_extract_entities()`, `_detect_intent()`, `_generate_keywords()`
- **Capabilities**: Language detection, translation, entity extraction, intent detection

#### **QueryProcessor**
- **Type**: Processing Orchestrator
- **Purpose**: Coordinates query processing workflow
- **Key Methods**: `process_query()`, `validate_query()`, `_clean_query()`
- **Role**: Validates and enhances LLM processing results

### 🔍 **Search & Ranking Classes**

#### **SearchEngine**
- **Type**: Search Service
- **Purpose**: Multi-database search execution
- **Key Methods**: `search()`, `_search_academic_library()`, `_search_experts_system()`
- **Strategy**: Hybrid search (semantic + traditional keyword search)

#### **RankingEngine**
- **Type**: Ranking Orchestrator
- **Purpose**: Combines multiple ranking algorithms
- **Key Methods**: `rank_search_results()`, `_calculate_heuristic_scores()`, `train_ltr_model()`
- **Algorithms**: Heuristic, TF-IDF, Intent, Embedding, Personalization, LTR

#### **LTRRanker**
- **Type**: Machine Learning Ranker
- **Purpose**: Learning-to-Rank with XGBoost
- **Key Methods**: `extract_features()`, `train_model()`, `rank_results()`
- **Features**: 50+ engineered features, NDCG evaluation

#### **EmbeddingRanker**
- **Type**: Semantic Similarity Service
- **Purpose**: Vector-based semantic search
- **Key Methods**: `search_similar_documents()`, `calculate_realtime_similarity()`, `build_index()`
- **Technology**: Sentence Transformers + FAISS indexing

#### **KnowledgeGraph**
- **Type**: Graph Analytics Service
- **Purpose**: Academic relationship modeling
- **Key Methods**: `add_paper()`, `calculate_pagerank()`, `get_authority_score()`
- **Technology**: NetworkX + PageRank algorithm

#### **EnhancedTextFeatures**
- **Type**: Feature Engineering Service
- **Purpose**: Advanced text analysis
- **Key Methods**: `extract_all_features()`, `calculate_bm25_scores()`, `calculate_ngram_overlap()`
- **Features**: BM25, n-grams, term proximity, complexity analysis

### 📊 **Data Collection Classes**

#### **FeedbackSystem**
- **Type**: Learning Data Service
- **Purpose**: User feedback collection and storage
- **Key Methods**: `submit_feedback()`, `get_result_feedback()`, `get_recent_feedback()`
- **Storage**: MongoDB + file backup

## 🔗 **Relationships and Dependencies**

### **Dependency Arrows (→)**
The dashed arrows show dependency relationships:

- **APOSSSApp** depends on all other components (dependency injection)
- **UserManager** depends on **DatabaseManager** for data access
- **SearchEngine** depends on **DatabaseManager** for multi-database queries
- **RankingEngine** depends on **LLMProcessor**, **EmbeddingRanker**, **LTRRanker**, **KnowledgeGraph**
- **QueryProcessor** depends on **LLMProcessor** for AI processing
- **LTRRanker** depends on **EnhancedTextFeatures** for feature engineering
- **FeedbackSystem** depends on **DatabaseManager** for feedback storage

### **Key Architectural Patterns**

1. **Dependency Injection**: APOSSSApp injects dependencies into all components
2. **Service Layer Pattern**: Each component encapsulates specific business logic
3. **Strategy Pattern**: RankingEngine uses multiple ranking strategies
4. **Repository Pattern**: DatabaseManager abstracts data access
5. **Facade Pattern**: QueryProcessor facades complex LLM interactions

## 🎯 **Design Principles Demonstrated**

### **Single Responsibility Principle**
Each class has a clear, focused responsibility:
- **DatabaseManager**: Only handles database connections
- **LLMProcessor**: Only handles LLM interactions
- **SearchEngine**: Only handles search operations
- **RankingEngine**: Only orchestrates ranking algorithms

### **Dependency Inversion Principle**
High-level modules (APOSSSApp) depend on abstractions, not concrete implementations.

### **Open/Closed Principle**
New ranking algorithms can be added to RankingEngine without modifying existing code.

### **Composition over Inheritance**
Components are composed together rather than using deep inheritance hierarchies.

## 📋 **How to Use This Diagram**

### **For Developers**
1. **Understanding Architecture**: See how components interact
2. **Dependency Management**: Understand what each component needs
3. **Code Navigation**: Know where to find specific functionality
4. **Testing Strategy**: Identify dependencies for unit testing

### **For System Design**
1. **Scalability Planning**: Identify components that might need scaling
2. **Performance Optimization**: See data flow and potential bottlenecks
3. **Feature Planning**: Understand where new features would fit
4. **Documentation**: Visual reference for system architecture

### **For Maintenance**
1. **Bug Fixing**: Trace issues through component relationships
2. **Refactoring**: Understand impact of changes
3. **Monitoring**: Know which components to monitor
4. **Deployment**: Understand component dependencies for deployment order

## 🔧 **Technical Implementation Notes**

### **Flask Integration**
- All components are initialized in `app.py` during Flask startup
- Dependencies are injected using constructor parameters
- Error handling includes graceful degradation when components fail

### **MongoDB Integration**
- DatabaseManager provides unified access to 6 databases
- Each component gets database access through DatabaseManager
- Connection pooling and health monitoring included

### **AI Model Integration**
- LLMProcessor wraps Gemini-2.0-flash API calls
- EmbeddingRanker uses Sentence Transformers
- LTRRanker uses XGBoost for machine learning

### **Caching Strategy**
- EmbeddingRanker implements comprehensive caching
- LLMProcessor includes result caching
- FAISS indexes provide fast similarity search

## 🚀 **Future Enhancements**

The modular design allows for easy extension:

1. **New Ranking Algorithms**: Add to RankingEngine
2. **Additional Databases**: Extend DatabaseManager
3. **More AI Models**: Extend LLMProcessor or create new processors
4. **Enhanced Features**: Add to EnhancedTextFeatures
5. **Real-time Updates**: Add event-driven components

This UML diagram serves as the definitive reference for understanding the APOSSS backend architecture and guiding development decisions. 