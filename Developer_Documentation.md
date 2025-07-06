# APOSSS Developer Documentation
## AI-Powered Open-Science Semantic Search System

### Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Module-wise Breakdown](#module-wise-breakdown)
4. [Data Flow Analysis](#data-flow-analysis)
5. [Key Algorithms and Models](#key-algorithms-and-models)
6. [Installation and Setup](#installation-and-setup)
7. [API Documentation](#api-documentation)
8. [Troubleshooting](#troubleshooting)

---

## Project Overview

APOSSS (AI-Powered Open-Science Semantic Search System) is a sophisticated multi-database search system that leverages artificial intelligence to provide intelligent semantic search across multiple academic and research databases. The system is currently in **Phase 3**, featuring advanced AI ranking algorithms and comprehensive user feedback collection.

### Key Features
- **Multi-database Search**: Searches across 6 MongoDB databases (Academic Library, Experts System, Research Papers, Laboratories, Funding, APOSSS)
- **Hybrid Ranking**: Combines heuristic, TF-IDF, embedding-based, and learning-to-rank algorithms
- **Semantic Understanding**: Uses Gemini-2.0-flash LLM for query processing and sentence transformers for embeddings
- **User Management**: Complete authentication system with OAuth support (Google, ORCID)
- **Feedback System**: Collects and stores user feedback for continuous improvement
- **Multilingual Support**: Handles queries in multiple languages with automatic translation

### Technology Stack
- **Backend**: Python 3.x, Flask
- **Databases**: MongoDB (6 databases)
- **AI/ML**: Google Gemini LLM, sentence-transformers, XGBoost, FAISS, scikit-learn, NetworkX
- **Ranking**: BM25 (rank-bm25), TF-IDF (scikit-learn), PageRank (NetworkX)
- **Authentication**: JWT, OAuth 2.0, bcrypt
- **Frontend**: Vanilla JavaScript, Tailwind CSS

---

## System Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend UI   │────│   Flask API     │────│Query Processor  │────│   LLM Processor │
│  (JavaScript)   │    │   (app.py)      │    │  (validation)   │    │   (Gemini)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
                               │                        │
                    ┌──────────┼──────────┐            │
                    │          │          │            │
           ┌─────────────┐ ┌──────────┐ ┌─────────────┐ │ ┌─────────────┐
           │Search Engine│ │ Ranking  │ │  Feedback   │ │ │User Manager │
           │   Module    │ │  Engine  │ │   System    │ │ │& OAuth      │
           └─────────────┘ └─────┬────┘ └─────────────┘ │ └─────────────┘
                    │          │          │            │
                    │    ┌─────┴──────┐   │            │
                    │    │  Ranking   │   │            │
                    │    │ Components │   │            │
                    │    │            │   │            │
                    │   ┌┴────────────┴┐  │            │
                    │   │ ・LTR Ranker  │  │            │
                    │   │ ・LLM Ranker  │  │            │
                    │   │ ・Embedding   │  │            │
                    │   │ ・Knowledge   │  │            │
                    │   │   Graph      │  │            │
                    │   │ ・Enhanced    │  │            │
                    │   │   Features   │  │            │
                    │   └──────────────┘  │            │
                    │          │          │            │
              ┌─────────────────────────────────────────────────┐
              │               Database Manager                  │
              └─────────────────────────────────────────────────┘
                               │
    ┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
    │Academic  │ Experts  │Research  │Laborato- │ Funding  │ APOSSS   │
    │Library   │ System   │ Papers   │ ries     │ System   │ DB       │
    │(MongoDB) │(MongoDB) │(MongoDB) │(MongoDB) │(MongoDB) │(MongoDB) │
    └──────────┘└──────────┘└──────────┘└──────────┘└──────────┘└──────────┘
```

### Component Interaction Flow

1. **User Query Input** → Frontend UI
2. **API Request** → Flask Application (app.py)
3. **Query Validation** → Query Processor (cleaning, validation)
4. **Query Analysis** → LLM Processor (Gemini API for NLP)
5. **Result Validation** → Query Processor (structured output)
6. **Multi-Database Search** → Search Engine Module
7. **Result Ranking** → Ranking Engine (Multiple Algorithms)
8. **Optional LLM Re-ranking** → LLM Ranker (advanced analysis)
9. **Response Delivery** → Frontend UI
10. **User Authentication** → User Manager & OAuth Manager
11. **User Feedback** → Feedback System
12. **Learning Loop** → LTR Model Training

---

## Module-wise Breakdown

### 1. LLM Processor (`modules/llm_processor.py`)

**Purpose**: Processes user queries using Google's Gemini-2.0-flash LLM for advanced query understanding.

**Internal Functionality**:
- **Language Detection**: Automatically detects query language using AI
- **Translation**: Translates non-English queries to English for consistent processing
- **Spelling Correction**: Corrects spelling errors in queries
- **Intent Detection**: Identifies user intent (find_research, find_expert, find_equipment, etc.)
- **Entity Extraction**: Extracts people, organizations, technologies, concepts
- **Keyword Extraction**: Identifies primary, secondary, and technical keywords
- **Synonym Generation**: Generates related terms and synonyms
- **Academic Field Mapping**: Maps queries to relevant academic disciplines

**Inputs**: Raw user query string

**Outputs**: Structured JSON containing:
```json
{
  "language_detection": {...},
  "translation": {...},
  "corrected_query": "string",
  "intent": {...},
  "entities": {...},
  "keywords": {...},
  "synonyms_and_related": {...},
  "academic_fields": {...},
  "search_context": {...}
}
```

**Dependencies**: 
- `google-generativeai` for Gemini API
- Environment variable: `GEMINI_API_KEY`

**Configuration**:
- Model: `gemini-2.0-flash-exp`
- Temperature: 0.3
- Safety settings: Block medium and above harmful content

### 2. Database Manager (`modules/database_manager.py`)

**Purpose**: Manages connections to all MongoDB databases and provides unified access interface.

**Internal Functionality**:
- **Connection Management**: Establishes and maintains connections to 6 MongoDB databases
- **Connection Testing**: Provides health check functionality
- **Error Handling**: Graceful handling of connection failures
- **Collection Access**: Unified interface for accessing collections

**Databases Managed**:
- **academic_library**: books, journals, projects
- **experts_system**: experts, certificates  
- **research_papers**: articles, conferences, theses
- **laboratories**: equipments, materials
- **funding**: research_projects, institutions, funding_records
- **aposss**: users, feedback, interactions, preferences

**Configuration**: MongoDB URI environment variables for each database

### 3. Search Engine (`modules/search_engine.py`)

**Purpose**: Performs multi-database search using hybrid approach combining traditional keyword search with semantic search.

**Internal Functionality**:
- **Hybrid Search**: Combines pre-built FAISS index search with traditional MongoDB queries
- **Query Building**: Constructs MongoDB queries from processed query parameters
- **Result Aggregation**: Merges results from multiple databases
- **Result Standardization**: Converts diverse document formats to unified structure
- **Semantic Enhancement**: Uses pre-built embeddings index for semantic similarity

**Search Strategy**:
1. **Semantic Search**: Uses pre-built FAISS index for fast semantic similarity
2. **Keyword Search**: Traditional MongoDB text search and regex matching  
3. **Result Merging**: Combines and deduplicates results from both approaches
4. **Multi-Modal Ranking**: Results are then processed through ranking engine with multiple modes

**Performance Optimizations**:
- Pre-built FAISS index for semantic search
- Batch processing for embedding calculations
- Caching of frequently accessed embeddings

### 4. Ranking Engine (`modules/ranking_engine.py`)

**Purpose**: Main orchestrator that combines multiple ranking algorithms to order search results by relevance.

**Core Ranking Algorithms**:

#### a) Heuristic Ranking (20% weight in hybrid mode)
- Keyword matching in titles vs. descriptions
- Exact phrase matching bonuses
- Field-specific scoring (title > abstract > content)
- Recent publication bonuses
- Availability status bonuses

#### b) TF-IDF Similarity (20% weight in hybrid mode)
- Uses scikit-learn TfidfVectorizer
- Cosine similarity between query and document vectors
- N-gram analysis (1-2 grams)
- Stop word filtering

#### c) BM25 Scoring
- Implemented in ranking engine for base scoring
- Also used as features in LTR model
- Separate BM25 scores for title and description

#### d) Intent Alignment (20% weight in hybrid mode)
- Resource type preference based on detected intent
- Academic field alignment
- Intent confidence weighting

#### e) Embedding Similarity (20% weight in hybrid mode)
- Real-time semantic similarity via embedding_ranker
- Model: `paraphrase-multilingual-MiniLM-L12-v2`
- 384-dimensional embeddings
- Cosine similarity calculation

#### f) Personalization Scoring (20% weight in hybrid mode)
- User type preferences from interaction history
- Field/category preferences
- Author preferences
- Recency and complexity preferences
- Query similarity to past successful searches

**Ranking Modes**: The system supports three distinct ranking modes:

#### 1. Traditional Mode (`"traditional"`)
Uses weighted combination of traditional algorithms:
- **Heuristic Ranking** (20% weight): Keyword matching, field scoring, recency bonuses
- **TF-IDF Similarity** (20% weight): Cosine similarity with scikit-learn
- **Intent Alignment** (20% weight): Resource type preferences, academic field matching
- **Embedding Similarity** (20% weight): Real-time semantic similarity via sentence transformers
- **Personalization Scoring** (20% weight): User preferences and interaction history

#### 2. LTR-Only Mode (`"ltr_only"`)
- **Learning-to-Rank (XGBoost)**: Uses all traditional scores as 50+ features for ML-based ranking
- **Pure ML Approach**: Relies entirely on trained XGBoost model with pairwise ranking objective

#### 3. Hybrid Mode (`"hybrid"`) - Default
- **LTR Score** (70% weight): XGBoost model using traditional scores as features
- **Traditional Score** (30% weight): Weighted combination of the 5 traditional algorithms
- **Best of Both**: Combines ML learning with traditional ranking reliability

**Additional Components**:
- **BM25 Scoring**: Used as features in LTR model and for base scoring
- **Knowledge Graph**: PageRank and authority scoring (provides features for LTR)
- **LLM Re-ranking**: Available but NOT integrated in main search flow (standalone module)

### 5. Embedding Ranker (`modules/embedding_ranker.py`)

**Purpose**: Provides semantic similarity calculations using sentence transformers and FAISS indexing.

**Internal Functionality**:
- **Pre-built Index**: FAISS index of pre-computed document embeddings
- **Real-time Similarity**: On-demand embedding calculation for new queries
- **Caching**: Intelligent caching of embeddings to improve performance
- **Multilingual Support**: Supports multiple languages through multilingual models

**Model Details**:
- **Model**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Dimensions**: 384
- **Index Type**: FAISS IndexFlatIP (Inner Product)
- **Similarity Metric**: Cosine similarity

### 6. LTR Ranker (`modules/ltr_ranker.py`)

**Purpose**: Implements machine learning-based ranking using Learning-to-Rank with XGBoost.

**Feature Engineering**:
- **Textual Features**: BM25, n-gram matches, term proximity
- **Metadata Features**: Publication date, author authority, citation count
- **User Features**: Click-through rates, feedback scores
- **LLM Features**: Intent matching, entity alignment
- **Current Scores**: Incorporates other ranking algorithm scores

**ML Model**:
- **Algorithm**: XGBoost with pairwise ranking objective
- **Features**: 50+ engineered features per query-document pair
- **Training**: Uses historical user feedback as relevance labels
- **Evaluation**: NDCG (Normalized Discounted Cumulative Gain)

### 7. Query Processor (`modules/query_processor.py`)

**Purpose**: Orchestrates the complete query processing workflow and ensures data quality.

**Core Functionality**:
- **Query Cleaning**: Removes noise characters and normalizes whitespace
- **LLM Orchestration**: Coordinates with LLM processor for query analysis
- **Result Validation**: Ensures all required fields are present and properly formatted
- **Data Enhancement**: Adds processing statistics and metadata
- **Error Handling**: Graceful fallback when LLM processing fails

**Quality Assurance**:
- **Field Validation**: Ensures required fields (intent, entities, keywords, etc.)
- **Keyword Cleaning**: Filters and limits keywords per category (max 10)
- **Intent Validation**: Validates against predefined intent types
- **Statistics Generation**: Calculates complexity scores and processing metrics

**Integration**:
- **Input**: Raw user query strings
- **Output**: Validated, structured query data for search engine
- **Fallback**: Provides default values when processing fails

### 8. User Manager (`modules/user_manager.py`)

**Purpose**: Comprehensive user management system handling authentication, profiles, and personalization.

**Authentication Features**:
- **Registration/Login**: Email/password authentication with bcrypt hashing
- **JWT Tokens**: Secure session management with configurable expiry
- **Social Authentication**: Integration with OAuth manager for Google/ORCID
- **Email Verification**: SMTP-based verification with codes
- **Password Management**: Secure password changes with validation

**User Profile Management**:
- **Complete Profiles**: First/last name, academic fields, organization, department, role, bio
- **Avatar System**: Profile picture upload, retrieval, and deletion
- **Academic Data**: Institution, job title, research specializations
- **Statistics Tracking**: Search counts, feedback patterns, favorite topics

**Preferences System**:
- **Search Preferences**: Resource types, databases, language, results per page
- **Notification Settings**: Email notifications, search alerts, feedback requests
- **Privacy Controls**: Profile visibility, interaction tracking, personalization
- **UI Preferences**: Theme, display language, interface settings

**Interaction Tracking & Personalization**:
- **Search History**: Detailed query logs with metadata and timestamps
- **Feedback Tracking**: User ratings, comments, and interaction patterns
- **Behavioral Analysis**: Pattern recognition for personalized recommendations
- **Statistics Generation**: Usage analytics, preference analysis, engagement metrics

**Database Management**:
- **Collections**: Users, interactions, preferences, sessions, verification codes
- **Indexing**: Optimized indexes for performance (email, username, user_id, timestamps)
- **Data Integrity**: Unique constraints and validation rules

### 9. OAuth Manager (`modules/oauth_manager.py`)

**Purpose**: Handles social authentication flows for Google and ORCID integration.

**Supported Providers**:
- **Google OAuth**: Full Google authentication with email, profile, and picture
- **ORCID OAuth**: Academic authentication with researcher profiles
- **Configuration**: Environment-based client ID and secret management

**OAuth Flow Management**:
- **Authorization URLs**: Generates secure authorization URLs with state tokens
- **Token Exchange**: Exchanges authorization codes for access tokens
- **User Info Retrieval**: Fetches user profile data from OAuth providers
- **State Validation**: CSRF protection with secure random state tokens

**Data Integration**:
- **Profile Mapping**: Maps OAuth provider data to internal user structure
- **Email Verification**: Automatic email verification for OAuth users
- **Account Linking**: Links social accounts to existing user accounts
- **Fallback Handling**: Graceful error handling for OAuth failures

**Security Features**:
- **Secure Tokens**: Uses secrets.token_urlsafe for state generation
- **HTTPS Enforcement**: Requires secure connections for OAuth flows
- **Error Isolation**: Comprehensive error handling and logging

### 10. LLM Ranker (`modules/llm_ranker.py`)

**Purpose**: Provides AI-powered re-ranking and relevance analysis using Google's Gemini LLM.

**Internal Functionality**:
- **Result Re-ranking**: Uses LLM to analyze and re-rank search results
- **Relevance Scoring**: Assigns detailed relevance scores (0-1) with explanations
- **Intent Matching**: Analyzes how well results match query intent
- **Key Aspects Analysis**: Identifies important aspects of each result
- **Quality Assessment**: Evaluates reliability, completeness, and overall quality

**LLM Integration**:
- **Model**: Uses same Gemini-2.0-flash model as LLM processor
- **Structured Output**: Returns JSON with scores, explanations, and analysis
- **Score Integration**: Adds LLM scores to result score breakdown
- **Error Handling**: Graceful fallback if LLM analysis fails

### 11. LTR Ranker (`modules/ltr_ranker.py`)

**Purpose**: Implements machine learning-based ranking using Learning-to-Rank with XGBoost.

**Feature Engineering** (50+ features):
- **Current Scores**: Incorporates heuristic, TF-IDF, intent, embedding scores
- **BM25 Features**: BM25 scores for title and description
- **Textual Features**: N-gram matches, term proximity, exact matches
- **Metadata Features**: Publication date, availability, authority metrics
- **LLM Features**: Intent confidence, field matching, entity alignment
- **User Features**: Click-through rates, feedback scores, interaction patterns
- **Graph Features**: PageRank, authority scores, connection strength

**ML Model**:
- **Algorithm**: XGBoost with pairwise ranking objective
- **Training**: Uses historical user feedback as relevance labels
- **Evaluation**: NDCG (Normalized Discounted Cumulative Gain)
- **Features**: Comprehensive feature extraction from multiple sources

**Integration**:
- **Hybrid Mode**: Combines LTR scores with traditional ranking (70% LTR, 30% traditional)
- **Fallback**: Returns neutral scores if model not trained
- **Continuous Learning**: Can retrain with new feedback data

### 12. Knowledge Graph (`modules/knowledge_graph.py`)

**Purpose**: Models relationships between papers, authors, keywords, equipment, and institutions using graph theory.

**Graph Structure**:
- **Nodes**: Papers, authors, keywords, equipment, projects, institutions
- **Edges**: Citations, authorship, keyword associations, collaborations, affiliations
- **Directed Graph**: Uses NetworkX DiGraph for relationship modeling

**Graph Analytics**:
- **PageRank Scoring**: Calculates authority scores for all nodes
- **Path Analysis**: Finds shortest paths between related entities
- **Connection Strength**: Measures relationship strength between nodes
- **Authority Scoring**: Comprehensive authority calculation based on multiple metrics

**Integration with Ranking**:
- **Authority Features**: Provides authority scores for results
- **Connection Features**: Measures connection strength to query terms
- **Graph Features**: Extracts graph-based features for LTR model
- **Real-time Updates**: Builds graph from search results dynamically

### 13. Feedback System (`modules/feedback_system.py`)

**Purpose**: Collects, stores, and analyzes user feedback for system improvement.

**Feedback Types**:
- **Rating Feedback**: 1-5 star ratings
- **Binary Feedback**: Thumbs up/down
- **Detailed Feedback**: Text comments and suggestions

**Storage Strategy**:
- **Primary**: MongoDB collection in APOSSS database
- **Fallback**: JSON Lines file for reliability
- **Analytics**: Real-time statistics and reporting

**Learning Integration**:
- **Training Data**: Converts feedback to LTR training examples
- **Quality Metrics**: Tracks system performance over time
- **Continuous Improvement**: Feeds back into ranking algorithms

---

## Data Flow Analysis

### 1. Query Processing Flow

```
User Query (Any Language)
    ↓
Query Cleaning & Validation (query_processor.py)
    ↓
Language Detection & Translation (llm_processor.py via LLM)
    ↓
Spelling Correction & Enhancement (llm_processor.py via LLM)
    ↓
Intent Detection & Entity Extraction (llm_processor.py via LLM)
    ↓
Keyword & Synonym Generation (llm_processor.py via LLM)
    ↓
Result Validation & Enhancement (query_processor.py)
    ↓
Structured Query Parameters
```

### 2. Search Execution Flow

```
Structured Query Parameters
    ↓
┌─────────────────────┬─────────────────────┐
│   Semantic Search   │   Keyword Search    │
│   (FAISS Index)     │   (MongoDB Queries) │
└─────────────────────┴─────────────────────┘
    ↓                           ↓
Semantic Results            Traditional Results
    ↓                           ↓
        Result Merging & Deduplication
                    ↓
            Aggregated Results
```

### 3. Ranking Flow

```
Raw Search Results + Query Context
    ↓
┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
│  Heuristic  │   TF-IDF    │   Intent    │ Embedding   │Personalization│
│   Scoring   │   Scoring   │   Scoring   │   Scoring   │   Scoring   │
└─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘
    ↓               ↓               ↓               ↓               ↓
                Traditional Weighted Score (20% each)
                            ↓
                ┌───────────────────────────┐
                │    Ranking Mode Logic     │
                └───────────────────────────┘
                            ↓
    ┌──────────────────┬──────────────────┬──────────────────┐
    │   Traditional    │    LTR Only      │     Hybrid       │
    │   (100% trad)    │   (100% LTR)     │ (70% LTR +       │
    │                  │                  │  30% trad)       │
    └──────────────────┴──────────────────┴──────────────────┘
                            ↓
                    Final Ranked Results
                    + Relevance Categories
```

### 4. User Interaction Flow

```
User Views Results
    ↓
User Provides Feedback (Rating/Comments)
    ↓
Feedback Storage (MongoDB + File Backup)
    ↓
Interaction Tracking (User Profile Update)
    ↓
Training Data Generation (LTR Features + Labels)
    ↓
Model Retraining (Batch Process)
    ↓
Improved Ranking Performance
```

---

## Key Algorithms and Models

### 1. Language Processing (Gemini-2.0-flash)

**Algorithm**: Transformer-based large language model
- **Purpose**: Query understanding, translation, entity extraction
- **Input**: Natural language query (any language)
- **Output**: Structured query analysis with 95%+ accuracy
- **Features**: Multilingual, context-aware, intent detection

### 2. Semantic Embeddings (Sentence Transformers)

**Model**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Architecture**: Transformer-based sentence embedding model
- **Dimensions**: 384-dimensional dense vectors
- **Training**: Paraphrase pairs from 50+ languages
- **Performance**: Fast inference (~10ms per sentence)

### 3. Vector Search (FAISS)

**Algorithm**: Facebook AI Similarity Search
- **Index Type**: IndexFlatIP (Inner Product for cosine similarity)
- **Search Complexity**: O(d*n) where d=dimensions, n=documents
- **Performance**: Sub-second search across 100K+ documents
- **Memory**: ~1.5GB for 100K documents at 384 dimensions

### 4. Learning-to-Rank (XGBoost)

**Algorithm**: Gradient Boosted Decision Trees
- **Objective**: Pairwise ranking with NDCG optimization
- **Features**: 50+ hand-crafted and learned features
- **Training**: Historical user feedback (implicit and explicit)
- **Performance**: 15-20% improvement over heuristic ranking

### 5. BM25 Scoring

**Algorithm**: Best Matching 25 (Okapi BM25)
- **Implementation**: Multiple implementations across modules
- **Parameters**: k1=1.2, b=0.75 (standard BM25 parameters)
- **Usage**: Base scoring in ranking engine, features in LTR, enhanced text features
- **Normalization**: Scores normalized to [0,1] range for consistency

### 6. Text Similarity (TF-IDF + Cosine)

**Algorithm**: Term Frequency-Inverse Document Frequency
- **Vectorization**: scikit-learn TfidfVectorizer
- **N-grams**: 1-gram and 2-gram analysis
- **Similarity**: Cosine similarity between query and document vectors
- **Normalization**: L2 normalization for fair comparison

### 7. Knowledge Graph Analytics (NetworkX + PageRank)

**Algorithm**: PageRank on academic knowledge graph
- **Graph**: NetworkX DiGraph with academic entities and relationships
- **PageRank**: Alpha=0.85, measures node authority in academic network
- **Features**: Authority scores, connection strength, shortest paths
- **Integration**: Provides graph-based features for ranking algorithms

### 8. LLM-Based Re-ranking (Gemini-2.0-flash)

**Algorithm**: Large language model analysis for result relevance
- **Purpose**: Advanced semantic understanding and quality assessment
- **Output**: Structured JSON with scores, explanations, and analysis
- **Integration**: Optional re-ranking layer with detailed relevance analysis
- **Fallback**: Graceful degradation if LLM analysis fails

---

## Installation and Setup

### Prerequisites
- Python 3.8+
- MongoDB 4.0+
- 8GB+ RAM (for embedding models)
- Google Gemini API key

### Environment Setup

1. **Clone Repository**
```bash
git clone <repository-url>
cd APOSSS
```

2. **Create Virtual Environment**
```bash
python -m venv APOSSSenv
source APOSSSenv/bin/activate  # Linux/Mac
# or
APOSSSenv\Scripts\activate     # Windows
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure Environment Variables**
```bash
cp config.env.example config.env
# Edit config.env with your settings:

# Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key

# MongoDB URIs
MONGODB_URI_ACADEMIC_LIBRARY=mongodb://localhost:27017/Academic_Library
MONGODB_URI_EXPERTS_SYSTEM=mongodb://localhost:27017/Experts_System
MONGODB_URI_RESEARCH_PAPERS=mongodb://localhost:27017/Research_Papers
MONGODB_URI_LABORATORIES=mongodb://localhost:27017/Laboratories
MONGODB_URI_FUNDING=mongodb://localhost:27017/Funding
MONGODB_URI_APOSSS=mongodb://localhost:27017/APOSSS

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
PORT=5000
```

5. **Initialize Databases**
```bash
# Ensure MongoDB is running
sudo systemctl start mongod

# Test database connections
python -c "from modules.database_manager import DatabaseManager; dm = DatabaseManager(); print(dm.test_connections())"
```

6. **Build Search Index (Optional)**
```bash
python build_index.py
```

7. **Run Application**
```bash
python app.py
```

### Production Deployment

1. **Environment Variables**
```bash
# Production settings
FLASK_ENV=production
FLASK_DEBUG=False

# Security
JWT_SECRET_KEY=your_secure_secret_key
SESSION_SECRET_KEY=your_session_secret

# Database (consider MongoDB Atlas)
MONGODB_URI_PREFIX=mongodb+srv://user:pass@cluster/

# OAuth Credentials
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
ORCID_CLIENT_ID=your_orcid_client_id
ORCID_CLIENT_SECRET=your_orcid_client_secret
```

2. **Performance Optimization**
```bash
# Pre-build embedding index
python build_index.py --production

# Set up process manager (PM2/Supervisor)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

---

## API Documentation

### Core Search Endpoints

#### POST `/api/search`
**Purpose**: Main search endpoint with AI ranking

**Request Body**:
```json
{
  "query": "machine learning for medical diagnosis",
  "user_id": "optional_user_id",
  "database_filters": ["academic_library", "research_papers"],
  "ranking_mode": "hybrid",
  "limit": 20
}
```

**Response**:
```json
{
  "results": [...],
  "total_results": 156,
  "search_metadata": {
    "query_analysis": {...},
    "search_time_ms": 450,
    "ranking_scores": {...}
  },
  "relevance_categories": {
    "high": [...],
    "medium": [...],
    "low": [...]
  }
}
```

#### POST `/api/feedback`
**Purpose**: Submit user feedback for results

**Request Body**:
```json
{
  "query_id": "uuid",
  "result_id": "result_id",
  "rating": 5,
  "feedback_type": "rating",
  "comment": "Very helpful result"
}
```

### Authentication Endpoints

#### POST `/api/auth/register`
**Purpose**: User registration

#### POST `/api/auth/login`
**Purpose**: User authentication

#### GET `/api/auth/google`
**Purpose**: Google OAuth authentication

### System Endpoints

#### GET `/api/health`
**Purpose**: System health check

#### GET `/api/embedding/stats`
**Purpose**: Embedding system statistics

#### POST `/api/ltr/train`
**Purpose**: Trigger LTR model training

---

## Troubleshooting

### Common Issues

#### 1. LLM Connection Errors
**Symptoms**: "GEMINI_API_KEY environment variable is required"
**Solution**: 
- Ensure `GEMINI_API_KEY` is set in config.env
- Verify API key is valid and has quota remaining
- Check network connectivity to Google AI APIs

#### 2. Database Connection Failures
**Symptoms**: "Failed to connect to [database] database"
**Solutions**:
- Verify MongoDB is running: `sudo systemctl status mongod`
- Check MongoDB URIs in config.env
- Ensure databases exist and are accessible
- Verify network connectivity and authentication

#### 3. Embedding Model Download Issues
**Symptoms**: Slow startup or download errors
**Solutions**:
- Ensure stable internet connection
- Use lighter model: `all-MiniLM-L6-v2` instead of multilingual variant
- Pre-download models: `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"`

#### 4. Memory Issues
**Symptoms**: Out of memory errors during embedding calculations
**Solutions**:
- Reduce batch size in `build_index.py`
- Use lighter embedding model
- Increase system RAM or use swap memory
- Process documents in smaller batches

#### 5. Search Performance Issues
**Symptoms**: Slow search response times
**Solutions**:
- Build and use pre-computed FAISS index: `python build_index.py`
- Enable embedding caching
- Optimize MongoDB indexes
- Use smaller result limits for testing

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Monitoring
Monitor key metrics:
- Search response time
- LLM API latency
- Database query performance
- Memory usage during embedding operations
- User feedback collection rates

### Development Tips
1. Use `/api/health` endpoint to verify system status
2. Test individual modules in isolation
3. Use smaller datasets during development
4. Monitor log files for detailed error information
5. Use MongoDB Compass for database debugging

---

This documentation provides a comprehensive technical overview of the APOSSS system for developers. For user-facing documentation, refer to the README.md file. For academic analysis, see the accompanying Academic Report. 