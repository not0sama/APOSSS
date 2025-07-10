# APOSSS System Entities - Comprehensive Description
## AI-Powered Open-Science Semantic Search System

### Table of Contents
1. [System Overview](#system-overview)
2. [Core System Entities](#core-system-entities)
3. [Data Entities](#data-entities)
4. [User Management Entities](#user-management-entities)
5. [Search & Processing Entities](#search--processing-entities)
6. [Ranking & Intelligence Entities](#ranking--intelligence-entities)
7. [Database Entities](#database-entities)
8. [Frontend Entities](#frontend-entities)
9. [Configuration & Infrastructure Entities](#configuration--infrastructure-entities)
10. [Workflow & Process Entities](#workflow--process-entities)
11. [Entity Relationships Matrix](#entity-relationships-matrix)

---

## System Overview

APOSSS is a sophisticated AI-powered search system that integrates multiple databases, advanced ranking algorithms, and machine learning components. The system operates in **Phase 3**, featuring comprehensive AI ranking, user feedback collection, and personalization capabilities.

### Key Architecture Patterns
- **Multi-tier Architecture**: Frontend (HTML/CSS/JS) → API Layer (Flask) → Business Logic (Python Modules) → Data Layer (MongoDB)
- **Microservice-oriented Modules**: Each major component is encapsulated in dedicated modules
- **AI-First Design**: LLM processing, semantic search, and machine learning ranking at the core
- **Event-driven Interactions**: User actions trigger feedback loops and model improvements

---

## Core System Entities

### 1. **APOSSS Application (app.py)**
**Type**: Main Application Entity  
**Purpose**: Central orchestrator and API gateway  

**Attributes**:
- Flask application instance
- CORS configuration
- Component initialization status
- Request routing and middleware
- Authentication middleware
- Error handling and logging

**Relationships**:
- **Initializes**: All system modules and managers
- **Exposes**: REST API endpoints to frontend
- **Manages**: User sessions and authentication flows
- **Coordinates**: Search requests across all components

**Key Responsibilities**:
- API endpoint management (40+ endpoints)
- Component lifecycle management
- Request/response handling and validation
- Security and authentication enforcement

### 2. **Database Manager**
**Type**: Data Access Layer Entity  
**Purpose**: Unified interface to all MongoDB databases  

**Attributes**:
```python
connections: Dict[str, MongoClient]
databases: Dict[str, Database]
db_configs: Dict[str, Config]
```

**Database Configurations**:
- **academic_library**: Books, journals, projects
- **experts_system**: Experts, certificates
- **research_papers**: Articles, conferences, theses
- **laboratories**: Equipment, materials
- **funding**: Research projects, institutions, funding records
- **aposss**: Users, feedback, interactions, preferences

**Relationships**:
- **Connects to**: 6 MongoDB databases
- **Provides data to**: All system modules requiring database access
- **Manages**: Connection pooling, failover, and health monitoring

### 3. **Query Processor**
**Type**: Orchestration Entity  
**Purpose**: Coordinates the complete query processing pipeline  

**Attributes**:
- LLM processor reference
- Validation rules and schemas
- Quality assurance metrics

**Processing Pipeline**:
1. Query cleaning and normalization
2. LLM processing orchestration
3. Result validation and enhancement
4. Error handling and fallback mechanisms

**Relationships**:
- **Uses**: LLM Processor for query analysis
- **Validates**: Query structure and completeness
- **Provides to**: Search Engine processed query data

---

## Data Entities

### 4. **Academic Resources**

#### 4.1 **Book Entity**
**Database**: academic_library.books  
**Purpose**: Academic books and publications  

**Attributes**:
```json
{
  "_id": "ObjectId",
  "title": "String",
  "author": "String", 
  "category": "String",
  "description": "String",
  "abstract": "String",
  "keywords": ["String"],
  "publication_date": "Date",
  "language": "String",
  "file_url": "String",
  "cover_image_url": "String",
  "uploaded_by": "String",
  "created_at": "DateTime"
}
```

**Relationships**:
- **Referenced by**: Search results and user bookmarks
- **Categorized in**: Academic classification hierarchy
- **Used in**: Knowledge graph as publication nodes

#### 4.2 **Journal Entity**
**Database**: academic_library.journals  

**Attributes**:
```json
{
  "_id": "ObjectId",
  "title": "String",
  "editor": "String",
  "issue_number": "Number",
  "issn": "String",
  "category": "String",
  "metadata": {
    "journal_url": "String",
    "publisher": "String",
    "country": "String",
    "license": "String",
    "review_process": "String",
    "apc_required": "Boolean"
  }
}
```

#### 4.3 **Project Entity**
**Database**: academic_library.projects  

**Attributes**:
```json
{
  "_id": "ObjectId",
  "title": "String",
  "student_name": "String",
  "supervisor": "String",
  "university": "String",
  "department": "String",
  "category": "String",
  "description": "String",
  "abstract": "String",
  "keywords": ["String"]
}
```

### 5. **Research Entities**

#### 5.1 **Article Entity**
**Database**: research_papers.articles  

**Attributes**:
```json
{
  "_id": "ObjectId",
  "title": "String",
  "link": "String",
  "year": "Number",
  "authors": ["String"],
  "citations": "Number",
  "abstract": "String"
}
```

#### 5.2 **Conference Entity**
**Database**: research_papers.conferences  

#### 5.3 **Thesis Entity**
**Database**: research_papers.theses  

**Attributes**:
```json
{
  "_id": "ObjectId",
  "user_id": "ObjectId",
  "title": "String",
  "student_name": "String",
  "supervisor": "String",
  "abstract": "String",
  "university": "String",
  "degree_type": "String",
  "defense_date": "Date",
  "status": "String"
}
```

### 6. **Expert System Entities**

#### 6.1 **Expert Entity**
**Database**: experts_system.experts  

**Attributes**:
```json
{
  "_id": "ObjectId",
  "name": "String",
  "email": "String",
  "phone": "String",
  "general_information": {
    "address": "String",
    "slogan": "String",
    "locations": "String",
    "languages": ["String"]
  },
  "job_complete": {
    "role": "String",
    "count": "Number"
  },
  "awards": ["String"],
  "social_sharing": {
    "facebook": "String",
    "twitter": "String",
    "linkedIn": "String"
  }
}
```

#### 6.2 **Certificate Entity**
**Database**: experts_system.certificates  

**Attributes**:
```json
{
  "_id": "ObjectId",
  "expert_id": "String",
  "title": "String",
  "degree": "String",
  "institution": "String",
  "description": "String",
  "years": "String",
  "status": "String",
  "submitted_at": "DateTime",
  "approved_at": "DateTime"
}
```

**Relationships**:
- **Belongs to**: Expert (via expert_id)
- **Validates**: Expert credentials and qualifications

### 7. **Laboratory Entities**

#### 7.1 **Equipment Entity**
**Database**: laboratories.equipments  

**Attributes**:
```json
{
  "equipment_id": "String",
  "equipment_name": "String",
  "description": "String",
  "model": "String",
  "serial_number": "String",
  "status": "String",
  "last_maintenance": "Date",
  "specifications": "String",
  "images": ["String"],
  "created_at": "Date",
  "updated_at": "Date"
}
```

#### 7.2 **Material Entity**
**Database**: laboratories.materials  

**Attributes**:
```json
{
  "material_id": "String",
  "material_name": "String",
  "description": "String",
  "quantity": "Number",
  "unit": "String",
  "status": "String",
  "expiration_date": "Date",
  "storage_location": "String",
  "safety_notes": "String",
  "supplier": {
    "name": "String",
    "contact_info": "String"
  }
}
```

### 8. **Funding System Entities**

#### 8.1 **Research Project Entity**
**Database**: funding.research_projects  

**Attributes**:
```json
{
  "_id": "ObjectId",
  "researcher_id": "ObjectId",
  "institution_id": "ObjectId", 
  "title": "String",
  "number": "String",
  "status": "String",
  "field_category": "String",
  "field_group": "String",
  "field_area": "String",
  "background": {
    "problem": "String",
    "importance": "String",
    "hypotheses": "String",
    "scope": "String"
  },
  "methodology": {
    "problem_handling": "String",
    "tools": ["String"],
    "alternatives": "String",
    "justification": "String"
  },
  "objectives": ["String"],
  "budget_requested": "Number",
  "funding_sources": ["String"]
}
```

#### 8.2 **Institution Entity**
**Database**: funding.institutions  

#### 8.3 **Funding Record Entity**
**Database**: funding.funding_records  

---

## User Management Entities

### 9. **User Entity**
**Database**: aposss.users  
**Purpose**: Complete user profile and preferences management  

**Attributes**:
```json
{
  "_id": "ObjectId",
  "user_id": "String (UUID)",
  "username": "String",
  "email": "String",
  "password_hash": "Binary",
  "profile": {
    "first_name": "String",
    "last_name": "String", 
    "academic_fields": ["String"],
    "institution": "String",
    "role": "String",
    "bio": "String",
    "avatar_url": "String"
  },
  "preferences": {
    "search_preferences": {
      "preferred_resource_types": ["String"],
      "preferred_databases": ["String"],
      "language_preference": "String",
      "results_per_page": "Number"
    },
    "notification_preferences": {
      "email_notifications": "Boolean",
      "search_alerts": "Boolean",
      "feedback_requests": "Boolean"
    },
    "privacy_settings": {
      "profile_visibility": "String",
      "interaction_tracking": "Boolean",
      "personalized_recommendations": "Boolean"
    }
  },
  "statistics": {
    "total_searches": "Number",
    "total_feedback": "Number",
    "average_rating": "Number",
    "favorite_topics": ["String"],
    "most_used_resources": ["String"]
  },
  "oauth_providers": {
    "google": {
      "provider_id": "String",
      "email": "String",
      "email_verified": "Boolean",
      "picture": "String"
    },
    "orcid": {
      "provider_id": "String",
      "orcid": "String"
    }
  },
  "created_at": "DateTime",
  "updated_at": "DateTime",
  "last_login": "DateTime",
  "is_active": "Boolean",
  "email_verified": "Boolean",
  "login_count": "Number"
}
```

**Relationships**:
- **Has many**: User interactions, feedback entries, preferences, sessions
- **Belongs to**: OAuth providers (optional)
- **Creates**: Search history, bookmarks
- **Participates in**: Feedback system and personalization

### 10. **User Session Entity**
**Database**: aposss.user_sessions  

**Attributes**:
```json
{
  "_id": "ObjectId",
  "user_id": "String",
  "token": "String (JWT)",
  "created_at": "DateTime",
  "expires_at": "DateTime", 
  "is_active": "Boolean"
}
```

### 11. **User Interaction Entity**
**Database**: aposss.user_interactions  

**Attributes**:
```json
{
  "_id": "ObjectId",
  "user_id": "String",
  "action": "String",
  "query": "String",
  "result_id": "String",
  "session_id": "String",
  "timestamp": "DateTime",
  "metadata": {
    "query_length": "Number",
    "query_words": "Number",
    "result_type": "String",
    "rating": "Number"
  }
}
```

### 12. **User Feedback Entity**
**Database**: aposss.user_feedback  

**Attributes**:
```json
{
  "_id": "ObjectId",
  "query_id": "String",
  "result_id": "String",
  "user_id": "String",
  "rating": "Number",
  "feedback_type": "String",
  "comment": "String",
  "user_session": "String",
  "submitted_at": "DateTime",
  "feedback_version": "String"
}
```

### 13. **User Preferences Entity**
**Database**: aposss.user_preferences  

### 14. **User Bookmarks Entity**
**Database**: aposss.user_bookmarks  

### 15. **User Search History Entity**
**Database**: aposss.user_search_history  

---

## Search & Processing Entities

### 16. **LLM Processor**
**Type**: AI Processing Entity  
**Purpose**: Advanced query understanding using Gemini-2.0-flash  

**Attributes**:
- API key and configuration
- Model instance (gemini-2.0-flash-exp)
- Generation parameters (temperature, top_p, etc.)
- Safety settings

**Processing Capabilities**:
- **Language Detection**: Identifies query language with high accuracy
- **Translation**: Translates non-English queries to English
- **Spelling Correction**: Fixes typos and grammatical errors
- **Intent Detection**: Categorizes user intent (find_research, find_expert, etc.)
- **Entity Extraction**: Identifies people, organizations, technologies, concepts
- **Keyword Extraction**: Extracts primary, secondary, and technical keywords
- **Synonym Generation**: Generates related terms and synonyms
- **Academic Field Mapping**: Maps queries to relevant academic disciplines

**Output Structure**:
```json
{
  "language_analysis": {
    "detected_language": "String",
    "language_name": "String", 
    "confidence_score": "Number",
    "is_english": "Boolean",
    "script_type": "String"
  },
  "query_processing": {
    "original_query": "String",
    "corrected_original": "String",
    "english_translation": "String",
    "translation_needed": "Boolean"
  },
  "intent_analysis": {
    "primary_intent": "String",
    "secondary_intents": ["String"],
    "confidence": "Number"
  },
  "entity_extraction": {
    "people": ["String"],
    "organizations": ["String"],
    "technologies": ["String"],
    "concepts": ["String"]
  },
  "keyword_analysis": {
    "primary_keywords": ["String"],
    "secondary_keywords": ["String"],
    "technical_terms": ["String"]
  },
  "semantic_expansion": {
    "synonyms": ["String"],
    "related_terms": ["String"],
    "broader_terms": ["String"]
  },
  "academic_classification": {
    "primary_field": "String",
    "secondary_fields": ["String"],
    "specializations": ["String"]
  }
}
```

### 17. **Search Engine**
**Type**: Multi-Database Search Entity  
**Purpose**: Performs hybrid search across all databases  

**Search Strategies**:
- **Semantic Search**: Uses pre-built FAISS index for semantic similarity
- **Keyword Search**: Traditional MongoDB text search and regex matching
- **Result Merging**: Combines and deduplicates results from both approaches

**Database Search Methods**:
- `_search_academic_library()`: Books, journals, projects
- `_search_experts_system()`: Experts, certificates  
- `_search_research_papers()`: Articles, conferences, theses
- `_search_laboratories()`: Equipment, materials
- `_search_funding_system()`: Research projects, institutions

**Result Standardization**:
```json
{
  "id": "String",
  "type": "String",
  "database": "String", 
  "collection": "String",
  "title": "String",
  "description": "String",
  "author": "String",
  "snippet": "String",
  "metadata": {}
}
```

---

## Ranking & Intelligence Entities

### 18. **Ranking Engine**
**Type**: Multi-Algorithm Ranking Entity  
**Purpose**: Orchestrates multiple ranking algorithms for optimal result ordering  

**Ranking Algorithms**:

#### 18.1 **Heuristic Ranking (20% weight)**
- Keyword matching in titles vs. descriptions
- Exact phrase matching bonuses
- Field-specific scoring (title > abstract > content)
- Recent publication bonuses
- Availability status bonuses

#### 18.2 **TF-IDF Similarity (20% weight)**
- Uses scikit-learn TfidfVectorizer
- Cosine similarity between query and document vectors
- N-gram analysis (1-2 grams)

#### 18.3 **Intent Alignment (20% weight)**
- Resource type preference based on detected intent
- Academic field alignment
- Intent confidence weighting

#### 18.4 **Embedding Similarity (20% weight)**
- Real-time semantic similarity via EmbeddingRanker
- Model: `paraphrase-multilingual-MiniLM-L12-v2`
- 384-dimensional embeddings

#### 18.5 **Personalization Scoring (20% weight)**
- User type preferences from interaction history
- Field/category preferences
- Author preferences
- Recency and complexity preferences

**Ranking Modes**:
- **Traditional**: Weighted combination of algorithms (a-e)
- **LTR Only**: 100% Learning-to-Rank (XGBoost)
- **Hybrid**: 70% LTR + 30% traditional (default)

### 19. **Learning-to-Rank (LTR) Ranker**
**Type**: Machine Learning Entity  
**Purpose**: Advanced ML-based ranking using XGBoost  

**Feature Engineering** (50+ features):
- **Current Scores**: Heuristic, TF-IDF, intent, embedding scores
- **BM25 Features**: BM25 scores for title and description  
- **Textual Features**: N-gram matches, term proximity, exact matches
- **Metadata Features**: Publication date, availability, authority metrics
- **LLM Features**: Intent confidence, field matching, entity alignment
- **User Features**: Click-through rates, feedback scores
- **Graph Features**: PageRank, authority scores, connection strength

**Model Configuration**:
- **Algorithm**: XGBoost with pairwise ranking objective
- **Training**: Uses historical user feedback as relevance labels
- **Evaluation**: NDCG (Normalized Discounted Cumulative Gain)
- **Storage**: Model persistence in `ltr_models/` directory

### 20. **Embedding Ranker**
**Type**: Semantic Similarity Entity  
**Purpose**: Provides semantic similarity calculations using sentence transformers  

**Components**:
- **Pre-built Index**: FAISS index of pre-computed document embeddings
- **Real-time Similarity**: On-demand embedding calculation for new queries
- **Caching**: Intelligent caching of embeddings for performance
- **Multilingual Support**: Handles multiple languages

**Model Details**:
- **Model**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Dimensions**: 384
- **Index Type**: FAISS IndexFlatIP (Inner Product)
- **Similarity Metric**: Cosine similarity

### 21. **LLM Ranker**
**Type**: AI Re-ranking Entity  
**Purpose**: Advanced semantic understanding and quality assessment  

**Capabilities**:
- Result re-ranking using LLM analysis
- Relevance scoring with explanations
- Intent matching analysis
- Key aspects identification
- Quality assessment

### 22. **Knowledge Graph**
**Type**: Graph Analytics Entity  
**Purpose**: Models relationships between academic entities using graph theory  

**Graph Structure**:
- **Nodes**: Papers, authors, keywords, equipment, projects, institutions
- **Edges**: Citations, authorship, keyword associations, collaborations
- **Type**: NetworkX DiGraph (directed graph)

**Analytics**:
- **PageRank Scoring**: Authority scores for all nodes
- **Path Analysis**: Shortest paths between related entities
- **Connection Strength**: Relationship strength measurements
- **Authority Calculation**: Comprehensive authority scores

**Features for Ranking**:
- Authority scores for results
- Connection strength to query terms
- Graph-based features for LTR model

### 23. **Enhanced Text Features**
**Type**: Advanced Feature Extraction Entity  
**Purpose**: Sophisticated textual analysis for improved ranking  

**Feature Types**:
- **BM25 Scores**: Advanced BM25 calculations with normalization
- **N-gram Features**: 1-gram, 2-gram, 3-gram overlap analysis
- **Proximity Features**: Query term proximity calculations
- **Complexity Features**: Text complexity metrics and user expertise matching

**Libraries Used**:
- NLTK for tokenization and text processing
- rank-bm25 for BM25 calculations
- textstat for readability metrics

---

## Database Entities

### 24. **MongoDB Database Schema**

#### 24.1 **Academic Library Database**
**Collections**: books, journals, projects  
**Purpose**: Academic publications and student work  

#### 24.2 **Experts System Database**  
**Collections**: experts, certificates  
**Purpose**: Expert profiles and qualifications  

#### 24.3 **Research Papers Database**
**Collections**: articles, conferences, theses  
**Purpose**: Research publications and academic papers  

#### 24.4 **Laboratories Database**
**Collections**: equipments, materials  
**Purpose**: Laboratory resources and materials  

#### 24.5 **Funding Database**
**Collections**: research_projects, institutions, funding_records  
**Purpose**: Research funding and institutional data  

#### 24.6 **APOSSS Database**
**Collections**: users, user_feedback, user_interactions, user_preferences, user_sessions, user_bookmarks, user_search_history, verification_codes  
**Purpose**: System users and application data  

### 25. **Index Management**
**Pre-built Indexes**:
- **FAISS Index**: Semantic embeddings for fast similarity search
- **MongoDB Indexes**: Text indexes on searchable fields
- **User Indexes**: Performance indexes on user collections

---

## Frontend Entities

### 26. **User Interface Components**

#### 26.1 **Main Application Pages**
- **index.html**: Homepage with search interface and LLM analysis display
- **results.html**: Advanced results display with filtering, sorting, and categorization
- **home.html**: Dashboard with search and analytics
- **login.html**: User authentication interface
- **signup.html**: User registration interface
- **profile.html**: User profile management
- **user_dashboard.html**: Personalized user dashboard

#### 26.2 **Frontend Data Structures**

**Search State Management**:
```javascript
let allResults = [];
let filteredResults = [];
let currentView = 'list'; // or 'grid'
let currentSort = 'relevance';
let activeFilters = {
    databases: new Set(),
    types: new Set(), 
    relevance: new Set()
};
let displayMode = 'sections'; // or 'all'
```

**User Interface Elements**:
- Search input and filters
- Result display components
- Feedback submission interfaces
- User profile forms
- Authentication flows
- Analytics dashboards

#### 26.3 **Interactive Features**
- **Real-time Search**: Instant search with debouncing
- **Advanced Filtering**: Multi-criteria filtering system
- **Result Categorization**: High/Medium/Low relevance grouping
- **Bookmarking System**: Save and manage favorite results
- **Feedback Collection**: Rating and comment system
- **Responsive Design**: Mobile and desktop optimization

---

## Configuration & Infrastructure Entities

### 27. **Environment Configuration**
**File**: config.env.example  

**Configuration Categories**:
```env
# AI/LLM Configuration
GEMINI_API_KEY=<api_key>

# Database Configuration  
MONGODB_URI_ACADEMIC_LIBRARY=<connection_string>
MONGODB_URI_EXPERTS_SYSTEM=<connection_string>
MONGODB_URI_RESEARCH_PAPERS=<connection_string>
MONGODB_URI_LABORATORIES=<connection_string>
MONGODB_URI_FUNDING=<connection_string>
MONGODB_URI_APOSSS=<connection_string>

# OAuth Configuration
GOOGLE_CLIENT_ID=<client_id>
GOOGLE_CLIENT_SECRET=<client_secret>
ORCID_CLIENT_ID=<client_id>
ORCID_CLIENT_SECRET=<client_secret>

# Application Configuration
BASE_URL=http://localhost:5000
FLASK_ENV=development
```

### 28. **OAuth Manager**
**Type**: Authentication Entity  
**Purpose**: Manages social authentication flows  

**Supported Providers**:
- **Google OAuth**: Full profile and email access
- **ORCID OAuth**: Academic researcher authentication

**OAuth Flow Management**:
- Authorization URL generation
- Token exchange handling
- User info retrieval
- Account linking capabilities

### 29. **Feedback System**
**Type**: Learning System Entity  
**Purpose**: Collects and analyzes user feedback for system improvement  

**Feedback Types**:
- **Rating Feedback**: 1-5 star ratings
- **Binary Feedback**: Thumbs up/down
- **Detailed Feedback**: Text comments and suggestions

**Storage Strategy**:
- **Primary**: MongoDB collection in APOSSS database
- **Fallback**: JSON Lines file for reliability
- **Analytics**: Real-time statistics and reporting

---

## Workflow & Process Entities

### 30. **Search Workflow Process**

**Complete Search Pipeline**:
1. **User Input** → Frontend UI
2. **Query Validation** → Query Processor
3. **LLM Analysis** → LLM Processor (Gemini)
4. **Multi-Database Search** → Search Engine
5. **Result Aggregation** → Search Engine
6. **Ranking Application** → Ranking Engine
7. **Personalization** → User Manager
8. **Result Delivery** → Frontend UI
9. **User Interaction** → Feedback System
10. **Learning Loop** → LTR Model Training

### 31. **Authentication Workflow**

**User Registration Flow**:
1. User provides registration data
2. Validation and uniqueness checks
3. Password hashing (bcrypt)
4. Profile creation with defaults
5. Preferences initialization
6. Email verification (optional)
7. JWT token generation

**Social Authentication Flow**:
1. OAuth provider authorization
2. Token exchange and validation
3. User info retrieval
4. Account creation or linking
5. Profile enhancement
6. Session establishment

### 32. **Feedback Learning Process**

**Continuous Improvement Cycle**:
1. **Collection**: User interactions and explicit feedback
2. **Storage**: MongoDB and file-based backup
3. **Analysis**: Statistical analysis and pattern recognition
4. **Feature Generation**: Convert feedback to LTR training data
5. **Model Training**: XGBoost model retraining
6. **Deployment**: Updated model integration
7. **Evaluation**: Performance measurement and validation

---

## Entity Relationships Matrix

### Primary Relationships

| Entity | Depends On | Provides To | Manages |
|--------|------------|-------------|---------|
| **APOSSS App** | All Modules | Frontend API | Request Routing |
| **Database Manager** | MongoDB | All Modules | Data Access |
| **User Manager** | Database Manager | Authentication | User Lifecycle |
| **LLM Processor** | Gemini API | Query Processor | Query Understanding |
| **Search Engine** | Database Manager, Query Processor | Ranking Engine | Multi-DB Search |
| **Ranking Engine** | All Rankers, Search Results | Frontend | Result Ordering |
| **LTR Ranker** | Feedback Data, Features | Ranking Engine | ML-based Ranking |
| **Embedding Ranker** | Sentence Transformers | Ranking Engine | Semantic Similarity |
| **Knowledge Graph** | Search Results | Ranking Features | Entity Relationships |
| **Feedback System** | User Interactions | LTR Training | Learning Data |

### Data Flow Relationships

| Source | Target | Data Type | Purpose |
|--------|--------|-----------|---------|
| **Frontend** | **API Endpoints** | HTTP Requests | User Actions |
| **Query Processor** | **LLM Processor** | Raw Query | Understanding |
| **LLM Processor** | **Search Engine** | Processed Query | Search Execution |
| **Search Engine** | **Ranking Engine** | Raw Results | Ranking |
| **Ranking Engine** | **Frontend** | Ranked Results | Display |
| **User Actions** | **Feedback System** | Interaction Data | Learning |
| **Feedback System** | **LTR Ranker** | Training Data | Model Improvement |

### Aggregation Relationships

| Parent | Children | Relationship Type |
|--------|----------|-------------------|
| **APOSSS Database** | Users, Feedback, Interactions, Preferences | One-to-Many |
| **User** | Sessions, Interactions, Bookmarks, History | One-to-Many |
| **Search Query** | Search Results, Feedback Entries | One-to-Many |
| **Expert** | Certificates | One-to-Many |
| **Research Project** | Funding Records | One-to-Many |
| **Knowledge Graph** | Nodes, Edges, Relationships | One-to-Many |

### Composition Relationships

| Composite | Components | Purpose |
|-----------|------------|---------|
| **Search Results** | Individual Result Objects | Result Set |
| **User Profile** | Personal Info, Preferences, Statistics | Complete User |
| **LLM Analysis** | Language, Intent, Entities, Keywords | Query Understanding |
| **Ranking Score** | Multiple Algorithm Scores | Final Relevance |
| **Feedback Entry** | Rating, Comments, Metadata | User Input |

---

## Conclusion

The APOSSS system represents a sophisticated integration of modern AI technologies, advanced search algorithms, and user-centric design. The entity architecture supports:

- **Scalability**: Modular design allows independent scaling of components
- **Intelligence**: Multi-layered AI integration from query understanding to result ranking
- **Personalization**: Comprehensive user modeling and preference learning
- **Adaptability**: Feedback-driven continuous improvement
- **Reliability**: Robust error handling and fallback mechanisms
- **Security**: Comprehensive authentication and authorization
- **Performance**: Optimized algorithms and caching strategies

This entity framework provides the foundation for a next-generation academic search system that combines the power of large language models, machine learning, and traditional information retrieval techniques to deliver highly relevant and personalized search experiences.