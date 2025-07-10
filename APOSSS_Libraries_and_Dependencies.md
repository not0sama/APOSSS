# APOSSS Libraries and Dependencies
## AI-Powered Open-Science Semantic Search System

### Table of Contents
1. [Core Dependencies](#core-dependencies)
2. [AI/ML Libraries](#aiml-libraries)
3. [Database Libraries](#database-libraries)
4. [Web Framework Libraries](#web-framework-libraries)
5. [Authentication & Security Libraries](#authentication--security-libraries)
6. [Data Processing Libraries](#data-processing-libraries)
7. [Text Processing & NLP Libraries](#text-processing--nlp-libraries)
8. [Utility Libraries](#utility-libraries)
9. [Development & Testing Dependencies](#development--testing-dependencies)
10. [System Dependencies](#system-dependencies)

---

## Core Dependencies

### Python Standard Library
- **`os`** - Operating system interface for environment variables and file operations
- **`logging`** - Comprehensive logging system for debugging and monitoring
- **`json`** - JSON encoder/decoder for data serialization
- **`re`** - Regular expression operations for text pattern matching
- **`uuid`** - UUID generation for unique identifiers
- **`datetime`** - Date and time manipulation
- **`hashlib`** - Secure hash and message digest algorithms
- **`pickle`** - Python object serialization for caching
- **`secrets`** - Cryptographically strong random number generation
- **`smtplib`** - SMTP client for email sending
- **`email.mime`** - Email message construction utilities
- **`urllib.parse`** - URL parsing utilities
- **`functools`** - Higher-order functions and operations on callable objects
- **`collections`** - Specialized container datatypes (defaultdict, Counter)
- **`math`** - Mathematical functions
- **`random`** - Random number generation
- **`typing`** - Type hints for better code documentation

---

## AI/ML Libraries

### Google Generative AI
- **`google-generativeai==0.5.4`**
  - **Purpose**: Integration with Google's Gemini-2.0-flash Large Language Model
  - **Usage**: Query understanding, intent detection, entity extraction, language detection
  - **Features**: Advanced natural language processing, multilingual support, structured output generation
  - **Key Capabilities**: Query translation, spelling correction, semantic analysis

### Sentence Transformers
- **`sentence-transformers==2.7.0`**
  - **Purpose**: Generate semantic embeddings for text similarity
  - **Model Used**: `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions)
  - **Usage**: Semantic search, document similarity, cross-lingual understanding
  - **Features**: Multilingual support, fast inference, high-quality embeddings

### FAISS (Facebook AI Similarity Search)
- **`faiss-cpu==1.8.0`**
  - **Purpose**: Efficient similarity search and clustering of dense vectors
  - **Usage**: Vector database for semantic search, nearest neighbor search
  - **Features**: IndexFlatIP for cosine similarity, scalable to millions of vectors
  - **Performance**: Sub-second search across 100K+ documents

### XGBoost
- **`xgboost>=3.0.0`**
  - **Purpose**: Gradient boosting framework for Learning-to-Rank (LTR)
  - **Usage**: Machine learning-based result ranking using 50+ features
  - **Features**: Pairwise ranking objective, NDCG optimization, feature importance analysis
  - **Algorithm**: Gradient Boosted Decision Trees

### Scikit-learn
- **`scikit-learn>=1.6.0`**
  - **Purpose**: Machine learning utilities and algorithms
  - **Usage**: TF-IDF vectorization, cosine similarity, train-test splitting, metrics
  - **Features**: Text vectorization, similarity calculations, model evaluation
  - **Components**: TfidfVectorizer, cosine_similarity, train_test_split, ndcg_score

### PyTorch
- **`torch==2.1.0`**
  - **Purpose**: Deep learning framework (dependency for sentence-transformers)
  - **Usage**: Neural network computations for transformer models
  - **Features**: GPU acceleration support, automatic differentiation

### NumPy
- **`numpy>=2.2.0`**
  - **Purpose**: Fundamental package for scientific computing
  - **Usage**: Array operations, mathematical computations, embedding manipulations
  - **Features**: N-dimensional arrays, broadcasting, linear algebra operations

### Pandas
- **`pandas>=2.3.0`**
  - **Purpose**: Data manipulation and analysis library
  - **Usage**: Feature engineering for LTR, data processing, structured data handling
  - **Features**: DataFrame operations, data cleaning, statistical analysis

### SciPy
- **`scipy>=1.15.0`**
  - **Purpose**: Scientific computing library
  - **Usage**: Statistical functions, correlation analysis, advanced mathematical operations
  - **Features**: Statistical functions (pearsonr), optimization algorithms

### NetworkX
- **`networkx`**
  - **Purpose**: Graph theory and network analysis
  - **Usage**: Knowledge graph construction, PageRank calculation, network analysis
  - **Features**: DiGraph support, shortest path algorithms, centrality measures

---

## Database Libraries

### PyMongo
- **`pymongo==4.4.1`**
  - **Purpose**: MongoDB driver for Python
  - **Usage**: Database operations across 6 MongoDB databases
  - **Features**: Connection management, CRUD operations, indexing
  - **Databases**: Academic Library, Experts System, Research Papers, Laboratories, Funding, APOSSS

---

## Web Framework Libraries

### Flask
- **`flask==2.3.2`**
  - **Purpose**: Lightweight WSGI web application framework
  - **Usage**: REST API server, route handling, request processing
  - **Features**: 40+ API endpoints, template rendering, session management
  - **Architecture**: Microframework with modular design

### Flask-CORS
- **`flask-cors==4.0.0`**
  - **Purpose**: Cross-Origin Resource Sharing (CORS) support
  - **Usage**: Enable cross-origin requests from frontend applications
  - **Features**: Configurable CORS policies, preflight handling

---

## Authentication & Security Libraries

### JSON Web Tokens (JWT)
- **`pyjwt==2.8.0`**
  - **Purpose**: JSON Web Token implementation
  - **Usage**: User authentication, session management, token-based security
  - **Features**: Token encoding/decoding, expiration handling, signature verification

### bcrypt
- **`bcrypt==4.0.1`**
  - **Purpose**: Password hashing library
  - **Usage**: Secure password storage, authentication verification
  - **Features**: Salt generation, adaptive hashing, timing attack resistance

### Authlib
- **`authlib==1.2.1`**
  - **Purpose**: OAuth and OpenID Connect library
  - **Usage**: OAuth 2.0 implementation for social authentication
  - **Features**: OAuth flows, token management, provider integrations

### Requests-OAuthlib
- **`requests-oauthlib==1.3.1`**
  - **Purpose**: OAuth library for requests
  - **Usage**: OAuth authentication flows with external providers
  - **Features**: Google OAuth, ORCID integration, secure token exchange

### Requests
- **`requests`**
  - **Purpose**: HTTP library for Python
  - **Usage**: API calls to OAuth providers, external service integration
  - **Features**: Session management, SSL verification, response handling

---

## Data Processing Libraries

### Joblib
- **`joblib>=1.5.0`**
  - **Purpose**: Efficient serialization and parallel computing
  - **Usage**: Model persistence, caching, parallel processing
  - **Features**: Efficient numpy array serialization, memory mapping

### Matplotlib
- **`matplotlib>=3.10.0`**
  - **Purpose**: Plotting and visualization library
  - **Usage**: Data visualization, performance metrics plotting, analytics dashboards
  - **Features**: Statistical plots, customizable charts, export capabilities

---

## Text Processing & NLP Libraries

### NLTK (Natural Language Toolkit)
- **`nltk==3.8.1`**
  - **Purpose**: Natural language processing library
  - **Usage**: Text tokenization, n-gram analysis, linguistic analysis
  - **Features**: Word tokenization, POS tagging, corpus access
  - **Components**: `word_tokenize`, `ngrams`, `punkt` tokenizer

### Rank-BM25
- **`rank_bm25==0.2.2`**
  - **Purpose**: BM25 ranking algorithm implementation
  - **Usage**: Traditional information retrieval scoring, feature engineering
  - **Features**: Okapi BM25, document scoring, relevance ranking
  - **Algorithm**: Best Matching 25 with configurable parameters

### TextStat
- **`textstat==0.7.3`**
  - **Purpose**: Text readability and complexity analysis
  - **Usage**: Content complexity assessment, readability scoring
  - **Features**: Flesch Reading Ease, SMOG Index, Coleman-Liau Index
  - **Metrics**: Automated Readability Index, complexity matching

---

## Utility Libraries

### Python-dotenv
- **`python-dotenv==1.0.0`**
  - **Purpose**: Environment variable management
  - **Usage**: Configuration management, secret handling, environment setup
  - **Features**: .env file loading, development/production configs

---

## Development & Testing Dependencies

### Build Index Script
- **`build_index.py`**
  - **Purpose**: Pre-compute FAISS indices for production deployment
  - **Usage**: Index building, embedding caching, performance optimization
  - **Features**: Batch processing, progress tracking, production optimization

### Testing Modules
- **`tests/`** directory contains:
  - **`test_api_integration.py`** - API endpoint testing
  - **`test_complete_system.py`** - End-to-end system testing
  - **`test_ltr_system.py`** - Learning-to-rank testing
  - **`test_realtime_similarity.py`** - Embedding system testing
  - **`test_multilingual_support.py`** - Language support testing

---

## System Dependencies

### Environment Variables Required
- **`GEMINI_API_KEY`** - Google Gemini LLM API key
- **`MONGODB_URI_*`** - MongoDB connection strings for 6 databases
- **`JWT_SECRET_KEY`** - Secret key for JWT token signing
- **`GOOGLE_CLIENT_ID/SECRET`** - Google OAuth credentials
- **`ORCID_CLIENT_ID/SECRET`** - ORCID OAuth credentials
- **`SMTP_*`** - Email server configuration

### Database Requirements
- **MongoDB 4.0+** - Document database for data storage
- **6 MongoDB Databases**:
  - `Academic_Library` - Books, journals, projects
  - `Experts_System` - Expert profiles, certificates
  - `Research_Papers` - Articles, conferences, theses
  - `Laboratories` - Equipment, materials
  - `Funding` - Research projects, institutions, funding records
  - `APOSSS` - Users, feedback, interactions, preferences

### Hardware Requirements
- **8GB+ RAM** - For embedding models and FAISS indices
- **Storage**: Variable based on database size and embedding cache
- **CPU**: Multi-core recommended for parallel processing

---

## Dependency Management

### Requirements File
```
flask==2.3.2
flask-cors==4.0.0
python-dotenv==1.0.0
pymongo==4.4.1
google-generativeai==0.5.4
scikit-learn>=1.6.0
numpy>=2.2.0
sentence-transformers==2.7.0
faiss-cpu==1.8.0
torch==2.1.0
xgboost>=3.0.0
pandas>=2.3.0
scipy>=1.15.0
joblib>=1.5.0
matplotlib>=3.10.0
rank_bm25==0.2.2
nltk==3.8.1
textstat==0.7.3
authlib==1.2.1
requests-oauthlib==1.3.1
bcrypt==4.0.1
pyjwt==2.8.0
```

### Installation Commands
```bash
# Create virtual environment
python -m venv APOSSSenv
source APOSSSenv/bin/activate  # Linux/Mac
# or APOSSSenv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"

# Pre-download sentence transformer model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"
```

---

## Architecture Integration

### Technology Stack Overview
```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Vanilla JS + Tailwind CSS)     │
├─────────────────────────────────────────────────────────────┤
│                    Flask REST API (Python)                  │
├─────────────────────────────────────────────────────────────┤
│  AI/ML Layer: Gemini LLM + Sentence Transformers + XGBoost │
├─────────────────────────────────────────────────────────────┤
│    Search Layer: FAISS + BM25 + TF-IDF + Knowledge Graph   │
├─────────────────────────────────────────────────────────────┤
│              Data Layer: MongoDB (6 Databases)              │
└─────────────────────────────────────────────────────────────┘
```

### Dependency Categories by Function
- **AI/ML Processing**: 8 libraries (Gemini, Transformers, XGBoost, etc.)
- **Web Framework**: 2 libraries (Flask, Flask-CORS)
- **Database**: 1 library (PyMongo)
- **Security**: 4 libraries (JWT, bcrypt, Authlib, OAuth)
- **Text Processing**: 3 libraries (NLTK, BM25, TextStat)
- **Scientific Computing**: 6 libraries (NumPy, Pandas, SciPy, etc.)
- **Utilities**: 1 library (python-dotenv)

### Performance Considerations
- **Memory Usage**: ~2-3GB for embedding models and indices
- **Startup Time**: ~30-60 seconds for model loading
- **Response Time**: <1 second for most search operations
- **Scalability**: Horizontal scaling supported through load balancing

This comprehensive dependency analysis demonstrates APOSSS as a sophisticated AI-powered system leveraging cutting-edge technologies across multiple domains including machine learning, natural language processing, graph theory, and modern web development practices. 