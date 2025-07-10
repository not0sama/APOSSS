# APOSSS API Documentation
## AI-Powered Open-Science Semantic Search System

### Table of Contents
1. [Authentication](#authentication)
2. [User Management](#user-management)
3. [Search & Core Functionality](#search--core-functionality)
4. [AI & Machine Learning](#ai--machine-learning)
5. [User Data Management](#user-data-management)
6. [Funding System](#funding-system)
7. [System & Utility](#system--utility)
8. [Frontend Routes](#frontend-routes)
9. [Request/Response Examples](#requestresponse-examples)

---

## Overview

The APOSSS API provides comprehensive functionality for an AI-powered academic search system. The API supports:
- **Multi-database semantic search** across 6 MongoDB databases
- **AI-powered ranking** with multiple algorithms (LTR, embeddings, heuristic)
- **User authentication** with OAuth support (Google, ORCID)
- **Personalization** through user preferences and interaction tracking
- **Feedback system** for continuous improvement
- **Machine learning** capabilities for learning-to-rank

**Base URL**: `http://localhost:5000` (development)  
**Authentication**: Bearer token (JWT) for protected endpoints  
**Content-Type**: `application/json` for all API requests

---

## Authentication

### POST `/api/auth/register`
**Purpose**: Register a new user account  
**Authentication**: None required  
**Request Body**:
```json
{
  "username": "string",
  "email": "string", 
  "password": "string",
  "first_name": "string",
  "last_name": "string",
  "academic_fields": ["string"],
  "institution": "string",
  "role": "string"
}
```
**Response**: User object with JWT token
**Status Codes**: 201 (Created), 400 (Validation Error), 500 (Server Error)

### POST `/api/auth/login`
**Purpose**: Authenticate user with email/password  
**Authentication**: None required  
**Request Body**:
```json
{
  "email": "string",
  "password": "string"
}
```
**Response**: User object with JWT token
**Status Codes**: 200 (Success), 400 (Invalid Credentials), 500 (Server Error)

### POST `/api/auth/verify`
**Purpose**: Verify JWT token validity  
**Authentication**: Bearer token required  
**Request Body**:
```json
{
  "token": "string"
}
```
**Response**: Token verification status and user info
**Status Codes**: 200 (Valid), 401 (Invalid), 500 (Server Error)

### GET `/api/auth/google`
**Purpose**: Initiate Google OAuth authentication flow  
**Authentication**: None required  
**Response**: Authorization URL for Google OAuth
**Status Codes**: 200 (Success), 500 (OAuth Not Configured)

### GET `/api/auth/google/callback`
**Purpose**: Handle Google OAuth callback with authorization code  
**Authentication**: None required  
**Query Parameters**: `code`, `state`, `error`  
**Response**: HTML with JavaScript postMessage for OAuth completion
**Status Codes**: 200 (Success), 400 (OAuth Error)

### GET `/api/auth/orcid`
**Purpose**: Initiate ORCID OAuth authentication flow  
**Authentication**: None required  
**Response**: Authorization URL for ORCID OAuth
**Status Codes**: 200 (Success), 500 (OAuth Not Configured)

### GET `/api/auth/orcid/callback`
**Purpose**: Handle ORCID OAuth callback with authorization code  
**Authentication**: None required  
**Query Parameters**: `code`, `state`, `error`  
**Response**: HTML with JavaScript postMessage for OAuth completion
**Status Codes**: 200 (Success), 400 (OAuth Error)

### POST `/api/auth/check-email`
**Purpose**: Check if email address is already registered  
**Authentication**: None required  
**Request Body**:
```json
{
  "email": "string"
}
```
**Response**: Email existence status
**Status Codes**: 200 (Success), 400 (Missing Email), 500 (Server Error)

### POST `/api/auth/check-username`
**Purpose**: Check if username is already taken  
**Authentication**: None required  
**Request Body**:
```json
{
  "username": "string"
}
```
**Response**: Username availability status
**Status Codes**: 200 (Success), 400 (Missing Username), 500 (Server Error)

### POST `/api/auth/forgot-password`
**Purpose**: Initiate password reset process  
**Authentication**: None required  
**Request Body**:
```json
{
  "email": "string"
}
```
**Response**: Password reset status
**Status Codes**: 200 (Success), 400 (Invalid Email), 500 (Server Error)

### POST `/api/auth/reset-password`
**Purpose**: Complete password reset with verification code  
**Authentication**: None required  
**Request Body**:
```json
{
  "email": "string",
  "verification_code": "string",
  "new_password": "string"
}
```
**Response**: Password reset completion status
**Status Codes**: 200 (Success), 400 (Invalid Code), 500 (Server Error)

---

## User Management

### GET `/api/user/profile`
**Purpose**: Get current user's profile information  
**Authentication**: Bearer token required  
**Response**: Complete user profile data including preferences and statistics
**Status Codes**: 200 (Success), 401 (Unauthorized), 500 (Server Error)

### PUT `/api/user/profile`
**Purpose**: Update user profile information  
**Authentication**: Bearer token required  
**Request Body**:
```json
{
  "first_name": "string",
  "last_name": "string",
  "academic_fields": ["string"],
  "institution": "string",
  "role": "string",
  "bio": "string"
}
```
**Response**: Updated user profile
**Status Codes**: 200 (Success), 401 (Unauthorized), 500 (Server Error)

### POST `/api/user/change-password`
**Purpose**: Change user's password  
**Authentication**: Bearer token required  
**Request Body**:
```json
{
  "current_password": "string",
  "new_password": "string"
}
```
**Response**: Password change status
**Status Codes**: 200 (Success), 400 (Invalid Password), 401 (Unauthorized)

### GET `/api/user/statistics`
**Purpose**: Get user's activity statistics and analytics  
**Authentication**: Bearer token required  
**Response**: Comprehensive user statistics including search patterns, feedback activity, and preferences
**Status Codes**: 200 (Success), 401 (Unauthorized), 500 (Server Error)

### POST `/api/user/profile-picture`
**Purpose**: Upload user profile picture  
**Authentication**: Bearer token required  
**Content-Type**: `multipart/form-data`  
**Request Body**: File upload with image file
**Response**: Profile picture upload status and URL
**Status Codes**: 200 (Success), 400 (Invalid File), 401 (Unauthorized)

### GET `/api/user/profile-picture/<user_id>`
**Purpose**: Get user's profile picture  
**Authentication**: None required  
**Path Parameters**: `user_id` - User identifier  
**Response**: Profile picture file or default avatar
**Status Codes**: 200 (Success), 404 (Not Found)

### DELETE `/api/user/profile-picture`
**Purpose**: Delete user's profile picture  
**Authentication**: Bearer token required  
**Response**: Deletion status
**Status Codes**: 200 (Success), 401 (Unauthorized), 500 (Server Error)

### POST `/api/user/send-verification-code`
**Purpose**: Send email verification code to user  
**Authentication**: Bearer token required  
**Response**: Verification code sending status
**Status Codes**: 200 (Success), 401 (Unauthorized), 500 (Server Error)

### POST `/api/user/verify-email`
**Purpose**: Verify user's email with verification code  
**Authentication**: Bearer token required  
**Request Body**:
```json
{
  "verification_code": "string"
}
```
**Response**: Email verification status
**Status Codes**: 200 (Success), 400 (Invalid Code), 401 (Unauthorized)

### GET `/api/user/preferences`
**Purpose**: Get user's search and notification preferences  
**Authentication**: Bearer token required  
**Response**: User preferences including search settings, ranking weights, and notification settings
**Status Codes**: 200 (Success), 401 (Unauthorized), 500 (Server Error)

### POST `/api/user/preferences`
**Purpose**: Update user's preferences  
**Authentication**: Bearer token required  
**Request Body**:
```json
{
  "search_preferences": {
    "preferred_resource_types": ["string"],
    "preferred_databases": ["string"],
    "language_preference": "string",
    "results_per_page": "integer"
  },
  "ranking_preferences": {
    "weight_recency": "float",
    "weight_relevance": "float", 
    "weight_authority": "float",
    "weight_user_feedback": "float"
  },
  "notification_preferences": {
    "email_notifications": "boolean",
    "search_alerts": "boolean",
    "feedback_requests": "boolean"
  }
}
```
**Response**: Updated preferences
**Status Codes**: 200 (Success), 401 (Unauthorized), 500 (Server Error)

---

## Search & Core Functionality

### POST `/api/search`
**Purpose**: Main search endpoint with AI-powered ranking across multiple databases  
**Authentication**: Optional (Bearer token for personalization)  
**Request Body**:
```json
{
  "query": "string",
  "ranking_mode": "hybrid|ltr_only|traditional",
  "database_filters": ["academic_library", "research_papers", "experts_system", "laboratories", "funding"],
  "session_id": "string",
  "user_id": "string"
}
```
**Response**: Comprehensive search results with ranking scores, query analysis, and categorized results
**Status Codes**: 200 (Success), 400 (Missing Query), 500 (Server Error)

**Key Features**:
- **Multi-database search** across 6 MongoDB databases
- **AI query processing** with Gemini LLM for intent detection and entity extraction
- **Multiple ranking modes**: Traditional weighted, LTR-only, or hybrid approach
- **Personalization** based on user history and preferences
- **Real-time semantic similarity** using sentence transformers
- **Feedback integration** for continuous learning

### POST `/api/feedback`
**Purpose**: Submit user feedback for search result relevance  
**Authentication**: Optional (tracks anonymous feedback)  
**Request Body**:
```json
{
  "query_id": "string",
  "result_id": "string", 
  "rating": "integer (1-5)",
  "feedback_type": "rating|thumbs_up|thumbs_down",
  "user_session": "string",
  "additional_data": "object"
}
```
**Response**: Feedback submission confirmation
**Status Codes**: 200 (Success), 400 (Missing Fields), 500 (Server Error)

### GET `/api/feedback/stats`
**Purpose**: Get aggregated feedback statistics  
**Authentication**: None required  
**Response**: System-wide feedback statistics and metrics
**Status Codes**: 200 (Success), 500 (Server Error)

### GET `/api/feedback/recent`
**Purpose**: Get recent feedback entries  
**Authentication**: None required  
**Query Parameters**: `limit` (default: 10)  
**Response**: List of recent feedback entries
**Status Codes**: 200 (Success), 500 (Server Error)

---

## AI & Machine Learning

### GET `/api/embedding/stats`
**Purpose**: Get embedding system statistics and performance metrics  
**Authentication**: None required  
**Response**: Embedding model statistics, cache info, and performance metrics
**Status Codes**: 200 (Success), 500 (Server Error)

### POST `/api/embedding/clear-cache`
**Purpose**: Clear embedding cache for fresh calculations  
**Authentication**: None required  
**Response**: Cache clearing status
**Status Codes**: 200 (Success), 500 (Server Error)

### POST `/api/similarity/calculate`
**Purpose**: Calculate semantic similarity between queries and documents  
**Authentication**: None required  
**Request Body**:
```json
{
  "query": "string",
  "documents": ["string"],
  "use_cache": "boolean"
}
```
**Response**: Similarity scores and calculations
**Status Codes**: 200 (Success), 400 (Missing Data), 500 (Server Error)

### POST `/api/similarity/pairwise`
**Purpose**: Calculate pairwise similarity between multiple texts  
**Authentication**: None required  
**Request Body**:
```json
{
  "texts": ["string"],
  "include_self": "boolean"
}
```
**Response**: Pairwise similarity matrix
**Status Codes**: 200 (Success), 400 (Missing Data), 500 (Server Error)

### POST `/api/embedding/warmup`
**Purpose**: Pre-compute embeddings for faster search performance  
**Authentication**: None required  
**Request Body**:
```json
{
  "documents": ["string"],
  "batch_size": "integer"
}
```
**Response**: Warmup completion status and statistics
**Status Codes**: 200 (Success), 500 (Server Error)

### GET `/api/embedding/realtime-stats`
**Purpose**: Get real-time embedding system performance statistics  
**Authentication**: None required  
**Response**: Live performance metrics and system status
**Status Codes**: 200 (Success), 500 (Server Error)

### GET `/api/ltr/stats`
**Purpose**: Get Learning-to-Rank model statistics and performance  
**Authentication**: None required  
**Response**: LTR model status, training metrics, and feature information
**Status Codes**: 200 (Success), 500 (Server Error)

### POST `/api/ltr/train`
**Purpose**: Train the Learning-to-Rank model with collected feedback data  
**Authentication**: None required  
**Request Body**:
```json
{
  "min_feedback_count": "integer"
}
```
**Response**: Training completion status and model performance metrics
**Status Codes**: 200 (Success), 400 (Insufficient Data), 500 (Server Error)

### POST `/api/ltr/features`
**Purpose**: Extract LTR features for query-result pairs (testing/debugging)  
**Authentication**: None required  
**Request Body**:
```json
{
  "query": "string",
  "results": ["object"],
  "processed_query": "object"
}
```
**Response**: Extracted features for machine learning analysis
**Status Codes**: 200 (Success), 400 (Missing Data), 500 (Server Error)

### GET `/api/ltr/feature-importance`
**Purpose**: Get feature importance from trained LTR model  
**Authentication**: None required  
**Response**: Feature importance rankings and scores
**Status Codes**: 200 (Success), 400 (Model Not Trained), 500 (Server Error)

### GET `/api/preindex/stats`
**Purpose**: Get pre-built search index statistics  
**Authentication**: None required  
**Response**: Index status, size, and performance metrics
**Status Codes**: 200 (Success), 500 (Server Error)

### POST `/api/preindex/search`
**Purpose**: Perform semantic search using pre-built FAISS index  
**Authentication**: None required  
**Request Body**:
```json
{
  "query": "string",
  "top_k": "integer",
  "threshold": "float"
}
```
**Response**: Semantic search results with similarity scores
**Status Codes**: 200 (Success), 400 (Missing Query), 500 (Server Error)

### POST `/api/preindex/build`
**Purpose**: Build or rebuild the pre-computed search index  
**Authentication**: None required  
**Request Body**:
```json
{
  "force_rebuild": "boolean",
  "batch_size": "integer"
}
```
**Response**: Index building initiation status
**Status Codes**: 200 (Success), 500 (Server Error)

### GET `/api/preindex/progress`
**Purpose**: Get progress of index building operation  
**Authentication**: None required  
**Response**: Index building progress and status
**Status Codes**: 200 (Success), 500 (Server Error)

---

## User Data Management

### GET `/api/user/bookmarks`
**Purpose**: Get user's bookmarked search results  
**Authentication**: Bearer token required  
**Response**: List of user's bookmarks with metadata
**Status Codes**: 200 (Success), 401 (Unauthorized), 500 (Server Error)

### POST `/api/user/bookmarks/toggle`
**Purpose**: Add or remove bookmark for a search result  
**Authentication**: Bearer token required  
**Request Body**:
```json
{
  "result_id": "string",
  "title": "string",
  "database": "string",
  "type": "string"
}
```
**Response**: Bookmark toggle status
**Status Codes**: 200 (Success), 401 (Unauthorized), 500 (Server Error)

### POST `/api/user/bookmarks/remove`
**Purpose**: Remove specific bookmark  
**Authentication**: Bearer token required  
**Request Body**:
```json
{
  "result_id": "string"
}
```
**Response**: Bookmark removal status
**Status Codes**: 200 (Success), 401 (Unauthorized), 500 (Server Error)

### DELETE `/api/user/bookmarks/<bookmark_id>`
**Purpose**: Delete bookmark by ID  
**Authentication**: Bearer token required  
**Path Parameters**: `bookmark_id` - Bookmark identifier  
**Response**: Deletion status
**Status Codes**: 200 (Success), 401 (Unauthorized), 404 (Not Found)

### DELETE `/api/user/bookmarks/clear`
**Purpose**: Clear all user bookmarks  
**Authentication**: Bearer token required  
**Response**: Clear operation status
**Status Codes**: 200 (Success), 401 (Unauthorized), 500 (Server Error)

### GET `/api/user/history`
**Purpose**: Get user's search history  
**Authentication**: Bearer token required  
**Query Parameters**: `limit`, `offset`  
**Response**: List of user's previous searches
**Status Codes**: 200 (Success), 401 (Unauthorized), 500 (Server Error)

### POST `/api/user/history`
**Purpose**: Add search to user's history  
**Authentication**: Bearer token required  
**Request Body**:
```json
{
  "query": "string",
  "filters": ["string"],
  "timestamp": "string"
}
```
**Response**: History addition status
**Status Codes**: 200 (Success), 401 (Unauthorized), 500 (Server Error)

### DELETE `/api/user/history`
**Purpose**: Delete specific search from history  
**Authentication**: Bearer token required  
**Request Body**:
```json
{
  "search_id": "string"
}
```
**Response**: Deletion status
**Status Codes**: 200 (Success), 401 (Unauthorized), 500 (Server Error)

### DELETE `/api/user/history/clear`
**Purpose**: Clear entire search history  
**Authentication**: Bearer token required  
**Response**: Clear operation status
**Status Codes**: 200 (Success), 401 (Unauthorized), 500 (Server Error)

### POST `/api/user/history/remove`
**Purpose**: Remove search entry from history  
**Authentication**: Bearer token required  
**Request Body**:
```json
{
  "query": "string",
  "timestamp": "string"
}
```
**Response**: Removal status
**Status Codes**: 200 (Success), 401 (Unauthorized), 500 (Server Error)

---

## Funding System

### GET `/api/funding/institution/<institution_id>`
**Purpose**: Get detailed information about a funding institution  
**Authentication**: None required  
**Path Parameters**: `institution_id` - Institution identifier  
**Response**: Complete institution details including contact info, projects, and funding records
**Status Codes**: 200 (Success), 404 (Not Found), 500 (Server Error)

### GET `/api/funding/project/<project_id>`
**Purpose**: Get detailed information about a research project  
**Authentication**: None required  
**Path Parameters**: `project_id` - Research project identifier  
**Response**: Complete project details including methodology, objectives, budget, and funding status
**Status Codes**: 200 (Success), 404 (Not Found), 500 (Server Error)

---

## System & Utility

### GET `/api/health`
**Purpose**: System health check and component status  
**Authentication**: None required  
**Response**: Health status of all system components including databases, AI models, and services
**Status Codes**: 200 (Healthy), 503 (Unhealthy)

### POST `/api/test-llm`
**Purpose**: Test LLM connectivity and functionality  
**Authentication**: None required  
**Request Body**:
```json
{
  "query": "string"
}
```
**Response**: LLM processing test results
**Status Codes**: 200 (Success), 500 (LLM Error)

### GET `/api/test-llm-enhanced`
**Purpose**: Enhanced LLM testing with comprehensive analysis  
**Authentication**: None required  
**Response**: Detailed LLM performance and capability test results
**Status Codes**: 200 (Success), 500 (Server Error)

### GET `/api/test-db`
**Purpose**: Test database connectivity and performance  
**Authentication**: None required  
**Response**: Database connection status and basic performance metrics
**Status Codes**: 200 (Success), 500 (Database Error)

---

## Frontend Routes

### GET `/`
**Purpose**: Serve main landing page  
**Authentication**: None required  
**Response**: HTML landing page

### GET `/dev`
**Purpose**: Serve developer interface for testing  
**Authentication**: None required  
**Response**: HTML developer interface

### GET `/results`
**Purpose**: Serve search results page  
**Authentication**: None required  
**Response**: HTML search results interface

### GET `/login`
**Purpose**: Serve user login page  
**Authentication**: None required  
**Response**: HTML login form

### GET `/signup`
**Purpose**: Serve user registration page  
**Authentication**: None required  
**Response**: HTML registration form

### GET `/profile`
**Purpose**: Serve user profile page  
**Authentication**: None required  
**Response**: HTML profile interface

### GET `/dashboard`
**Purpose**: Serve user dashboard  
**Authentication**: None required  
**Response**: HTML user dashboard

### GET `/user-dashboard`
**Purpose**: Alternative user dashboard route  
**Authentication**: None required  
**Response**: HTML user dashboard

### GET `/forgot-password`
**Purpose**: Serve password reset page  
**Authentication**: None required  
**Response**: HTML password reset form

### GET `/test-history`
**Purpose**: Serve test history page for debugging  
**Authentication**: None required  
**Response**: HTML test interface

---

## Request/Response Examples

### Search Request Example
```json
POST /api/search
{
  "query": "machine learning for medical diagnosis",
  "ranking_mode": "hybrid",
  "database_filters": ["research_papers", "academic_library"],
  "session_id": "session_123456"
}
```

### Search Response Example
```json
{
  "success": true,
  "query_analysis": {
    "language_detection": {"detected_language": "en", "confidence": 0.99},
    "corrected_query": "machine learning for medical diagnosis",
    "intent": {"primary_intent": "find_research", "confidence": 0.95},
    "entities": {"technologies": ["machine learning"], "fields": ["medical"]},
    "keywords": {"primary": ["machine learning", "medical", "diagnosis"]},
    "academic_fields": ["Computer Science", "Medicine"]
  },
  "search_results": {
    "results": [...],
    "total_results": 156
  },
  "categorized_results": {
    "high_relevance": [...],
    "medium_relevance": [...], 
    "low_relevance": [...]
  },
  "query_id": "query_hash_timestamp",
  "ranking_mode": "hybrid",
  "personalization_applied": true,
  "user_type": "authenticated"
}
```

### User Profile Response Example
```json
{
  "success": true,
  "user": {
    "user_id": "uuid",
    "username": "researcher123",
    "email": "user@university.edu",
    "profile": {
      "first_name": "John",
      "last_name": "Doe",
      "academic_fields": ["Computer Science", "AI"],
      "institution": "University of Technology",
      "role": "PhD Student",
      "bio": "AI researcher focused on medical applications"
    },
    "statistics": {
      "total_searches": 245,
      "total_feedback": 67,
      "average_rating": 4.2,
      "favorite_topics": ["machine learning", "neural networks"]
    }
  }
}
```

### Error Response Example
```json
{
  "success": false,
  "error": "Query is required",
  "error_code": "MISSING_QUERY",
  "timestamp": "2025-01-21T10:30:00Z"
}
```

---

## Authentication & Authorization

**JWT Token Format**: Bearer tokens are used for authentication  
**Token Header**: `Authorization: Bearer <jwt_token>`  
**Token Expiry**: 24 hours (configurable)  
**Refresh**: Login required for new token

**Anonymous Usage**: Many endpoints support anonymous access with limited functionality  
**Rate Limiting**: Applied to prevent abuse (configurable per endpoint)  
**CORS**: Enabled for cross-origin requests

---

## Status Codes Summary

| Code | Meaning | Usage |
|------|---------|-------|
| 200 | OK | Successful request |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Invalid request data |
| 401 | Unauthorized | Authentication required |
| 403 | Forbidden | Access denied |
| 404 | Not Found | Resource not found |
| 500 | Internal Server Error | Server-side error |
| 503 | Service Unavailable | System component unavailable |

---

## Rate Limits

- **Search API**: 100 requests/hour for anonymous, 1000/hour for authenticated
- **Feedback API**: 500 submissions/hour  
- **Authentication**: 50 attempts/hour per IP
- **File Upload**: 10 uploads/hour per user

---

## Additional Notes

1. **Database Filtering**: Search can be limited to specific databases for focused results
2. **Ranking Modes**: 
   - `traditional`: Uses weighted combination of heuristic algorithms
   - `ltr_only`: Uses only machine learning model
   - `hybrid`: Combines ML (70%) with traditional (30%) - **recommended**
3. **Personalization**: Authenticated users get personalized results based on history and preferences
4. **Multilingual**: Supports queries in multiple languages with automatic translation
5. **Real-time**: Most endpoints provide real-time responses with caching for performance
6. **Feedback Loop**: User feedback continuously improves ranking algorithms
7. **Analytics**: Comprehensive tracking for system optimization and user insights 