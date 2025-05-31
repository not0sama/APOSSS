# APOSSS - AI-Powered Open-Science Semantic Search System

## Phase 3: AI Ranking & User Feedback System ✅

This repository contains the implementation of **Phase 3** of the APOSSS project, featuring **intelligent result ranking** and **user feedback collection** that builds upon the multi-database search from Phase 2.

## 🚀 New Phase 3 Features

- **🧠 AI-Powered Ranking**: Hybrid ranking algorithm combining heuristic, TF-IDF, embedding, and intent-based scoring
- **🎯 Semantic Similarity**: Deep learning embeddings for understanding meaning beyond keywords
- **📊 Score Visualization**: Visual ranking score bars with component breakdowns
- **🎯 Relevance Categories**: Results organized into High, Medium, and Low relevance tiers
- **👍 User Feedback System**: Thumbs up/down feedback collection for each result
- **📈 Feedback Analytics**: Real-time feedback statistics and storage
- **🔄 Learning Foundation**: Infrastructure for future ranking model improvements
- **🎨 Enhanced UI**: Modernized interface with ranking displays and interactive feedback

## 🏗️ System Architecture (Phase 3)

### Core Components
1. **LLM Query Processing** (Phase 1) - Gemini-2.0-flash powered query understanding
2. **Multi-Database Search** (Phase 2) - Search across 4 MongoDB databases (Academic, Experts, Research, Labs)
3. **🆕 AI Ranking Engine** (Phase 3) - Intelligent result ranking with multiple algorithms
4. **🆕 Feedback System** (Phase 3) - User feedback collection and storage in dedicated APOSSS database

### Ranking Algorithm
- **Heuristic Scoring (30%)**: Keyword matching in titles, descriptions, and metadata
- **TF-IDF Similarity (30%)**: Semantic similarity using scikit-learn TF-IDF vectorization
- **🆕 Embedding Similarity (20%)**: Deep semantic understanding using sentence transformers
- **Intent Alignment (20%)**: Resource type preference based on detected query intent

### Feedback Storage
- **Primary**: MongoDB APOSSS database (`APOSSS/user_feedback` collection)
- **Fallback**: JSON Lines file (`feedback_data.jsonl`) for reliability

## 📋 Requirements

### Dependencies
```
Flask==3.0.0
PyMongo==4.6.1
google-generativeai==0.8.3
python-dotenv==1.0.0
requests==2.31.0
flask-cors==4.0.0
scikit-learn==1.3.2
numpy==1.24.3
sentence-transformers==2.7.0
faiss-cpu==1.8.0
torch==2.1.0
```

### Environment Setup
Create a `config.env` file with:
```env
# Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# MongoDB Connection Strings
ACADEMIC_LIBRARY_URI=mongodb://localhost:27017/academic_library
EXPERTS_SYSTEM_URI=mongodb://localhost:27017/experts_system
RESEARCH_PAPERS_URI=mongodb://localhost:27017/research_papers
LABORATORIES_URI=mongodb://localhost:27017/laboratories
APOSSS_URI=mongodb://localhost:27017/APOSSS

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
PORT=5000
```

## 🚀 Quick Start

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd APOSSS
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   cp config.env.example config.env
   # Edit config.env with your API keys and database URIs
   ```

3. **Run Application**
   ```bash
   python app.py
   ```

4. **Access Interface**
   - Open http://localhost:5000 in your browser
   - Test with sample queries or enter your own research questions

## 🎯 API Endpoints

### Core Search
- `POST /api/search` - Full search with ranking and categorization
- `POST /api/test-llm` - LLM query analysis testing
- `GET /api/test-db` - Database connectivity testing
- `GET /api/health` - System health check

### Feedback System
- `POST /api/feedback` - Submit user feedback for results
- `GET /api/feedback/stats` - Get feedback statistics
- `GET /api/feedback/recent` - Get recent feedback entries

### 🆕 Embedding System
- `GET /api/embedding/stats` - Get embedding system statistics and cache info
- `POST /api/embedding/clear-cache` - Clear embedding cache for reprocessing

## 🔍 Usage Examples

### 1. Intelligent Search
```bash
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning for medical diagnosis"}'
```

### 2. Submit Feedback
```bash
curl -X POST http://localhost:5000/api/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "query_id": "uuid-here",
    "result_id": "result-id-here", 
    "rating": 5,
    "feedback_type": "thumbs_up"
  }'
```

### 3. Get Feedback Stats
```bash
curl http://localhost:5000/api/feedback/stats
```

## 📊 Phase 3 Key Features

### Ranking Engine
- **Multi-Algorithm Scoring**: Combines heuristic, TF-IDF, and intent-based scoring
- **Relevance Categorization**: Automatically groups results into relevance tiers
- **Score Transparency**: Provides breakdown of scoring components
- **Intent-Aware Ranking**: Prioritizes results based on detected user intent

### User Feedback System
- **Simple Interface**: Thumbs up/down for each result
- **Persistent Storage**: MongoDB with file fallback
- **Real-time Stats**: Live feedback analytics
- **Learning Ready**: Infrastructure for future ML model training

### Enhanced UI
- **Visual Score Bars**: Color-coded relevance scoring display
- **Relevance Sections**: Results organized by relevance level
- **Interactive Feedback**: One-click rating system
- **Ranking Transparency**: Score component visualization

## 🧪 Testing

### Test Search & Ranking
1. Visit http://localhost:5000
2. Try sample queries or enter custom research questions
3. Observe ranking scores and relevance categories
4. Provide feedback on results to test the feedback system

### Monitor Feedback
1. Click "View Feedback Stats" to see collected feedback
2. Submit various ratings to test the feedback storage
3. Check the `feedback_data.jsonl` file for stored feedback

## 📈 Future Development (Phase 4 & 5)

### Planned Enhancements
- **Learning Algorithm**: Train ranking models using collected feedback
- **Advanced Filtering**: Date, type, and field-based result filtering
- **A/B Testing**: Compare different ranking algorithms
- **User Personalization**: Adapt rankings to user preferences
- **Performance Optimization**: Caching and query optimization

### Integration Ready
- **Feedback Loop**: Ready for ML model retraining
- **Analytics Dashboard**: Foundation for detailed usage analytics
- **API Extensibility**: Designed for easy feature additions

## 🏆 Technical Achievements

### Phase 3 Implementations
✅ **Hybrid Ranking Algorithm** - Multi-component scoring system  
✅ **TF-IDF Integration** - Semantic similarity scoring  
✅ **Intent-Based Ranking** - Query intent alignment  
✅ **User Feedback Collection** - Complete feedback system  
✅ **Relevance Categorization** - Automatic result grouping  
✅ **Score Visualization** - Transparent ranking display  
✅ **Feedback Analytics** - Real-time statistics  
✅ **Robust Storage** - MongoDB with file backup  

## 🔧 Troubleshooting

### Common Issues
1. **Ranking Errors**: Ensure scikit-learn and numpy are installed
2. **Feedback Storage**: Check MongoDB connection or file permissions
3. **Score Display**: Verify all ranking components are functioning
4. **Memory Usage**: TF-IDF can be memory-intensive for large result sets

### Debug Mode
- Set `FLASK_DEBUG=True` for detailed error messages
- Check application logs for ranking and feedback system status
- Use `/api/health` endpoint to verify all components

## 📝 License

This project is developed for academic research purposes.

---

**APOSSS Phase 3** - Intelligent ranking and user feedback system for open science research discovery.

### 🧠 Embedding Technology
- **Model**: all-MiniLM-L6-v2 sentence transformer (384-dimensional embeddings)
- **Vector Storage**: FAISS for efficient similarity search and caching
- **Semantic Understanding**: Captures meaning beyond keyword matching for better relevance 