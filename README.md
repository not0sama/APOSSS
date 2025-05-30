# APOSSS - AI-Powered Open-Science Semantic Search System

## Phase 2: Multi-Database Search Implementation ✅

This repository contains the implementation of **Phase 2** of the APOSSS project, featuring **complete multi-database search functionality** that builds upon the LLM query understanding from Phase 1.

## 🚀 New Phase 2 Features

- **🔍 Multi-Database Search**: Search across all four MongoDB databases simultaneously
- **📊 Intelligent Query Processing**: Uses Gemini-2.0-flash LLM to understand and enhance queries
- **🎯 Smart Result Aggregation**: Combines and standardizes results from different database schemas
- **📈 Results Analytics**: Displays search statistics and breakdowns by database/type
- **🎨 Enhanced UI**: Modern interface with collapsible sections and result cards
- **⚡ Real-time Search**: Fast, responsive search with loading indicators

## 🗄️ Database Coverage

The search engine queries across:

1. **Academic Library**: Books, Journals, Projects
2. **Experts System**: Experts, Certificates  
3. **Research Papers**: Articles, Conferences, Theses
4. **Laboratories**: Equipment, Materials

**Total Collections Searched**: 10 collections across 4 databases

## 📋 Prerequisites

- Python 3.8+
- MongoDB (local installations for all four databases)
- Gemini API key from Google AI Studio

## 🛠️ Installation

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   ```bash
   # Copy the example file
   cp config.env.example .env
   
   # Edit .env with your Gemini API key
   ```

3. **Ensure MongoDB is running**:
   - Start your MongoDB service
   - The databases can be empty for testing - the system will handle empty collections gracefully

## 🚀 Running the Application

1. **Start the Flask application**:
   ```bash
   python app.py
   ```

2. **Open your web browser** and navigate to:
   ```
   http://localhost:5000
   ```

## 🧪 Testing Phase 2

### Main Search Interface

1. **Enter a research query** in the main search box
2. **Click "🔍 Search All Databases"**
3. **View comprehensive results** with:
   - Results summary (total count, breakdown by database/type)
   - Individual result cards with metadata
   - Source information and snippets

### Sample Search Queries

Try these queries to test the system:

- **"reducing carbon emissions in cars"** - Environmental engineering focus
- **"machine learning for medical diagnosis"** - AI healthcare applications  
- **"renewable energy storage solutions"** - Clean energy technology
- **"IoT sensors for agriculture"** - Smart farming technology

### Expected Search Results

For each query, you'll see:

- **📊 Results Summary**: Total count and distribution across databases
- **📚 Result Cards**: Individual resources with:
  - Title and author information
  - Description/snippet preview  
  - Resource type badges (book, expert, equipment, etc.)
  - Metadata (publication date, institution, status, etc.)
  - Source database and collection information

## 🔍 API Endpoints

### Full Search (Phase 2)
```http
POST /api/search
Content-Type: application/json

{
    "query": "your research query here"
}
```
**Response**: Complete search results with LLM analysis and database results

### Legacy Endpoints (Phase 1)
- **Health Check**: `GET /api/health`
- **Database Test**: `GET /api/test-db`  
- **LLM Test**: `POST /api/test-llm`

## 📁 Updated Project Structure

```
APOSSS/
├── app.py                      # Main Flask application with search endpoint
├── requirements.txt            # Python dependencies
├── config.env.example         # Environment configuration template
├── README.md                   # This file
├── modules/                    # Core modules
│   ├── __init__.py
│   ├── database_manager.py     # MongoDB connection management
│   ├── llm_processor.py        # Gemini LLM processing
│   ├── query_processor.py      # Query processing orchestration
│   └── search_engine.py        # 🆕 Multi-database search engine
└── templates/
    └── index.html              # Enhanced web interface with search UI
```

## 🎯 How Phase 2 Works

1. **Query Processing**: User query is analyzed by Gemini-2.0-flash LLM
2. **Parameter Extraction**: Keywords, entities, and search context extracted
3. **Multi-Database Search**: Parallel searching across all 4 databases
4. **Result Standardization**: Different schemas normalized to common format
5. **Aggregation**: Results combined with metadata and statistics
6. **Display**: Organized presentation with filtering and categorization

## 🔧 Search Algorithm Features

- **Smart Field Mapping**: Automatically searches relevant fields for each collection type
- **Priority-Based Matching**: Primary keywords weighted higher than secondary terms
- **Regex Escaping**: Safe handling of special characters in search terms
- **Result Limiting**: Maximum 50 results per collection to ensure performance
- **Error Handling**: Graceful handling of database connection issues

## 🎨 UI/UX Improvements

- **Collapsible LLM Testing**: Advanced query analysis available but not prominent
- **Result Type Badges**: Color-coded badges for different resource types
- **Hover Effects**: Interactive result cards with smooth animations
- **Loading States**: Clear feedback during search operations
- **Mobile Responsive**: Works well on different screen sizes

## 🐛 Troubleshooting

### Search Returns No Results

1. **Check Database Connections**: Use "Test Database Connections" button
2. **Verify Collections Exist**: Even empty collections will show 0 results
3. **Try Simpler Queries**: Start with single keywords
4. **Check Logs**: Look for search errors in console output

### Performance Issues

1. **Database Response Time**: Check MongoDB server status
2. **Result Limits**: Each collection limited to 50 results
3. **Query Complexity**: Very complex queries may be slower

## 📝 Next Steps (Phase 3)

Phase 2 is now **complete**! Ready for Phase 3:

1. **AI Ranking Model**: Implement intelligent result ranking
2. **User Feedback System**: Add thumbs up/down for results
3. **Learning Algorithm**: Use feedback to improve ranking
4. **Advanced Filtering**: Date ranges, resource types, etc.

## ✅ Phase 2 Achievements

- ✅ Multi-database search engine implemented
- ✅ All 10 collections searchable  
- ✅ Results aggregation and standardization
- ✅ Enhanced UI with search interface
- ✅ Comprehensive error handling
- ✅ Performance optimization (result limits, timeouts)
- ✅ Full integration between LLM processing and database search

**Phase 2 Goal Met**: ✅ *"Retrieve a comprehensive set of potentially relevant items from all databases based on the LLM-processed query"*

## 🤝 Contributing

Phase 2 focus areas for improvement:

- Search query optimization
- Result relevance tuning  
- UI/UX enhancements
- Performance optimization
- Additional metadata extraction

## 📄 License

[Your License Here]

## 📞 Support

For Phase 2 issues:
1. Check system health via the web interface
2. Test database connections  
3. Try sample queries first
4. Review console logs for detailed error information 