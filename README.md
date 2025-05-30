# APOSSS - AI-Powered Open-Science Semantic Search System

## Phase 1: Foundation & LLM Query Understanding

This repository contains the implementation of Phase 1 of the APOSSS project, focusing on setting up the foundation and implementing LLM-based query understanding using Gemini-2.0-flash.

## 🚀 Features

- **LLM Integration**: Gemini-2.0-flash for advanced query understanding
- **Database Connections**: MongoDB connections to all four databases
- **Query Processing**: Structured analysis of user queries including:
  - Spelling correction
  - Intent detection
  - Named Entity Recognition (NER)
  - Keyword extraction
  - Synonym generation
  - Academic field identification
- **Web Interface**: Interactive testing interface with real-time results
- **Health Monitoring**: System status and database connection testing

## 📋 Prerequisites

- Python 3.8+
- MongoDB (local installations for all four databases)
- Gemini API key from Google AI Studio

## 🛠️ Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <your-repo-url>
   cd APOSSS
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   - Copy `config.env.example` to `.env`
   - Add your Gemini API key and database URIs

   ```bash
   # Copy the example file
   cp config.env.example .env
   
   # Edit the .env file with your configurations
   # Add your GEMINI_API_KEY and update MongoDB URIs as needed
   ```

4. **Ensure MongoDB is running**:
   - Start your MongoDB service
   - Ensure all four databases are accessible:
     - Academic_Library
     - Experts_System
     - Research_Papers
     - Laboratories

## 🔧 Configuration

### Environment Variables

Edit your `.env` file with the following:

```env
# Gemini API Configuration
GEMINI_API_KEY=your_actual_gemini_api_key_here

# MongoDB Configuration (update as needed)
MONGODB_URI_ACADEMIC_LIBRARY=mongodb://localhost:27017/Academic_Library
MONGODB_URI_EXPERTS_SYSTEM=mongodb://localhost:27017/Experts_System
MONGODB_URI_RESEARCH_PAPERS=mongodb://localhost:27017/Research_Papers
MONGODB_URI_LABORATORIES=mongodb://localhost:27017/Laboratories

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
```

### Getting a Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/)
2. Create a new project or select an existing one
3. Navigate to "Get API Key" section
4. Generate a new API key
5. Copy the key to your `.env` file

## 🚀 Running the Application

1. **Start the Flask application**:
   ```bash
   python app.py
   ```

2. **Open your web browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **Test the system**:
   - Check system health
   - Test database connections
   - Try the LLM query processing with sample queries

## 🧪 Testing

### System Health Check

The application includes several endpoints for testing:

- **Health Check**: `GET /api/health`
- **Database Test**: `GET /api/test-db`
- **LLM Test**: `POST /api/test-llm`

### Sample Queries

Try these sample queries to test the LLM processing:

- "reducing carbon emissions in cars"
- "machine learning for medical diagnosis"
- "renewable energy storage solutions"
- "IoT sensors for agriculture"

### Expected Results

The LLM processor should return structured JSON containing:

- **Corrected Query**: Spelling-corrected version
- **Intent Analysis**: Primary/secondary intents with confidence
- **Entities**: Technologies, concepts, people, organizations
- **Keywords**: Primary, secondary, and technical terms
- **Synonyms & Related Terms**: Alternative and related terminology
- **Academic Fields**: Primary field and specializations
- **Search Context**: Resource preferences and scope

## 📁 Project Structure

```
APOSSS/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── config.env.example         # Environment configuration template
├── README.md                   # This file
├── modules/                    # Core modules
│   ├── __init__.py
│   ├── database_manager.py     # MongoDB connection management
│   ├── llm_processor.py        # Gemini LLM processing
│   └── query_processor.py      # Query processing orchestration
└── templates/
    └── index.html              # Web interface
```

## 🔍 API Endpoints

### Health Check
```http
GET /api/health
```
Returns system component status.

### Database Test
```http
GET /api/test-db
```
Tests all MongoDB database connections and returns collection counts.

### LLM Processing Test
```http
POST /api/test-llm
Content-Type: application/json

{
    "query": "your research query here"
}
```
Processes a query through the LLM and returns structured analysis.

## 🐛 Troubleshooting

### Common Issues

1. **Gemini API Errors**:
   - Verify your API key is correct
   - Check your Google Cloud billing account
   - Ensure the Generative AI API is enabled

2. **Database Connection Issues**:
   - Verify MongoDB is running
   - Check database URIs in `.env`
   - Ensure databases exist (they can be empty for testing)

3. **Module Import Errors**:
   - Verify all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility

4. **JSON Parsing Errors**:
   - This may occur with complex queries
   - The system includes fallback responses
   - Check logs for details

### Logs

The application logs important information to the console. Look for:
- Component initialization status
- Database connection results
- LLM processing success/failure
- Error details

## 📝 Next Steps (Phase 2)

After Phase 1 is working correctly:

1. Implement multi-database search functionality
2. Add basic ranking algorithms
3. Integrate search results display
4. Begin collecting user feedback

## 🤝 Contributing

This is Phase 1 of the development. Focus areas for improvement:

- LLM prompt optimization
- Error handling enhancement
- UI/UX improvements
- Performance optimization

## 📄 License

[Your License Here]

## 📞 Support

For issues with Phase 1 setup:
1. Check this README thoroughly
2. Review application logs
3. Test individual components using the web interface
4. Verify all prerequisites are met 