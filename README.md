# APOSSS - AI-Powered Open-Science Semantic Search System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3.2-green.svg)](https://flask.palletsprojects.com)
[![MongoDB](https://img.shields.io/badge/MongoDB-4.4+-green.svg)](https://mongodb.com)
[![License](https://img.shields.io/badge/License-Academic%20Research-blue.svg)](LICENSE)

## 🎯 Project Overview

**APOSSS** is a comprehensive AI-powered semantic search system designed for open science research discovery. The system intelligently searches across multiple academic databases and provides ranked, relevant results using advanced machine learning algorithms.

### ✨ Key Features

- **🧠 AI-Powered Query Understanding**: Advanced LLM processing with Gemini-2.0-flash for natural language query interpretation
- **🔍 Multi-Database Search**: Seamless search across 6 MongoDB databases (Academic Library, Experts, Research Papers, Laboratories, Funding, and APOSSS)
- **📊 Intelligent Ranking**: Hybrid ranking algorithm combining heuristic, TF-IDF, embedding similarity, and intent-based scoring
- **🎯 Semantic Understanding**: Deep learning embeddings for meaning-based search beyond keyword matching
- **👍 User Feedback System**: Complete feedback collection and analytics for continuous improvement
- **🔐 User Authentication**: Secure user management with OAuth integration (Google, ORCID)
- **📈 Learning-to-Rank**: Advanced LTR models for personalized result ranking
- **🎨 Modern UI**: Responsive web interface with real-time feedback and analytics

## 🏗️ System Architecture

### Core Components

1. **LLM Query Processing** - Natural language understanding and query enrichment
2. **Multi-Database Search Engine** - Unified search across 6 specialized databases
3. **AI Ranking Engine** - Hybrid ranking with multiple ML algorithms
4. **User Feedback System** - Feedback collection and analytics
5. **User Management** - Authentication and user profiles
6. **OAuth Integration** - Social login with Google and ORCID

### Database Integration

- **Academic Library**: Books, journals, and academic projects
- **Experts System**: Research experts and certifications
- **Research Papers**: Articles, conferences, and theses
- **Laboratories**: Equipment and materials
- **Funding**: Research funding opportunities
- **APOSSS**: User data, feedback, and system analytics

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- MongoDB 4.4+
- Google Gemini API key
- Email configuration (for user registration)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd APOSSS
   ```

2. **Create virtual environment**
   ```bash
   python -m venv APOSSSenv
   source APOSSSenv/bin/activate  # On Windows: APOSSSenv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp config.env.example .env
   # Edit .env with your configuration
   ```

5. **Start the application**
   ```bash
   python app.py
   ```

6. **Access the system**
   - Open http://localhost:5000 in your browser
   - Register a new account or use OAuth login

## ⚙️ Configuration

### Required Environment Variables

```env
# Security (Required)
JWT_SECRET_KEY=your_super_secure_jwt_secret_key_here_minimum_32_characters
SECRET_KEY=your_flask_secret_key_for_sessions_here

# AI/LLM (Required)
GEMINI_API_KEY=your_gemini_api_key_here

# Email (Required for registration)
SMTP_USERNAME=your_email@domain.com
SMTP_PASSWORD=your_email_app_password
```

### Optional Configuration

```env
# Database URLs (defaults to localhost if not set)
MONGODB_URI_ACADEMIC_LIBRARY=mongodb://localhost:27017/Academic_Library
MONGODB_URI_EXPERTS_SYSTEM=mongodb://localhost:27017/Experts_System
MONGODB_URI_RESEARCH_PAPERS=mongodb://localhost:27017/Research_Papers
MONGODB_URI_LABORATORIES=mongodb://localhost:27017/Laboratories
MONGODB_URI_FUNDING=mongodb://localhost:27017/Funding
MONGODB_URI_APOSSS=mongodb://localhost:27017/APOSSS

# OAuth (Optional)
GOOGLE_CLIENT_ID=your_google_client_id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your_google_client_secret
ORCID_CLIENT_ID=your_orcid_client_id
ORCID_CLIENT_SECRET=your_orcid_client_secret
```

## 🔧 API Endpoints

### Core Search
- `POST /api/search` - Intelligent search with AI ranking
- `GET /api/health` - System health check
- `GET /api/test-db` - Database connectivity test

### User Management
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `POST /api/auth/logout` - User logout
- `GET /api/auth/profile` - User profile

### Feedback System
- `POST /api/feedback` - Submit user feedback
- `GET /api/feedback/stats` - Feedback analytics
- `GET /api/feedback/recent` - Recent feedback

### Advanced Features
- `GET /api/embedding/stats` - Embedding system statistics
- `POST /api/embedding/clear-cache` - Clear embedding cache
- `GET /api/ranking/stats` - Ranking system analytics

## 🧠 AI & Machine Learning Features

### Ranking Algorithms

1. **Heuristic Scoring (30%)**: Keyword matching and metadata analysis
2. **TF-IDF Similarity (30%)**: Text similarity using scikit-learn
3. **Embedding Similarity (20%)**: Semantic understanding with sentence transformers
4. **Intent Alignment (20%)**: Query intent-based resource prioritization

### Learning-to-Rank Models

- **XGBoost Ranking**: Gradient boosting for personalized ranking
- **Feature Engineering**: 20+ features including text similarity, metadata, and user behavior
- **Continuous Learning**: Model updates based on user feedback

### Embedding Technology

- **Model**: all-MiniLM-L6-v2 sentence transformer (384-dimensional)
- **Vector Storage**: FAISS for efficient similarity search
- **Caching**: Intelligent caching for performance optimization

## 📊 User Interface

### Search Interface
- **Natural Language Input**: Type research questions in plain English
- **Real-time Results**: Instant search with AI-powered ranking
- **Relevance Categories**: Results organized by High/Medium/Low relevance
- **Score Visualization**: Transparent ranking with component breakdowns

### User Dashboard
- **Profile Management**: Update user information and preferences
- **Search History**: Track previous searches and results
- **Feedback Analytics**: View personal feedback statistics
- **OAuth Integration**: Login with Google or ORCID

### Feedback System
- **One-click Rating**: Thumbs up/down for each result
- **Real-time Analytics**: Live feedback statistics
- **Learning Integration**: Feedback used for model improvement

## 🚀 Production Deployment

### Environment Setup

1. **Production Configuration**
   ```env
   FLASK_ENV=production
   FLASK_DEBUG=false
   DEVELOPMENT_MODE=false
   ```

2. **Security Hardening**
   ```env
   FORCE_HTTPS=true
   RATE_LIMIT_PER_MINUTE=60
   MAX_FAILED_LOGIN_ATTEMPTS=5
   ```

3. **Performance Optimization**
   ```env
   EMBEDDING_CACHE_SIZE_MB=1024
   FAISS_INDEX_REBUILD_HOURS=24
   WORKERS=4
   ```

### Deployment Checklist

- [ ] Configure production environment variables
- [ ] Set up SSL/HTTPS certificates
- [ ] Configure email server (SMTP)
- [ ] Set up MongoDB with proper authentication
- [ ] Configure OAuth applications (Google, ORCID)
- [ ] Set up monitoring and logging
- [ ] Configure backup strategies

## 🧪 Testing

### Manual Testing
1. **Search Functionality**: Test various query types and observe ranking
2. **User Registration**: Test email verification and OAuth login
3. **Feedback System**: Submit feedback and verify storage
4. **Performance**: Test with large result sets

### Automated Testing
```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/integration/

# Run performance tests
python -m pytest tests/performance/
```

## 📈 Performance & Monitoring

### System Metrics
- **Search Latency**: Average response time for queries
- **Ranking Accuracy**: Feedback-based ranking quality metrics
- **User Engagement**: Search frequency and feedback rates
- **System Health**: Database connectivity and API availability

### Logging
- **Application Logs**: `logs/aposss.log`
- **Error Tracking**: Comprehensive error logging and monitoring
- **Performance Metrics**: Query performance and ranking statistics

## 🔒 Security Features

### Authentication & Authorization
- **JWT Tokens**: Secure session management
- **Password Hashing**: bcrypt for password security
- **OAuth Integration**: Secure social login
- **Rate Limiting**: Protection against abuse

### Data Protection
- **Input Validation**: Comprehensive input sanitization
- **SQL Injection Protection**: Parameterized queries
- **XSS Protection**: Content Security Policy headers
- **HTTPS Enforcement**: Secure data transmission

## 📚 Documentation

### Additional Resources
- [Deployment Guide](DEPLOYMENT_SETUP.md) - Production deployment instructions
- [Developer Documentation](Developer_Documentation.md) - Technical implementation details
- [OAuth Setup Guide](MDfiles/OAUTH_SETUP_GUIDE.md) - OAuth configuration
- [Email Setup Guide](MDfiles/EMAIL_SETUP_GUIDE.md) - Email configuration

### API Documentation
- Complete API reference available at `/api/docs` (when running)
- Interactive API testing with Swagger UI
- Example requests and responses for all endpoints

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Code Standards
- Follow PEP 8 Python style guidelines
- Add docstrings for all functions and classes
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is developed for academic research purposes. See the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Getting Help
- Check the [troubleshooting section](#troubleshooting) for common issues
- Review the logs: `tail -f logs/aposss.log`
- Verify your `.env` configuration
- Ensure all required services are running

### Common Issues

**Issue**: JWT_SECRET_KEY not set
- **Solution**: Set a strong JWT secret key in your `.env` file

**Issue**: Email sending failed
- **Solution**: Use Gmail App Password (not regular password) and enable 2FA

**Issue**: Database connection failed
- **Solution**: Ensure MongoDB is running and check connection strings

## 🏆 Project Status

✅ **COMPLETED** - The APOSSS system is fully functional and ready for production deployment.

### Completed Features
- ✅ Multi-database search integration
- ✅ AI-powered ranking system
- ✅ User authentication and management
- ✅ Feedback collection and analytics
- ✅ OAuth integration
- ✅ Learning-to-rank models
- ✅ Production-ready deployment configuration
- ✅ Comprehensive testing suite
- ✅ Security hardening
- ✅ Performance optimization

---

**APOSSS** - Empowering open science research through intelligent semantic search and AI-driven discovery.

*Built with ❤️ for the academic research community* 