# APOSSS Production Deployment Setup Guide

This guide helps you configure APOSSS for production deployment with all necessary environment variables and security settings.

## 🚀 Quick Setup

### 1. Configuration Files

Choose one of the configuration options:

**Option A: Production Deployment**
```bash
cp config.env.example .env
# Edit .env with your production values
```

**Option B: Development/Testing**
```bash
cp config.env.development .env
# Edit .env with your development values
```

### 2. Required Environment Variables

The following environment variables are **REQUIRED** for the system to work:

#### 🔑 Critical Security Settings
```bash
# REQUIRED: JWT secret for user authentication
JWT_SECRET_KEY=your_super_secure_jwt_secret_key_here_minimum_32_characters

# REQUIRED: Gemini API for LLM functionality
GEMINI_API_KEY=your_gemini_api_key_here

# REQUIRED: Flask secret key for sessions
SECRET_KEY=your_flask_secret_key_for_sessions_here
```

#### 📧 Email Configuration (Required for user registration)
```bash
SMTP_USERNAME=your_email@domain.com
SMTP_PASSWORD=your_email_app_password
```

### 3. Optional but Recommended

#### 🗄️ Database URLs (uses localhost defaults if not set)
```bash
MONGODB_URI_ACADEMIC_LIBRARY=mongodb://localhost:27017/Academic_Library
MONGODB_URI_EXPERTS_SYSTEM=mongodb://localhost:27017/Experts_System
MONGODB_URI_RESEARCH_PAPERS=mongodb://localhost:27017/Research_Papers
MONGODB_URI_LABORATORIES=mongodb://localhost:27017/Laboratories
MONGODB_URI_FUNDING=mongodb://localhost:27017/Funding
MONGODB_URI_APOSSS=mongodb://localhost:27017/APOSSS
```

#### 🔐 OAuth (for social login)
```bash
GOOGLE_CLIENT_ID=your_google_client_id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your_google_client_secret
ORCID_CLIENT_ID=your_orcid_client_id
ORCID_CLIENT_SECRET=your_orcid_client_secret
```

## 🛡️ Security Configuration

### 1. Generate Secure Keys

**JWT Secret Key (minimum 32 characters):**
```bash
# Generate a secure JWT secret
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

**Flask Secret Key:**
```bash
# Generate a secure Flask secret
python3 -c "import secrets; print(secrets.token_hex(32))"
```

### 2. Email Configuration

For Gmail SMTP (recommended for development):

1. Enable 2-Factor Authentication on your Google account
2. Generate an App Password:
   - Google Account → Security → 2-Step Verification → App passwords
   - Generate password for "Mail"
3. Use the generated password as `SMTP_PASSWORD`

### 3. HTTPS/SSL Configuration

For production, enable HTTPS:

```bash
SSL_CERT_PATH=/path/to/ssl/cert.pem
SSL_KEY_PATH=/path/to/ssl/private.key
FORCE_HTTPS=true
BASE_URL=https://yourdomain.com
```

## 🌐 Production Deployment

### 1. Production Environment Settings

```bash
# Production configuration
FLASK_ENV=production
FLASK_DEBUG=false
DEVELOPMENT_MODE=false

# Production URLs
BASE_URL=https://yourdomain.com
FRONTEND_URL=https://yourdomain.com
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

### 2. Performance Settings

```bash
# Server configuration
HOST=0.0.0.0
PORT=5000
WORKERS=4
MAX_CONNECTIONS=1000

# Caching
EMBEDDING_CACHE_SIZE_MB=1024
FAISS_INDEX_REBUILD_HOURS=24
```

### 3. Security Hardening

```bash
# Strong password policy
MIN_PASSWORD_LENGTH=8
REQUIRE_PASSWORD_COMPLEXITY=true

# Rate limiting
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000

# Session security
SESSION_TIMEOUT_HOURS=24
MAX_FAILED_LOGIN_ATTEMPTS=5
ACCOUNT_LOCKOUT_MINUTES=30
```

## 📊 Monitoring & Logging

### 1. Logging Configuration

```bash
LOG_LEVEL=INFO
LOG_FILE_PATH=logs/aposss.log
LOG_MAX_SIZE_MB=100
LOG_BACKUP_COUNT=5
```

### 2. Health Monitoring

```bash
HEALTH_CHECK_INTERVAL_SECONDS=30
PERFORMANCE_MONITORING=true
ERROR_TRACKING=true
```

## 🗂️ Directory Structure

Ensure these directories exist and have proper permissions:

```bash
mkdir -p logs
mkdir -p uploads
mkdir -p backups
mkdir -p embedding_cache
mkdir -p production_index_cache
mkdir -p ltr_models

# Set proper permissions
chmod 755 logs uploads backups
chmod 755 embedding_cache production_index_cache ltr_models
```

## 🧪 Testing Your Configuration

### 1. Environment Validation

```bash
# Test if required environment variables are set
python3 -c "
import os
required = ['JWT_SECRET_KEY', 'GEMINI_API_KEY', 'SECRET_KEY']
missing = [key for key in required if not os.getenv(key)]
if missing:
    print(f'❌ Missing required environment variables: {missing}')
else:
    print('✅ All required environment variables are set')
"
```

### 2. Start the Application

```bash
python app.py
```

### 3. Health Check

```bash
curl http://localhost:5000/api/health
```

## 🔧 Common Issues

### Issue: JWT_SECRET_KEY not set
**Error:** `ValueError: JWT_SECRET_KEY environment variable is required`
**Solution:** Set a strong JWT secret key in your .env file

### Issue: Email sending failed
**Error:** `SMTP authentication failed`
**Solution:** 
1. Use Gmail App Password (not regular password)
2. Enable 2FA on your Google account
3. Verify SMTP_USERNAME and SMTP_PASSWORD

### Issue: Database connection failed
**Error:** `MongoDB connection timeout`
**Solution:** 
1. Ensure MongoDB is running
2. Check database URIs in .env file
3. Verify network connectivity

## 📋 Environment Variables Reference

### Required Variables
- `JWT_SECRET_KEY` - JWT token signing (minimum 32 characters)
- `GEMINI_API_KEY` - Google Gemini API access
- `SECRET_KEY` - Flask session encryption

### Authentication
- `SMTP_USERNAME` - Email server username
- `SMTP_PASSWORD` - Email server password (App Password for Gmail)

### Database
- `MONGODB_URI_*` - MongoDB connection strings (6 databases)

### OAuth (Optional)
- `GOOGLE_CLIENT_ID` - Google OAuth client ID
- `GOOGLE_CLIENT_SECRET` - Google OAuth client secret
- `ORCID_CLIENT_ID` - ORCID OAuth client ID
- `ORCID_CLIENT_SECRET` - ORCID OAuth client secret

### Application
- `BASE_URL` - Application base URL
- `APP_NAME` - Application display name
- `FLASK_ENV` - Flask environment (development/production)

See `config.env.example` for complete list of available configuration options.

## 🆘 Support

If you encounter issues:

1. Check the logs: `tail -f logs/aposss.log`
2. Verify your .env file configuration
3. Ensure all required services (MongoDB) are running
4. Check network connectivity and firewall settings

For additional help, refer to the main README.md or project documentation. 