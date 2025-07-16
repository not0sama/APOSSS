#!/usr/bin/env python3
"""
APOSSS Configuration Validation Script
Checks if all required environment variables are properly configured
"""

import os
import sys
from dotenv import load_dotenv

def load_environment():
    """Load environment variables from .env file"""
    if os.path.exists('.env'):
        load_dotenv('.env')
        print("✅ Loaded .env file")
        return True
    else:
        print("⚠️  No .env file found. Using system environment variables only.")
        return False

def check_required_variables():
    """Check critical required environment variables"""
    print("\n🔍 Checking required environment variables...")
    
    required_vars = {
        'JWT_SECRET_KEY': 'JWT secret for user authentication (minimum 32 characters)',
        'GEMINI_API_KEY': 'Google Gemini API key for LLM functionality',
        'SECRET_KEY': 'Flask secret key for session encryption'
    }
    
    missing = []
    warnings = []
    
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value:
            missing.append(f"❌ {var}: {description}")
        else:
            # Additional validation
            if var == 'JWT_SECRET_KEY' and len(value) < 32:
                warnings.append(f"⚠️  {var}: Should be at least 32 characters (current: {len(value)})")
            else:
                print(f"✅ {var}: Set ({len(value)} characters)")
    
    return missing, warnings

def check_email_configuration():
    """Check email/SMTP configuration"""
    print("\n📧 Checking email configuration...")
    
    smtp_username = os.getenv('SMTP_USERNAME')
    smtp_password = os.getenv('SMTP_PASSWORD')
    
    if not smtp_username or not smtp_password:
        print("⚠️  Email configuration incomplete:")
        if not smtp_username:
            print("   - SMTP_USERNAME not set")
        if not smtp_password:
            print("   - SMTP_PASSWORD not set")
        print("   Email features (registration, password reset) will not work")
        return False
    else:
        print(f"✅ SMTP configured for: {smtp_username}")
        return True

def check_database_configuration():
    """Check database configuration"""
    print("\n🗄️  Checking database configuration...")
    
    databases = [
        'MONGODB_URI_ACADEMIC_LIBRARY',
        'MONGODB_URI_EXPERTS_SYSTEM', 
        'MONGODB_URI_RESEARCH_PAPERS',
        'MONGODB_URI_LABORATORIES',
        'MONGODB_URI_FUNDING',
        'MONGODB_URI_APOSSS'
    ]
    
    configured = 0
    for db in databases:
        value = os.getenv(db)
        if value:
            print(f"✅ {db}: {value}")
            configured += 1
        else:
            print(f"⚠️  {db}: Using default (localhost)")
    
    print(f"Database configuration: {configured}/{len(databases)} explicitly configured")
    return configured

def check_oauth_configuration():
    """Check OAuth configuration"""
    print("\n🔐 Checking OAuth configuration...")
    
    google_configured = bool(os.getenv('GOOGLE_CLIENT_ID') and os.getenv('GOOGLE_CLIENT_SECRET'))
    orcid_configured = bool(os.getenv('ORCID_CLIENT_ID') and os.getenv('ORCID_CLIENT_SECRET'))
    
    if google_configured:
        print("✅ Google OAuth: Configured")
    else:
        print("⚠️  Google OAuth: Not configured")
    
    if orcid_configured:
        orcid_env = os.getenv('ORCID_ENVIRONMENT', 'production')
        print(f"✅ ORCID OAuth: Configured ({orcid_env})")
    else:
        print("⚠️  ORCID OAuth: Not configured")
    
    if not google_configured and not orcid_configured:
        print("   Social login features will not be available")
    
    return google_configured or orcid_configured

def check_security_configuration():
    """Check security-related configuration"""
    print("\n🛡️  Checking security configuration...")
    
    checks = []
    
    # Flask environment
    flask_env = os.getenv('FLASK_ENV', 'development')
    flask_debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    
    if flask_env == 'production' and flask_debug:
        checks.append("⚠️  FLASK_DEBUG is enabled in production environment")
    else:
        checks.append(f"✅ Flask environment: {flask_env} (debug: {flask_debug})")
    
    # Password policy
    min_password_length = int(os.getenv('MIN_PASSWORD_LENGTH', '8'))
    if min_password_length < 8:
        checks.append(f"⚠️  MIN_PASSWORD_LENGTH is {min_password_length} (recommended: 8+)")
    else:
        checks.append(f"✅ Password minimum length: {min_password_length}")
    
    # Rate limiting
    rate_limit = int(os.getenv('RATE_LIMIT_PER_MINUTE', '60'))
    checks.append(f"✅ Rate limiting: {rate_limit} requests/minute")
    
    for check in checks:
        print(check)
    
    return len([c for c in checks if c.startswith('⚠️')]) == 0

def check_ssl_configuration():
    """Check SSL/HTTPS configuration"""
    print("\n🔒 Checking SSL/HTTPS configuration...")
    
    ssl_cert = os.getenv('SSL_CERT_PATH')
    ssl_key = os.getenv('SSL_KEY_PATH')
    force_https = os.getenv('FORCE_HTTPS', 'false').lower() == 'true'
    base_url = os.getenv('BASE_URL', 'http://localhost:5000')
    
    if ssl_cert and ssl_key:
        if os.path.exists(ssl_cert) and os.path.exists(ssl_key):
            print("✅ SSL certificates: Found and accessible")
        else:
            print("❌ SSL certificates: Configured but files not found")
            return False
    else:
        print("⚠️  SSL certificates: Not configured")
    
    if base_url.startswith('https://'):
        print(f"✅ Base URL: {base_url} (HTTPS)")
    else:
        print(f"⚠️  Base URL: {base_url} (HTTP)")
    
    if force_https:
        print("✅ HTTPS enforcement: Enabled")
    else:
        print("⚠️  HTTPS enforcement: Disabled")
    
    return True

def generate_missing_config():
    """Generate commands to create missing configuration"""
    print("\n🔧 Quick fix commands:")
    print("# Generate secure keys:")
    print('export JWT_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")')
    print('export SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")')
    print("\n# Add to your .env file:")
    print("echo \"JWT_SECRET_KEY=$JWT_SECRET_KEY\" >> .env")
    print("echo \"SECRET_KEY=$SECRET_KEY\" >> .env")
    print("echo \"GEMINI_API_KEY=your_actual_api_key_here\" >> .env")

def main():
    """Main configuration check"""
    print("🔍 APOSSS Configuration Validator")
    print("=" * 50)
    
    # Load environment
    env_loaded = load_environment()
    
    # Check all configuration sections
    missing, warnings = check_required_variables()
    email_ok = check_email_configuration()
    db_count = check_database_configuration() 
    oauth_ok = check_oauth_configuration()
    security_ok = check_security_configuration()
    ssl_ok = check_ssl_configuration()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Configuration Summary")
    print("=" * 50)
    
    if missing:
        print("❌ CRITICAL ISSUES:")
        for issue in missing:
            print(f"   {issue}")
        print("\n🚨 System will NOT start without these variables!")
        generate_missing_config()
        sys.exit(1)
    
    if warnings:
        print("⚠️  WARNINGS:")
        for warning in warnings:
            print(f"   {warning}")
    
    print("\n✅ Required configuration: Complete")
    
    # Feature availability
    print("\n🎯 Feature Availability:")
    print(f"   📧 Email features: {'✅' if email_ok else '❌'}")
    print(f"   🔐 Social login: {'✅' if oauth_ok else '❌'}")
    print(f"   🗄️  Database: {'✅' if db_count > 0 else '❌'}")
    print(f"   🔒 HTTPS/SSL: {'✅' if ssl_ok else '⚠️'}")
    
    if not email_ok:
        print("\n💡 To enable email features:")
        print("   1. Set SMTP_USERNAME and SMTP_PASSWORD in .env")
        print("   2. For Gmail: use App Password, not regular password")
    
    if not oauth_ok:
        print("\n💡 To enable social login:")
        print("   1. Set up Google/ORCID OAuth applications")
        print("   2. Add client IDs and secrets to .env")
    
    print(f"\n🚀 System ready to start!")
    print("   Run: python app.py")

if __name__ == "__main__":
    main() 