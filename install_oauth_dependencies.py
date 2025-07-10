#!/usr/bin/env python3
"""
OAuth Dependencies Installation and Configuration Check Script
For APOSSS Social Login Integration
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def install_dependencies():
    """Install required OAuth dependencies"""
    print("🔧 Installing OAuth dependencies...")
    
    dependencies = [
        "authlib==1.2.1",
        "requests-oauthlib==1.3.1", 
        "bcrypt==4.0.1",
        "pyjwt==2.8.0"
    ]
    
    for dep in dependencies:
        print(f"  📦 Installing {dep}...")
        success, stdout, stderr = run_command(f"pip install {dep}")
        if success:
            print(f"  ✅ {dep} installed successfully")
        else:
            print(f"  ❌ Failed to install {dep}: {stderr}")
            return False
    
    return True

def check_environment_config():
    """Check if OAuth environment variables are configured"""
    print("\n🔍 Checking OAuth configuration...")
    
    # Load environment variables from .env file if it exists
    env_file = Path('.env')
    if env_file.exists():
        print("  📄 Found .env file")
        with open(env_file, 'r') as f:
            env_content = f.read()
            
        # Check for OAuth configurations
        oauth_vars = {
            'GOOGLE_CLIENT_ID': 'Google OAuth Client ID',
            'GOOGLE_CLIENT_SECRET': 'Google OAuth Client Secret',
            'ORCID_CLIENT_ID': 'ORCID OAuth Client ID', 
            'ORCID_CLIENT_SECRET': 'ORCID OAuth Client Secret',
            'BASE_URL': 'Base URL for OAuth callbacks',
            'FRONTEND_URL': 'Frontend URL'
        }
        
        for var, description in oauth_vars.items():
            if var in env_content and not f'{var}=your_' in env_content:
                print(f"  ✅ {description} configured")
            else:
                print(f"  ⚠️  {description} needs to be configured")
    else:
        print("  ⚠️  .env file not found. Please create one based on config.env.example")

def create_sample_env():
    """Create a sample .env file if it doesn't exist"""
    env_file = Path('.env')
    example_file = Path('config.env.example')
    
    if not env_file.exists() and example_file.exists():
        print("\n📄 Creating .env file from example...")
        try:
            with open(example_file, 'r') as source:
                content = source.read()
            
            with open(env_file, 'w') as target:
                target.write(content)
            
            print("  ✅ .env file created successfully")
            print("  📝 Please edit .env and add your OAuth credentials")
        except Exception as e:
            print(f"  ❌ Failed to create .env file: {e}")

def check_oauth_modules():
    """Check if OAuth modules are working"""
    print("\n🧪 Testing OAuth modules...")
    
    try:
        # Test OAuth manager import
        sys.path.append('modules')
        from oauth_manager import OAuthManager
        
        oauth_manager = OAuthManager()
        print("  ✅ OAuth Manager imported successfully")
        
        # Check if providers are configured
        google_configured = oauth_manager.is_configured('google')
        orcid_configured = oauth_manager.is_configured('orcid')
        
        print(f"  {'✅' if google_configured else '⚠️ '} Google OAuth: {'Configured' if google_configured else 'Not configured'}")
        print(f"  {'✅' if orcid_configured else '⚠️ '} ORCID OAuth: {'Configured' if orcid_configured else 'Not configured'}")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Failed to import OAuth manager: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Error testing OAuth modules: {e}")
        return False

def test_database_connection():
    """Test database connection for user management"""
    print("\n🗄️  Testing database connection...")
    
    try:
        sys.path.append('modules')
        from database_manager import DatabaseManager
        
        db_manager = DatabaseManager()
        status = db_manager.test_connections()
        
        if 'aposss' in status and status['aposss']['connected']:
            print("  ✅ APOSSS database connection successful")
            return True
        else:
            print("  ⚠️  APOSSS database not connected")
            return False
            
    except Exception as e:
        print(f"  ❌ Database connection test failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n📋 Next Steps:")
    print("1. 🔑 Set up OAuth providers:")
    print("   - Follow the OAUTH_SETUP_GUIDE.md for detailed instructions")
    print("   - Create Google Cloud Console project and OAuth credentials")
    print("   - Register ORCID application and get API credentials")
    print("   ")
    print("2. ⚙️  Configure environment variables:")
    print("   - Edit .env file with your OAuth credentials")
    print("   - Ensure BASE_URL and FRONTEND_URL are correct")
    print("   ")
    print("3. 🧪 Test the integration:")
    print("   - Start your Flask application")
    print("   - Navigate to /signup")
    print("   - Test Google and ORCID OAuth buttons")
    print("   ")
    print("4. 📖 Read the documentation:")
    print("   - Check OAUTH_SETUP_GUIDE.md for troubleshooting")
    print("   - Review security considerations")

def main():
    """Main installation and setup function"""
    print("🚀 APOSSS OAuth Setup Script")
    print("=" * 50)
    
    # Step 1: Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies. Please check your pip installation.")
        return False
    
    # Step 2: Create .env file if needed
    create_sample_env()
    
    # Step 3: Check environment configuration
    check_environment_config()
    
    # Step 4: Test OAuth modules
    oauth_working = check_oauth_modules()
    
    # Step 5: Test database connection
    db_working = test_database_connection()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Setup Summary:")
    print(f"  Dependencies: {'✅ Installed' if True else '❌ Failed'}")
    print(f"  OAuth Modules: {'✅ Working' if oauth_working else '❌ Failed'}")
    print(f"  Database: {'✅ Connected' if db_working else '⚠️  Needs setup'}")
    
    if oauth_working and db_working:
        print("\n🎉 OAuth setup is ready!")
        print("   Configure your OAuth providers and test the integration.")
    else:
        print("\n⚠️  Some components need attention.")
        print("   Please resolve the issues above before testing OAuth.")
    
    # Print next steps
    print_next_steps()
    
    return True

if __name__ == "__main__":
    main() 