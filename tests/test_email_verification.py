#!/usr/bin/env python3
"""
Test script for email verification system
Run this to test if everything is working correctly
"""

import requests
import json

# Configuration
BASE_URL = "http://localhost:5000"
TEST_EMAIL = "verifytest@example.com"
TEST_PASSWORD = "password123"

def test_email_verification():
    print("🔧 Testing Email Verification System")
    print("=" * 50)
    
    # Step 1: Register a test user
    print("1. Registering test user...")
    register_data = {
        "username": "testverify123",
        "email": TEST_EMAIL,
        "password": TEST_PASSWORD,
        "firstName": "Test",
        "lastName": "User"
    }
    
    response = requests.post(f"{BASE_URL}/api/auth/register", json=register_data)
    
    if response.status_code == 201:
        data = response.json()
        token = data['token']
        print(f"✅ User registered successfully!")
        print(f"📧 Email: {TEST_EMAIL}")
    elif response.status_code == 400 and "already exists" in response.text:
        print("ℹ️  User already exists, attempting login...")
        # Try to login
        login_response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "identifier": TEST_EMAIL,
            "password": TEST_PASSWORD
        })
        if login_response.status_code == 200:
            data = login_response.json()
            token = data['token']
            print(f"✅ Logged in successfully!")
        else:
            print("❌ Login failed. Please check credentials.")
            return
    else:
        print(f"❌ Registration failed: {response.text}")
        return
    
    # Step 2: Send verification code
    print("\n2. Sending verification code...")
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.post(f"{BASE_URL}/api/user/send-verification-code", headers=headers)
    
    if response.status_code == 200:
        print("✅ Verification code sent successfully!")
        print("📱 Check the server console for the verification code")
        print("🔐 Look for: 'VERIFICATION CODE FOR [email]: [6-digit-code]'")
        
        # Get verification code from user
        verification_code = input("\n📝 Enter the 6-digit verification code: ").strip()
        
        if len(verification_code) == 6 and verification_code.isdigit():
            # Step 3: Verify the code
            print("\n3. Verifying the code...")
            verify_response = requests.post(
                f"{BASE_URL}/api/user/verify-email", 
                headers=headers,
                json={"verification_code": verification_code}
            )
            
            if verify_response.status_code == 200:
                print("✅ Email verified successfully!")
                print("🎉 Email verification system is working perfectly!")
            else:
                error_data = verify_response.json()
                print(f"❌ Verification failed: {error_data.get('error', 'Unknown error')}")
        else:
            print("❌ Invalid verification code format. Must be 6 digits.")
            
    else:
        error_data = response.json()
        print(f"❌ Failed to send verification code: {error_data.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 50)
    print("🏁 Test completed!")

if __name__ == "__main__":
    try:
        test_email_verification()
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Flask application.")
        print("🔧 Make sure the Flask app is running: python3 app.py")
    except KeyboardInterrupt:
        print("\n⚠️  Test cancelled by user.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}") 