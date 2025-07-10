# Email Configuration for APOSSS
# This file contains email settings for the verification system

# Gmail SMTP Configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "osama01k2@gmail.com"

# IMPORTANT: Replace this with your Gmail App Password
# How to get Gmail App Password:
# 1. Go to https://myaccount.google.com/security
# 2. Make sure 2-Step Verification is enabled
# 3. Click on "App passwords" (you may need to sign in again)
# 4. Select "Mail" as the app and choose your device
# 5. Copy the 16-character password and paste it below
SMTP_PASSWORD = "taca jdll iryj zhld"

# Alternative: Use environment variable (more secure)
# SMTP_PASSWORD = os.getenv('GMAIL_APP_PASSWORD', 'YOUR_16_CHAR_APP_PASSWORD_HERE')

# Email Templates
EMAIL_SUBJECT = "Email Verification - Libyan Open Science"
EMAIL_SENDER_NAME = "Libyan Open Science Team" 