# OAuth Setup Guide for APOSSS

This guide provides step-by-step instructions for setting up Google and ORCID OAuth authentication for the APOSSS (Academic Portal for Open Science Search System) platform.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Google OAuth Setup](#google-oauth-setup)
3. [ORCID OAuth Setup](#orcid-oauth-setup)
4. [Environment Configuration](#environment-configuration)
5. [Testing OAuth Integration](#testing-oauth-integration)
6. [Troubleshooting](#troubleshooting)
7. [Security Considerations](#security-considerations)

## Prerequisites

Before setting up OAuth providers, ensure you have:

- A running APOSSS application
- Administrative access to Google Cloud Console and ORCID
- SSL/HTTPS enabled for production (required by OAuth providers)
- The following Python packages installed:
  ```bash
  pip install authlib requests-oauthlib bcrypt pyjwt
  ```

## Google OAuth Setup

### Step 1: Create a Google Cloud Project

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Select a project" → "New Project"
3. Enter project name: `APOSSS-OAuth` (or your preferred name)
4. Select your organization (if applicable)
5. Click "Create"

### Step 2: Enable Google+ API

1. In your Google Cloud project, go to "APIs & Services" → "Library"
2. Search for "Google+ API" or "People API"
3. Click on "Google+ API" and then "Enable"
4. Also enable "Gmail API" for email access (optional but recommended)

### Step 3: Configure OAuth Consent Screen

1. Go to "APIs & Services" → "OAuth consent screen"
2. Choose "External" for user type (unless you have a Google Workspace)
3. Fill in the required information:
   - **App name**: `APOSSS - Libyan Open Science`
   - **User support email**: Your support email
   - **App logo**: Upload your APOSSS logo (recommended: 120x120px)
   - **App domain**: Your domain (e.g., `yourdomain.com`)
   - **Authorized domains**: Add your domain
   - **Developer contact information**: Your email
4. **Scopes**: Add the following scopes:
   - `../auth/userinfo.email`
   - `../auth/userinfo.profile`
   - `openid`
5. **Test users**: Add email addresses that can test the app during development
6. Save and continue

### Step 4: Create OAuth Credentials

1. Go to "APIs & Services" → "Credentials"
2. Click "Create Credentials" → "OAuth 2.0 Client IDs"
3. Select "Web application" as the application type
4. **Name**: `APOSSS Web Client`
5. **Authorized JavaScript origins**:
   - Development: `http://localhost:5000`
   - Production: `https://yourdomain.com`
6. **Authorized redirect URIs**:
   - Development: `http://localhost:5000/api/auth/google/callback`
   - Production: `https://yourdomain.com/api/auth/google/callback`
7. Click "Create"
8. **Important**: Copy the Client ID and Client Secret - you'll need these for your environment configuration

### Step 5: Request Verification (For Production)

For production use, you may need to verify your app:
1. Go to "OAuth consent screen" → "Publishing status"
2. Click "Publish app"
3. If your app uses sensitive scopes, submit for verification
4. This process can take several days to weeks

## ORCID OAuth Setup

### Step 1: Register Your Application

1. Go to [ORCID Developer Tools](https://orcid.org/developer-tools)
2. Click "Register for the free ORCID public API"
3. Create an ORCID account if you don't have one
4. Fill in the application registration form:
   - **Application name**: `APOSSS - Libyan Open Science Portal`
   - **Application website**: Your main website URL
   - **Application description**: Brief description of your research portal
   - **Redirect URI**: 
     - Development: `http://localhost:5000/api/auth/orcid/callback`
     - Production: `https://yourdomain.com/api/auth/orcid/callback`

### Step 2: Get Your Credentials

1. After registration, you'll receive:
   - **Client ID**: Your ORCID application identifier
   - **Client Secret**: Your application secret
2. **Important**: Store these securely - the secret will only be shown once

### Step 3: Configure API Access

1. In your ORCID developer dashboard:
   - Choose the appropriate API: Public API (free) or Member API (paid)
   - Set your scope to `/authenticate` for basic authentication
   - For additional data access, you may need `/read-limited` scope
2. Test your integration using ORCID's sandbox environment first:
   - Sandbox base URL: `https://sandbox.orcid.org`
   - Production base URL: `https://orcid.org`

## Environment Configuration

Create a `.env` file in your project root with the following configuration:

```bash
# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True

# Application URLs
BASE_URL=http://localhost:5000
FRONTEND_URL=http://localhost:5000

# Google OAuth Configuration
GOOGLE_CLIENT_ID=your_google_client_id_here.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your_google_client_secret_here

# ORCID OAuth Configuration  
ORCID_CLIENT_ID=APP-XXXXXXXXXXXXXXXXX
ORCID_CLIENT_SECRET=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

# MongoDB Configuration (ensure this is set)
MONGODB_URI_APOSSS=mongodb://localhost:27017/APOSSS
```

### Production Environment Variables

For production, ensure you:

1. **Use HTTPS URLs**:
   ```bash
   BASE_URL=https://yourdomain.com
   FRONTEND_URL=https://yourdomain.com
   ```

2. **Secure your secrets**:
   - Use environment variables or secure key management
   - Never commit secrets to version control
   - Use different credentials for development and production

3. **Update OAuth provider settings**:
   - Update redirect URIs in Google Cloud Console
   - Update redirect URIs in ORCID developer dashboard

## Testing OAuth Integration

### Test Google OAuth

1. Start your APOSSS application
2. Navigate to the signup page
3. Click "Sign up with Google"
4. You should be redirected to Google's OAuth consent screen
5. Authorize the application
6. Verify you're redirected back and logged in successfully

### Test ORCID OAuth

1. Click "Sign up with ORCID"
2. You should be redirected to ORCID's authorization page
3. Sign in with your ORCID credentials
4. Authorize the application
5. Verify successful authentication and account creation

### Testing Checklist

- [ ] Google OAuth popup opens correctly
- [ ] Google consent screen displays your app information
- [ ] Google authorization completes successfully
- [ ] User profile data is captured correctly
- [ ] ORCID OAuth popup opens correctly
- [ ] ORCID authorization completes successfully
- [ ] ORCID profile data is captured correctly
- [ ] Account linking works for existing email addresses
- [ ] Error handling works for denied permissions
- [ ] Users are properly redirected after OAuth completion

## Troubleshooting

### Common Google OAuth Issues

**"Error 400: redirect_uri_mismatch"**
- Ensure redirect URI in Google Cloud Console exactly matches your app's callback URL
- Check for trailing slashes and HTTP vs HTTPS

**"This app isn't verified"**
- Normal for development; users can click "Advanced" → "Go to [app name] (unsafe)"
- For production, submit your app for verification

**"Access blocked: This app's request is invalid"**
- Check that all required scopes are properly configured
- Verify OAuth consent screen is properly filled out

### Common ORCID OAuth Issues

**"Invalid redirect URI"**
- Ensure redirect URI exactly matches what's registered in ORCID developer tools
- ORCID is case-sensitive for redirect URIs

**"Invalid client credentials"**
- Double-check your Client ID and Client Secret
- Ensure you're using the correct ORCID environment (sandbox vs production)

**"Insufficient scope"**
- Verify you're requesting the correct scopes (`/authenticate` for basic auth)
- Check if you need additional permissions for your use case

### General OAuth Issues

**"OAuth popup blocked"**
- Modern browsers may block popups; ensure users allow popups for your site
- Consider implementing fallback redirect-based OAuth flow

**"OAuth state mismatch"**
- This is a security feature; ensure state parameter is properly handled
- Check for timing issues or multiple concurrent OAuth attempts

**"Network errors during OAuth"**
- Verify all URLs are correct and accessible
- Check for firewall or proxy issues
- Ensure SSL certificates are valid for production

### Debugging Tips

1. **Enable detailed logging**:
   ```python
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Check browser console** for JavaScript errors during OAuth flow

3. **Verify environment variables** are properly loaded:
   ```python
   print("Google Client ID:", os.getenv('GOOGLE_CLIENT_ID'))
   print("ORCID Client ID:", os.getenv('ORCID_CLIENT_ID'))
   ```

4. **Test OAuth endpoints directly**:
   - Visit `/api/auth/google` and `/api/auth/orcid` in your browser
   - Check if proper redirect URLs are generated

## Security Considerations

### Data Protection

1. **User Data Handling**:
   - Only request necessary permissions
   - Store minimal user data required for your application
   - Implement proper data retention policies
   - Provide users with data deletion options

2. **Token Security**:
   - Store access tokens securely (if needed for API calls)
   - Implement token refresh logic for long-lived access
   - Never expose tokens in client-side code
   - Use HTTPS for all OAuth communications

### Access Control

1. **Scope Limitations**:
   - Request only required OAuth scopes
   - Regularly review and audit permissions
   - Implement proper authorization checks in your application

2. **Session Management**:
   - Implement secure session handling
   - Use secure, httpOnly cookies for session tokens
   - Implement proper logout functionality

### Compliance

1. **Privacy Policies**:
   - Update your privacy policy to reflect OAuth data collection
   - Clearly explain what data is collected from OAuth providers
   - Provide opt-out mechanisms where required

2. **Terms of Service**:
   - Update terms to include OAuth provider integrations
   - Comply with Google and ORCID terms of service
   - Implement proper user consent mechanisms

## Support and Resources

### Documentation Links

- [Google OAuth 2.0 Documentation](https://developers.google.com/identity/protocols/oauth2)
- [ORCID API Documentation](https://info.orcid.org/documentation/)
- [Flask-OAuthlib Documentation](https://flask-oauthlib.readthedocs.io/)

### Getting Help

1. **Google OAuth Support**:
   - Google Cloud Console support
   - Stack Overflow with `google-oauth` tag
   - Google Identity Platform documentation

2. **ORCID Support**:
   - ORCID Support Center
   - ORCID API Users Group
   - ORCID technical documentation

### Best Practices

1. **Regular Updates**:
   - Keep OAuth libraries updated
   - Monitor OAuth provider announcements for API changes
   - Regularly test OAuth flows to ensure continued functionality

2. **Monitoring**:
   - Implement OAuth success/failure rate monitoring
   - Log OAuth errors for debugging
   - Set up alerts for OAuth service disruptions

3. **User Experience**:
   - Provide clear instructions for OAuth authorization
   - Handle OAuth errors gracefully with user-friendly messages
   - Offer alternative registration methods if OAuth fails

---

**Note**: This guide assumes you're implementing OAuth for the APOSSS platform. Adjust URLs, application names, and specific configurations according to your deployment environment and requirements. 