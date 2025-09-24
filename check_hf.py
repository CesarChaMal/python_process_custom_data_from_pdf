#!/usr/bin/env python3
"""
Hugging Face Hub Connection Tester

This utility script validates the Hugging Face Hub authentication token
and tests connectivity to the Hugging Face API. It's useful for:
- Verifying API token configuration
- Testing network connectivity to Hugging Face
- Debugging authentication issues
- Validating user permissions

Usage:
    python check_hf.py

Requirements:
    - HUGGING_FACE_HUB_TOKEN in .env file
    - Internet connection
    - Valid Hugging Face account

Author: CesarChaMal
License: MIT
"""

# =============================================================================
# IMPORTS AND DEPENDENCIES
# =============================================================================

import os
from dotenv import load_dotenv  # Environment variable management
from huggingface_hub import HfApi  # Hugging Face API client

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# HUGGING FACE CONNECTION TESTING
# =============================================================================

def check_hf_connection():
    """
    Test connection to Hugging Face Hub and validate authentication.
    
    This function performs comprehensive validation of:
    - Environment token configuration
    - API connectivity and authentication
    - User account information retrieval
    - Permission validation
    
    Returns:
        bool: True if connection successful, False otherwise
        
    Raises:
        None: All exceptions are caught and handled gracefully
    """
    
    # =============================================================================
    # TOKEN VALIDATION
    # =============================================================================
    
    print("🔍 Checking Hugging Face Hub configuration...")
    
    # Retrieve token from environment
    token = os.getenv('HUGGING_FACE_HUB_TOKEN')
    
    if not token:
        print("❌ No HUGGING_FACE_HUB_TOKEN found in environment")
        print("\n💡 To fix this:")
        print("   1. Get a token from: https://huggingface.co/settings/tokens")
        print("   2. Add to .env file: HUGGING_FACE_HUB_TOKEN=hf_your_token_here")
        print("   3. Ensure token has read/write permissions")
        return False
    
    # Validate token format
    if not token.startswith('hf_'):
        print("⚠️  Token format looks incorrect (should start with 'hf_')")
        print("Please verify your Hugging Face token")
    
    print(f"✅ Token found: {token[:10]}...{token[-4:]}")
    
    # =============================================================================
    # API CONNECTION TEST
    # =============================================================================
    
    try:
        print("🌐 Testing connection to Hugging Face Hub...")
        
        # Initialize API client
        api = HfApi()
        
        # Test authentication and get user info
        user_info = api.whoami(token=token)
        
        # Display connection success and user details
        print(f"✅ Successfully connected to Hugging Face Hub!")
        print(f"\n👤 User Information:")
        print(f"   • Username: {user_info['name']}")
        print(f"   • Email: {user_info.get('email', 'Not provided')}")
        print(f"   • Account Type: {user_info.get('type', 'user')}")
        
        # Check for organizations if available
        if 'orgs' in user_info and user_info['orgs']:
            print(f"   • Organizations: {', '.join([org['name'] for org in user_info['orgs']])}")
        
        # Test basic API operations
        print("\n🔧 Testing API permissions...")
        
        # Test if we can list user's repositories (basic read permission)
        try:
            repos = list(api.list_repos(token=token, limit=1))
            print("✅ Read permissions: OK")
        except Exception as e:
            print(f"⚠️  Read permissions: Limited ({e})")
        
        print("\n🎉 Hugging Face Hub connection test completed successfully!")
        print("💡 You can now upload datasets and models to Hugging Face")
        
        return True
        
    except Exception as e:
        # Handle various connection and authentication errors
        error_msg = str(e).lower()
        
        print(f"❌ Failed to connect to Hugging Face Hub")
        print(f"\n🔍 Error details: {e}")
        
        # Provide specific troubleshooting based on error type
        if "401" in error_msg or "unauthorized" in error_msg:
            print("\n💡 Authentication Error - Possible solutions:")
            print("   • Verify your token is correct")
            print("   • Check token hasn't expired")
            print("   • Ensure token has proper permissions")
            
        elif "403" in error_msg or "forbidden" in error_msg:
            print("\n💡 Permission Error - Possible solutions:")
            print("   • Check token permissions (read/write)")
            print("   • Verify account is in good standing")
            
        elif "network" in error_msg or "connection" in error_msg:
            print("\n💡 Network Error - Possible solutions:")
            print("   • Check internet connection")
            print("   • Verify firewall/proxy settings")
            print("   • Try again in a few minutes")
            
        else:
            print("\n💡 General troubleshooting:")
            print("   • Check https://status.huggingface.co/ for service status")
            print("   • Verify token format (should start with 'hf_')")
            print("   • Try regenerating your token")
        
        return False

# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Main entry point for the Hugging Face connection tester.
    
    Provides a simple command-line interface for testing HF Hub connectivity.
    """
    print("🤗 Hugging Face Hub Connection Tester")
    print("=" * 40)
    
    try:
        success = check_hf_connection()
        
        if success:
            print("\n✨ All tests passed! Ready to use Hugging Face Hub.")
        else:
            print("\n❌ Connection test failed. Please check the issues above.")
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print("Please report this issue if it persists")