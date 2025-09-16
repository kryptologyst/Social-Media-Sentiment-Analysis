#!/usr/bin/env python3
"""
Quick start script for the Social Media Sentiment Analysis application
"""

import os
import sys

def main():
    """Run the sentiment analysis application"""
    print("🚀 Starting Social Media Sentiment Analysis")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("❌ Error: app.py not found. Please run this script from the project root directory.")
        sys.exit(1)
    
    # Import and run the Flask app
    try:
        from app import app, initialize_analyzer
        
        # Initialize the analyzer
        if initialize_analyzer():
            print("🌐 Starting web server at http://localhost:5000")
            print("📱 Open your browser and navigate to the URL above")
            print("⏹️  Press Ctrl+C to stop the server")
            print("-" * 50)
            app.run(debug=False, host='0.0.0.0', port=5000)
        else:
            print("❌ Failed to initialize the sentiment analyzer")
            sys.exit(1)
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you've installed all dependencies: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
