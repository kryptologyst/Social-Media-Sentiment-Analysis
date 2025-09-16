"""
Flask Web Application for Social Media Sentiment Analysis
Interactive UI for real-time sentiment prediction
"""

from flask import Flask, render_template, request, jsonify
import sys
import os
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from sentiment_analyzer import SentimentAnalyzer

app = Flask(__name__)

# Global analyzer instance
analyzer = None

def initialize_analyzer():
    """Initialize and train the sentiment analyzer"""
    global analyzer
    try:
        analyzer = SentimentAnalyzer()
        
        # Try to load existing model first
        model_path = 'models/sentiment_model.pkl'
        if os.path.exists(model_path):
            analyzer.load_model(model_path)
            print("✅ Loaded existing model")
        else:
            # Train new model
            print("🔄 Training new model...")
            df = analyzer.load_data()
            if df is not None:
                X, y = analyzer.preprocess_data(df)
                if X is not None:
                    analyzer.train_model(X, y)
                    analyzer.save_model(model_path)
                    print("✅ Model trained and saved")
                else:
                    print("❌ Failed to preprocess data")
                    return False
            else:
                print("❌ Failed to load data")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error initializing analyzer: {str(e)}")
        return False

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    """Analyze sentiment of submitted text"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Please enter some text to analyze'}), 400
        
        if not analyzer or not analyzer.is_trained:
            return jsonify({'error': 'Model not initialized'}), 500
        
        # Get prediction
        result = analyzer.predict_sentiment(text)
        
        if result:
            return jsonify({
                'success': True,
                'sentiment': result['sentiment'],
                'confidence': round(result['confidence'], 3),
                'probabilities': {k: round(v, 3) for k, v in result['probabilities'].items()}
            })
        else:
            return jsonify({'error': 'Failed to analyze sentiment'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    """Analyze multiple texts at once"""
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({'error': 'Please provide texts to analyze'}), 400
        
        if not analyzer or not analyzer.is_trained:
            return jsonify({'error': 'Model not initialized'}), 500
        
        results = []
        for text in texts:
            if text.strip():
                result = analyzer.predict_sentiment(text.strip())
                if result:
                    results.append({
                        'text': text,
                        'sentiment': result['sentiment'],
                        'confidence': round(result['confidence'], 3)
                    })
        
        return jsonify({
            'success': True,
            'results': results,
            'total_analyzed': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': f'Batch analysis failed: {str(e)}'}), 500

@app.route('/model_info')
def model_info():
    """Get information about the current model"""
    try:
        if not analyzer:
            return jsonify({'error': 'Model not initialized'}), 500
        
        return jsonify({
            'is_trained': analyzer.is_trained,
            'labels': list(analyzer.label_map.values()) if analyzer.label_map else [],
            'features': analyzer.vectorizer.get_feature_names_out().shape[0] if analyzer.is_trained else 0
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to get model info: {str(e)}'}), 500

if __name__ == '__main__':
    print("🚀 Starting Social Media Sentiment Analysis App")
    print("=" * 50)
    
    # Initialize the analyzer
    if initialize_analyzer():
        print("🌐 Starting web server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to initialize. Please check the logs above.")
