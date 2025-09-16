"""
Sentiment Analysis for Social Media Posts
Enhanced ML pipeline with proper error handling and evaluation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class SentimentAnalyzer:
    """
    A comprehensive sentiment analysis class for social media posts
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        self.label_map = {}
        self.is_trained = False
        
    def load_data(self, data_source=None):
        """Load data from various sources"""
        try:
            if data_source is None:
                # Use mock data generator
                from data.mock_social_media_data import generate_mock_data
                df = generate_mock_data()
                print(f"Loaded {len(df)} posts from mock data generator")
            elif isinstance(data_source, str) and data_source.endswith('.csv'):
                df = pd.read_csv(data_source)
                print(f"Loaded {len(df)} posts from {data_source}")
            elif isinstance(data_source, pd.DataFrame):
                df = data_source.copy()
                print(f"Loaded {len(df)} posts from DataFrame")
            else:
                raise ValueError("Invalid data source")
                
            # Validate required columns
            if 'Post' not in df.columns or 'Sentiment' not in df.columns:
                raise ValueError("Data must contain 'Post' and 'Sentiment' columns")
                
            # Clean data
            df = df.dropna()
            df['Post'] = df['Post'].astype(str)
            
            print(f"Data distribution:")
            print(df['Sentiment'].value_counts())
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def preprocess_data(self, df):
        """Preprocess the data for training"""
        try:
            # Encode labels
            df['Sentiment_Label'] = df['Sentiment'].astype('category').cat.codes
            self.label_map = dict(enumerate(df['Sentiment'].astype('category').cat.categories))
            
            # Text vectorization
            X = self.vectorizer.fit_transform(df['Post'])
            y = df['Sentiment_Label']
            
            print(f"Features extracted: {X.shape[1]} TF-IDF features")
            print(f"Label mapping: {self.label_map}")
            
            return X, y
            
        except Exception as e:
            print(f"Error preprocessing data: {str(e)}")
            return None, None
    
    def train_model(self, X, y, test_size=0.2):
        """Train the sentiment analysis model"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            print(f"Training set size: {X_train.shape[0]}")
            print(f"Test set size: {X_test.shape[0]}")
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
            print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            # Evaluate on test set
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Test accuracy: {accuracy:.3f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=self.label_map.values()))
            
            self.is_trained = True
            return X_test, y_test, y_pred
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return None, None, None
    
    def plot_confusion_matrix(self, y_test, y_pred, save_path=None):
        """Plot and optionally save confusion matrix"""
        try:
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.label_map.values(), 
                       yticklabels=self.label_map.values())
            plt.title("Sentiment Analysis - Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Confusion matrix saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error plotting confusion matrix: {str(e)}")
    
    def predict_sentiment(self, text):
        """Predict sentiment for a single text"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Vectorize text
            text_vector = self.vectorizer.transform([text])
            
            # Predict
            prediction = self.model.predict(text_vector)[0]
            probability = self.model.predict_proba(text_vector)[0]
            
            sentiment = self.label_map[prediction]
            confidence = max(probability)
            
            return {
                'text': text,
                'sentiment': sentiment,
                'confidence': confidence,
                'probabilities': {self.label_map[i]: prob for i, prob in enumerate(probability)}
            }
            
        except Exception as e:
            print(f"Error predicting sentiment: {str(e)}")
            return None
    
    def save_model(self, model_path='models/sentiment_model.pkl'):
        """Save the trained model"""
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            model_data = {
                'vectorizer': self.vectorizer,
                'model': self.model,
                'label_map': self.label_map,
                'is_trained': self.is_trained
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"Model saved to {model_path}")
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
    
    def load_model(self, model_path='models/sentiment_model.pkl'):
        """Load a trained model"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.vectorizer = model_data['vectorizer']
            self.model = model_data['model']
            self.label_map = model_data['label_map']
            self.is_trained = model_data['is_trained']
            
            print(f"Model loaded from {model_path}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")

def main():
    """Main function to demonstrate the sentiment analyzer"""
    print("🚀 Social Media Sentiment Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Load data
    df = analyzer.load_data()
    if df is None:
        return
    
    # Preprocess data
    X, y = analyzer.preprocess_data(df)
    if X is None:
        return
    
    # Train model
    X_test, y_test, y_pred = analyzer.train_model(X, y)
    if X_test is None:
        return
    
    # Plot confusion matrix
    analyzer.plot_confusion_matrix(y_test, y_pred, 'results/confusion_matrix.png')
    
    # Save model
    analyzer.save_model()
    
    # Test predictions
    print("\n🔍 Testing Predictions:")
    print("-" * 30)
    
    test_posts = [
        "I absolutely love this new feature! It's amazing!",
        "This app is terrible and crashes all the time!",
        "The update is okay, nothing special though.",
        "Best app ever! Highly recommend to everyone!",
        "Worst experience ever, completely disappointed!"
    ]
    
    for post in test_posts:
        result = analyzer.predict_sentiment(post)
        if result:
            print(f"Text: {result['text'][:50]}...")
            print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.3f})")
            print("-" * 30)

if __name__ == "__main__":
    main()
