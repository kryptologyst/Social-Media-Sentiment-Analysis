# Social Media Sentiment Analysis

A comprehensive machine learning project that analyzes the emotional tone of social media posts using Natural Language Processing and supervised learning techniques.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-v2.3.3-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.3.0-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Features

- **Real-time Sentiment Analysis**: Classify social media posts as Positive, Negative, or Neutral
- **Interactive Web Interface**: Beautiful, responsive UI for easy interaction
- **Batch Processing**: Analyze multiple posts simultaneously
- **Confidence Scoring**: Get probability scores for each sentiment class
- **Model Persistence**: Save and load trained models
- **Comprehensive Evaluation**: Detailed performance metrics and visualizations

## Demo

### Single Post Analysis
![Single Analysis Demo](docs/single_analysis.png)

### Batch Analysis
![Batch Analysis Demo](docs/batch_analysis.png)

## Model Performance

- **Accuracy**: ~85-90% on test data
- **Features**: TF-IDF vectorization with n-grams
- **Algorithm**: Logistic Regression with balanced class weights
- **Cross-validation**: 5-fold CV for robust evaluation

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-social-media.git
   cd sentiment-analysis-social-media
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:5000`

## 📁 Project Structure

```
sentiment-analysis-social-media/
├── app.py                          # Flask web application
├── requirements.txt                # Python dependencies
├── README.md                      # Project documentation
├── 0064.py                        # Original implementation
├── src/
│   └── sentiment_analyzer.py     # Core ML pipeline
├── data/
│   └── mock_social_media_data.py  # Mock data generator
├── templates/
│   └── index.html                 # Web interface
├── models/                        # Saved models (created at runtime)
├── results/                       # Analysis results (created at runtime)
└── docs/                          # Documentation assets
```

## 🔧 Usage

### Web Interface

1. **Single Post Analysis**
   - Enter a social media post in the text area
   - Click "Analyze Sentiment"
   - View results with confidence scores and probability breakdown

2. **Batch Analysis**
   - Enter multiple posts (one per line) in the batch text area
   - Click "Analyze Batch"
   - View results for all posts simultaneously

3. **Sample Posts**
   - Use the provided sample buttons to test different sentiment types

### Command Line Usage

```python
from src.sentiment_analyzer import SentimentAnalyzer

# Initialize analyzer
analyzer = SentimentAnalyzer()

# Load and train model
df = analyzer.load_data()
X, y = analyzer.preprocess_data(df)
analyzer.train_model(X, y)

# Make predictions
result = analyzer.predict_sentiment("I love this new feature!")
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.3f}")
```

## API Endpoints

### POST /analyze
Analyze sentiment of a single text.

**Request:**
```json
{
  "text": "I love this new feature!"
}
```

**Response:**
```json
{
  "success": true,
  "sentiment": "Positive",
  "confidence": 0.892,
  "probabilities": {
    "Positive": 0.892,
    "Negative": 0.054,
    "Neutral": 0.054
  }
}
```

### POST /batch_analyze
Analyze multiple texts at once.

**Request:**
```json
{
  "texts": [
    "I love this app!",
    "This is terrible!",
    "It's okay, nothing special."
  ]
}
```

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "text": "I love this app!",
      "sentiment": "Positive",
      "confidence": 0.892
    }
  ],
  "total_analyzed": 3
}
```

## Technical Details

### Machine Learning Pipeline

1. **Data Preprocessing**
   - Text cleaning and normalization
   - TF-IDF vectorization with stop word removal
   - N-gram features (unigrams and bigrams)

2. **Model Training**
   - Logistic Regression with balanced class weights
   - Cross-validation for model selection
   - Train/test split with stratification

3. **Evaluation Metrics**
   - Accuracy, Precision, Recall, F1-score
   - Confusion matrix visualization
   - Confidence intervals

### Features Used

- **TF-IDF Vectorization**: Captures word importance
- **N-grams**: Captures phrase-level sentiment
- **Stop Word Removal**: Reduces noise
- **Max Features**: 5000 most important features
- **Min/Max Document Frequency**: Filters rare/common words

## Model Evaluation

The model is evaluated using multiple metrics:

- **Classification Report**: Precision, recall, F1-score per class
- **Confusion Matrix**: Visual representation of predictions
- **Cross-validation**: 5-fold CV for robust performance estimation
- **Confidence Scores**: Probability distributions for predictions

## Use Cases

- **Social Media Monitoring**: Track brand sentiment
- **Customer Feedback Analysis**: Analyze product reviews
- **Market Research**: Understand public opinion
- **Content Moderation**: Identify negative content
- **Academic Research**: Study sentiment patterns

## Future Enhancements

- [ ] Deep learning models (LSTM, BERT)
- [ ] Real-time social media API integration
- [ ] Multi-language support
- [ ] Emotion detection (beyond sentiment)
- [ ] Advanced preprocessing (emoji handling, slang)
- [ ] Model comparison dashboard
- [ ] Export functionality for results

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **scikit-learn**: For machine learning algorithms
- **Flask**: For web framework
- **Bootstrap**: For responsive UI components
- **Font Awesome**: For beautiful icons

## Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **LinkedIn**: [Your LinkedIn](https://linkedin.com/in/yourprofile)



# Social-Media-Sentiment-Analysis
