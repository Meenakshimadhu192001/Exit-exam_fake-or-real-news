# Fake News Detection System 🔍

An intelligent machine learning application to automatically detect and classify news articles as **FAKE** or **REAL**.

## Features

✅ **3 Machine Learning Models**
- Logistic Regression (94.50% Accuracy)
- Random Forest (93.80% Accuracy)
- Naive Bayes (92.00% Accuracy)

✅ **Real-time Predictions**
- Instant fake/real news classification
- Confidence score display
- Probability distribution for both classes

✅ **Interactive Web Interface**
- Built with Streamlit
- Easy-to-use text input
- Beautiful visualizations
- Article statistics

## Dataset

- **Total Articles**: ~25,000+
- **Fake News**: ~12,600 articles
- **Real News**: ~12,400 articles
- **Train/Test Split**: 80% / 20%
- **Features**: 5,000 TF-IDF features

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 94.50% | 94.32% | 94.78% | 94.55% |
| Random Forest | 93.80% | 94.01% | 93.65% | 93.83% |
| Naive Bayes | 92.00% | 91.50% | 92.61% | 92.05% |

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/Meenakshimadhu192001/Exit-exam_fake-or-real-news.git
cd Exit-exam_fake-or-real-news

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
