import pandas as pd
import numpy as np
import re
import nltk
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ============================================================
# DATA ACQUISITION AND INITIALIZATION
# ============================================================

# The exact name of the official Yelp academic dataset file
file_name = "yelp_academic_dataset_review.json"

# Verification of file existence in the working directory to prevent NameErrors
if not os.path.exists(file_name):
    print(f"CRITICAL ERROR: The file '{file_name}' was not found.")
    print("Action Required: Place the JSON file in the same directory as this script.")
    sys.exit()

try:
    # Ingesting the JSON-Lines dataset with a row limit for memory stability
    # lines=True is mandatory for the Yelp dataset structure
    df = pd.read_json(file_name, lines=True, nrows=200000)
    print(f"Successfully loaded {len(df)} records from {file_name}.")
except Exception as e:
    print(f"Data ingestion error: {e}")
    sys.exit()

# ============================================================
# DATA PREPROCESSING AND FEATURE ENGINEERING
# ============================================================

# Cleaning the dataset by removing null entries in text and rating columns
df = df.dropna(subset=['text', 'stars'])

# Binary sentiment labeling: Positive (1) for 4-5 stars, Negative (0) for 1-3 stars
df["sentiment"] = df["stars"].apply(lambda x: 1 if x >= 4 else 0)

# Initialization of NLP components for text normalization
nltk.download('stopwords', quiet=True)
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def process_text_data(text):
    """
    Standardizes text through noise reduction, tokenization, and stemming.
    """
    # Filtering non-alphabetic characters using regular expressions
    text = re.sub(r'[^a-zA-Z]', ' ', str(text).lower())
    words = text.split()

    # Removing stop-words and applying the Porter Stemmer algorithm
    cleaned_words = [stemmer.stem(w) for w in words if w not in stop_words]
    return ' '.join(cleaned_words)


# Implementing the normalization pipeline across the text corpus
print("Executing text normalization and feature preparation...")
df["clean_text"] = df["text"].apply(process_text_data)

# ============================================================
# VECTORIZATION AND FEATURE SPACE REPRESENTATION
# ============================================================

# Converting textual data into numerical TF-IDF vectors
# Dimensionality is limited to 5000 features to optimize computational performance
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df["clean_text"])
y = df["sentiment"]

# Partitioning the feature space into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================================
# MODEL SELECTION AND GRADIENT OPTIMIZATION
# ============================================================

# Utilizing Stochastic Gradient Descent (SGD) for large-scale classification
# This model provides high efficiency for big data iterations
model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)
model.fit(X_train, y_train)

# ============================================================
# ASPECT MINING THROUGH UNSUPERVISED LEARNING
# ============================================================

# Applying Latent Dirichlet Allocation (LDA) to extract latent review aspects
print("Performing unsupervised aspect extraction...")
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X_train)


def log_top_aspect_terms(model, vectorizer, n_words):
    """Logs the top significant terms for each identified aspect."""
    words = vectorizer.get_feature_names_out()
    for index, topic in enumerate(model.components_):
        print(f"Aspect {index + 1}:")
        print([words[i] for i in topic.argsort()[:-n_words - 1:-1]])


log_top_aspect_terms(lda, vectorizer, 10)

# ============================================================
# PERFORMANCE QUANTIFICATION AND EVALUATION
# ============================================================

# Validating the model performance on the hold-out test set
y_pred = model.predict(X_test)
print(f"\nModel Accuracy Index: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Evaluation Report:\n", classification_report(y_test, y_pred))

# ============================================================
# TEMPORAL TREND ANALYSIS
# ============================================================

# Visualizing sentiment trends across the temporal axis
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    # Aggregating average sentiment indices by year
    sentiment_trend = df.groupby("year")["sentiment"].mean()

    plt.figure(figsize=(12, 6))
    plt.plot(sentiment_trend.index, sentiment_trend.values, marker='o', color='#2c3e50')
    plt.title("Longitudinal Analysis of Customer Sentiment")
    plt.xlabel("Year")
    plt.ylabel("Mean Sentiment Index")
    plt.grid(True, alpha=0.3)
    plt.show()


# ============================================================
# INFERENCE ENGINE FOR PREDICTION
# ============================================================

def infer_sentiment(review_instance):
    """Processes raw input to predict sentiment polarity."""
    normalized_input = process_text_data(review_instance)
    feature_vector = vectorizer.transform([normalized_input])
    prediction_output = model.predict(feature_vector)
    return "Positive" if prediction_output[0] == 1 else "Negative"


# Verification of system inference capability
print("\nSystem Inference Test:")
sample_text = "The quality was exceptional and the delivery was prompt."
print(f"Sample Input: {sample_text}")
print(f"Inference Result: {infer_sentiment(sample_text)}")