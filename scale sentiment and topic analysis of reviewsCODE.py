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
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from wordcloud import WordCloud

# ============================================================
# 1. DATA ACQUISITION & INTEGRITY AUDIT
# ============================================================
file_name = "yelp_academic_dataset_review.json"

if not os.path.exists(file_name):
    print(f"CRITICAL ERROR: The file '{file_name}' was not found.")
    sys.exit()

try:
    # Reading 200,000 records for robust analysis
    print("Step 1: Ingesting Dataset and Auditing Data Quality...")
    df = pd.read_json(file_name, lines=True, nrows=200000)

    # Initial Integrity Check
    print("\n[Data Integrity Audit - Initial Missing Values]")
    print(df[['text', 'stars']].isnull().sum())

    # Cleaning: Removing incomplete records
    df = df.dropna(subset=['text', 'stars'])

    # Labeling Sentiment (0: Negative, 1: Positive)
    df["sentiment"] = df["stars"].apply(lambda x: 1 if x >= 4 else 0)

    # ------------------------------------------------------------
    # NEW ADDITION: CLASS BALANCING (UNDERSAMPLING)
    # This prevents the model from being biased towards Positive reviews
    # ------------------------------------------------------------
    print("\nApplying Class Balancing Strategy (Undersampling)...")
    negative_count = len(df[df['sentiment'] == 0])
    positive_indices = df[df['sentiment'] == 1].index

    # Randomly select positive samples to match negative sample count
    random_positive_indices = np.random.choice(positive_indices, negative_count, replace=False)
    negative_indices = df[df['sentiment'] == 0].index

    # Combine and shuffle
    balanced_indices = np.concatenate([negative_indices, random_positive_indices])
    df = df.loc[balanced_indices].sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Final Balanced Dataset Size: {len(df)} records ({negative_count} per class).")
    # ------------------------------------------------------------

except Exception as e:
    print(f"Data ingestion error: {e}")
    sys.exit()

# ============================================================
# 2. DATA PREPROCESSING & FEATURE ENGINEERING
# ============================================================
nltk.download('stopwords', quiet=True)
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def process_text_data(text):
    """
    Normalizes text through noise reduction, tokenization, and stemming.
    """
    # Remove symbols and convert to lowercase
    text = re.sub(r'[^a-zA-Z]', ' ', str(text).lower())
    # Split into words, remove stop words, and apply stemming
    words = text.split()
    cleaned_words = [stemmer.stem(w) for w in words if w not in stop_words]
    return ' '.join(cleaned_words)

print("\nStep 2: Executing Text Normalization Pipeline...")
df["clean_text"] = df["text"].apply(process_text_data)

# ============================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA VISUALS)
# ============================================================
print("Step 3: Generating Exploratory Visualizations...")

# Visual 1: Balanced Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='sentiment', data=df, hue='sentiment', palette='viridis', legend=False)
plt.title('Balanced Sentiment Distribution (Target Classes)')
plt.show()

# Visual 2: Word Cloud for Positive Reviews
pos_text = ' '.join(df[df['sentiment']==1]['clean_text'].head(1000))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(pos_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('High-Frequency Terms in Positive Corpus (After Balancing)')
plt.axis('off')
plt.show()

# ============================================================
# 4. VECTORIZATION & MODEL COMPARISON
# ============================================================
print("Step 4: Vectorizing Text and Training Comparative Models...")
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df["clean_text"])
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training SGD Classifier
model_sgd = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)
model_sgd.fit(X_train, y_train)

# Training Random Forest Classifier
model_rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
model_rf.fit(X_train, y_train)

# ============================================================
# 5. UNSUPERVISED ASPECT MINING (LDA)
# ============================================================
print("\nStep 5: Performing Unsupervised Aspect Extraction (LDA)...")
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X_train)

def log_top_aspect_terms(model, vectorizer, n_words):
    words = vectorizer.get_feature_names_out()
    for index, topic in enumerate(model.components_):
        print(f"Aspect {index + 1}:")
        print([words[i] for i in topic.argsort()[:-n_words - 1:-1]])

log_top_aspect_terms(lda, vectorizer, 10)

# ============================================================
# 6. FINAL PERFORMANCE EVALUATION & ROC ANALYSIS
# ============================================================
print("\nStep 6: Generating Performance Analytics...")

models = [('SGD Classifier', model_sgd), ('Random Forest', model_rf)]
plt.figure(figsize=(8, 6)) # Main ROC Figure

for name, model in models:
    y_pred = model.predict(X_test)
    print(f"\nResults for {name}:")
    print(f"Accuracy Index: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    # Heatmap Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='PuBuGn')
    plt.title(f'Balanced Confusion Matrix: {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # ROC Curve Data
    if hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_score)
    plt.figure(3) # Comparison Plot
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.2f})')

# Finalizing the Comparative ROC Plot
plt.figure(3)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Comparison - Balanced Data')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

print("\n--- Project Execution Complete (Final Balanced Version) ---")