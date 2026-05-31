 Large-Scale Sentiment and Topic Analysis of Reviews

 Project Overview

This project performs large-scale text analytics on customer reviews from the Yelp Academic Dataset. The objective is to transform unstructured textual feedback into actionable business insights through sentiment analysis, topic modeling, exploratory data analysis, and predictive classification.

The project processes 200,000 Yelp reviews and applies a complete analytics workflow including data preprocessing, class balancing, feature engineering, sentiment classification, topic extraction, and performance evaluation.



 Dataset

Source: Yelp Academic Dataset

Dataset Characteristics:

* Customer review text
* Star ratings (1–5)
* Review dates

A total of 200,000 review records were analyzed.

Sentiment labels were generated as follows:

* Positive = Rating ≥ 4
* Negative = Rating < 4

To reduce class imbalance, undersampling was applied, resulting in a balanced dataset of 121,372 reviews.



 Project Workflow

 1. Data Preparation

* Data quality auditing
* Missing value handling
* Sentiment labeling
* Class balancing using undersampling

 2. Text Preprocessing

* Text normalization
* Tokenization
* Stop-word removal
* Porter stemming

 3. Feature Engineering

TF-IDF Vectorization was used to transform textual reviews into numerical features.

Configuration:

* Maximum Features: 3000
* N-Grams: Unigrams and Bigrams

 4. Exploratory Data Analysis

Visualizations include:

* Balanced sentiment distribution
* Word cloud of frequent terms in positive reviews

 5. Sentiment Classification

Two machine learning models were trained and evaluated:

* SGD Classifier
* Random Forest Classifier

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC Curve
* AUC

 6. Topic Modeling

Latent Dirichlet Allocation (LDA) was applied to identify hidden discussion topics and operational aspects within customer reviews.

 7. Trend Analysis

Customer sentiment was aggregated across years to identify long-term behavioral patterns and sentiment trends.


 Key Outputs

* Balanced sentiment distribution analysis
* Positive review word cloud
* Topic extraction using LDA
* Confusion matrices
* ROC comparison curves
* Sentiment trend visualization
* Classification performance reports


 Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* Scikit-Learn
* Matplotlib
* Seaborn
* WordCloud


 Results

The analysis demonstrates that linear models such as SGD Classifier perform more effectively on high-dimensional sparse text representations than tree-based models such as Random Forest.

Topic modeling successfully identified recurring business themes related to customer experience, service quality, food quality, hospitality, and operational performance.

The project also provides sentiment trend analysis that can support business monitoring and customer experience evaluation.


 Repository Contents

* Source Code
* Project Report (PDF)
* Visualizations and Results
* Sample Outputs
