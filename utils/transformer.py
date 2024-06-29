import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder  # Added import here
import nltk

# Download necessary NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

# Setup tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = str(text).lower() if not pd.isna(text) else ''
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and word.isalnum()]
    return ' '.join(words)

def get_embeddings(texts):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze().numpy())
    return np.array(embeddings)

# Load and preprocess data
features_df = load_data('features.csv')
indicator_df = load_data('indicators.csv')

# Combine texts from 'Examples of Indicators' and 'Key Initiatives' and preprocess
features_df['combined_text'] = features_df['Examples of Indicators'] + " " + features_df['Key Initiatives']
features_df['processed_text'] = features_df['combined_text'].apply(preprocess_text)
features_embeddings = get_embeddings(features_df['processed_text'])

# Assuming 'Core Features' column is used to define clusters or as a label
if 'Core Features' in features_df:
    # Mapping core features to numeric labels if necessary
    label_encoder = LabelEncoder()
    features_df['labels'] = label_encoder.fit_transform(features_df['Core Features'])
    kmeans = KMeans(n_clusters=len(features_df['labels'].unique()), random_state=0).fit(features_embeddings, features_df['labels'])
else:
    kmeans = KMeans(n_clusters=3, random_state=0).fit(features_embeddings)

# Process and classify indicators
indicator_df['processed_text'] = indicator_df['Indicator'].apply(preprocess_text)
indicator_embeddings = get_embeddings(indicator_df['processed_text'])
indicator_labels = kmeans.predict(indicator_embeddings)

# Add labels to the DataFrame and print results
indicator_df['Cluster'] = indicator_labels
print(indicator_df[['Indicator', 'Cluster']])

# Calculate and print silhouette score
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(indicator_embeddings, indicator_labels)
print(f"Silhouette Score: {silhouette_avg:.2f}")
