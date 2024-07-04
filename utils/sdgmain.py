import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Downloads for NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    if not isinstance(text, str):
        text = ''
    tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
    return ' '.join([word for word in lemmatized if word not in stopwords.words('english')])

# Load the indicator data
indicator_df = pd.read_csv('indicators.csv')

# Load the SDG 11 indicators
sdg11_df = pd.read_csv('sdg11.csv')

# Preprocess the texts from the indicators
texts = []
indicator_weights = {'Category': 1, 'Subcategory': 1, 'Indicator': 1, 'Description': 1}
for _, row in indicator_df.iterrows():
    text = ' '.join([preprocess_text(row['Category']) * indicator_weights['Category'], 
                     preprocess_text(row['Subcategory']) * indicator_weights['Subcategory'], 
                     preprocess_text(row['Indicator']) * indicator_weights['Indicator'], 
                     preprocess_text(row['Description']) * indicator_weights['Description']])
    texts.append(text)

# Check if texts are not empty
if not texts:
    raise ValueError("All preprocessed texts are empty. Check the preprocessing steps and input data.")

# Vectorization and NMF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=1, ngram_range=(1, 2))
X = vectorizer.fit_transform(texts)

nmf = NMF(n_components=16, random_state=0, init='nndsvd')
W = nmf.fit_transform(X)
W_normalized = normalize(W, norm='l1', axis=1)

# Determine cluster memberships with a threshold
threshold = 0.05  # Lower the threshold to include more indicators
memberships = (W_normalized > threshold).astype(int)

# Ensure that each indicator is assigned to at least one cluster
for i in range(memberships.shape[0]):
    if not memberships[i].any():
        memberships[i, W_normalized[i].argmax()] = 1

# Display the top terms per cluster to help identify appropriate names
features = vectorizer.get_feature_names_out()
top_terms = 10
print("\nTop terms per cluster:")
for i, comp in enumerate(nmf.components_):
    terms_comp = zip(features, comp)
    sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:top_terms]
    print(f"Cluster {i+1}: " + ", ".join([t[0] for t in sorted_terms]))

# Based on the top terms, assign the cluster names if desired
cluster_names = [f'Class {i+1}' for i in range(16)]

# Display counts of indicators per cluster
print("Counts of indicators in each cluster:")
for name, idx in zip(cluster_names, range(len(cluster_names))):
    print(f"{name}: {sum(memberships[:, idx] == 1)}")

# Directory to save plots
plot_dir = 'plots'
os.makedirs(plot_dir, exist_ok=True)

# Heatmap showing relationship between indicators and SDG 11 clusters
plt.figure(figsize=(12, 8))
sns.heatmap(W_normalized, cmap='coolwarm', xticklabels=cluster_names, yticklabels=indicator_df['Indicator'], cbar=True)
plt.title('Heatmap of Indicators vs. SDG 11 Clusters')
plt.xlabel('SDG 11 Clusters')
plt.ylabel('Indicators')
plt.savefig(os.path.join(plot_dir, 'heatmap_indicators_vs_sdg11.png'))
plt.close()

# Boxplot of Cluster Membership Strengths with group names
plt.figure(figsize=(12, 8))
sns.boxplot(data=W_normalized)
plt.title('Boxplot of Cluster Membership Strengths')
plt.xlabel('Clusters')
plt.ylabel('Membership Strength')
plt.xticks(ticks=range(len(cluster_names)), labels=cluster_names, rotation=45)
plt.savefig(os.path.join(plot_dir, 'boxplot_membership_strengths.png'))
plt.close()

# Pie chart of Cluster Distribution with group names
cluster_counts = [sum(memberships[:, i] == 1) for i in range(len(cluster_names))]
plt.figure(figsize=(12, 12))
plt.pie(cluster_counts, labels=cluster_names, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("coolwarm", len(cluster_names)))
plt.title('Pie Chart of Cluster Distribution')
plt.savefig(os.path.join(plot_dir, 'pie_chart_cluster_distribution.png'))
plt.close()

cluster_data = pd.DataFrame(memberships, columns=cluster_names)
# Add the 'Indicator' column from your 'indicator_df' to the cluster_data DataFrame
cluster_data.insert(0, 'Indicator', indicator_df['Indicator'])

# Export to CSV
output_csv_path = 'cluster_assignments_16_classes.csv'
cluster_data.to_csv(output_csv_path, index=False)
print(f"Cluster assignments for 16 classes saved to {output_csv_path}")
