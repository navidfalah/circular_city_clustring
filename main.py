import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
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

# Load feature data from CSV (assuming similar structure as before)
features_df = pd.read_csv('features.csv')
indicator_df = pd.read_csv('indicators.csv')  # Load indicator data from the new CSV

# Apply weights and preprocess text
texts = []
indicator_weights = {'Category': 1, 'Subcategory': 1, 'Indicator': 1, 'Description': 1}
for _, row in indicator_df.iterrows():
    text = ' '.join([preprocess_text(row['Category']) * indicator_weights['Category'], 
                     preprocess_text(row['Subcategory']) * indicator_weights['Subcategory'], 
                     preprocess_text(row['Indicator']) * indicator_weights['Indicator'], 
                     preprocess_text(row['Description']) * indicator_weights['Description']])
    texts.append(text)

# Apply weights to features and preprocess text
feature_weights = {'Dimension': 1, 'Core Features': 1, 'Examples of Indicators': 1, 'Key Initiatives': 1}
for _, row in features_df.iterrows():
    text = ' '.join([preprocess_text(row['Dimension']) * feature_weights['Dimension'], 
                     preprocess_text(row['Core Features']) * feature_weights['Core Features'], 
                     preprocess_text(row['Examples of Indicators']) * feature_weights['Examples of Indicators'], 
                     preprocess_text(row['Key Initiatives']) * feature_weights['Key Initiatives']])
    texts.append(text)

# Vectorization and NMF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=1, ngram_range=(1, 2))
X = vectorizer.fit_transform(texts)

nmf = NMF(n_components=3, random_state=0, init='nndsvd')
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

# Based on the top terms, manually assign the cluster names
cluster_names = ['Economic', 'Environmental', 'Social']

# Calculate intersections for Venn diagram
venn_labels = {
    '100': sum((memberships[:, 0] == 1) & (memberships[:, 1] == 0) & (memberships[:, 2] == 0)),
    '010': sum((memberships[:, 0] == 0) & (memberships[:, 1] == 1) & (memberships[:, 2] == 0)),
    '001': sum((memberships[:, 0] == 0) & (memberships[:, 1] == 0) & (memberships[:, 2] == 1)),
    '110': sum((memberships[:, 0] == 1) & (memberships[:, 1] == 1) & (memberships[:, 2] == 0)),
    '101': sum((memberships[:, 0] == 1) & (memberships[:, 1] == 0) & (memberships[:, 2] == 1)),
    '011': sum((memberships[:, 0] == 0) & (memberships[:, 1] == 1) & (memberships[:, 2] == 1)),
    '111': sum((memberships[:, 0] == 1) & (memberships[:, 1] == 1) & (memberships[:, 2] == 1))
}

# Display counts of indicators per cluster
print("Counts of indicators in each cluster:")
for name, idx in zip(cluster_names, range(len(cluster_names))):
    print(f"{name}: {sum(memberships[:, idx] == 1)}")

# Directory to save plots
plot_dir = 'plots'
os.makedirs(plot_dir, exist_ok=True)

# Venn Diagram visualization
plt.figure(figsize=(10, 8))
venn3(subsets=venn_labels, set_labels=cluster_names)
plt.title("Venn Diagram of Indicator Clusters")
plt.savefig(os.path.join(plot_dir, 'venn_diagram.png'))
plt.close()

# Optional: Display the top terms per cluster with assigned names
print("\nTop terms per cluster with assigned names:")
for name, comp in zip(cluster_names, nmf.components_):
    terms_comp = zip(features, comp)
    sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:top_terms]
    print(f"Cluster {name}: " + ", ".join([t[0] for t in sorted_terms]))

# Additional Statistics and Visualizations

# Heatmap of Cluster Memberships with group names
plt.figure(figsize=(12, 6))
sns.heatmap(W_normalized, cmap='coolwarm', cbar=True, xticklabels=cluster_names, yticklabels=False)
plt.title('Heatmap of Cluster Memberships')
plt.xlabel('Clusters')
plt.ylabel('Documents')
plt.savefig(os.path.join(plot_dir, 'heatmap_cluster_memberships.png'))
plt.close()

# Boxplot of Cluster Membership Strengths with group names
plt.figure(figsize=(10, 6))
sns.boxplot(data=W_normalized)
plt.title('Boxplot of Cluster Membership Strengths')
plt.xlabel('Clusters')
plt.ylabel('Membership Strength')
plt.xticks(ticks=range(len(cluster_names)), labels=cluster_names)
plt.savefig(os.path.join(plot_dir, 'boxplot_membership_strengths.png'))
plt.close()

# Pie chart of Cluster Distribution with group names
cluster_counts = [sum(memberships[:, i] == 1) for i in range(len(cluster_names))]
plt.figure(figsize=(8, 8))
plt.pie(cluster_counts, labels=cluster_names, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("coolwarm", len(cluster_names)))
plt.title('Pie Chart of Cluster Distribution')
plt.savefig(os.path.join(plot_dir, 'pie_chart_cluster_distribution.png'))
plt.close()
