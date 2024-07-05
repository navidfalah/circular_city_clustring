import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Read CSV files
indicators_df = pd.read_csv('indicators.csv')
classes_df = pd.read_csv('sdg11.csv')

# Step 2: Preprocess text
def preprocess_text(text):
    return ' '.join(word.lower() for word in str(text).split() if word.isalnum())

indicators_df['Processed_Indicator'] = indicators_df['Indicator'].apply(preprocess_text)
classes_df['Processed_Indicator'] = classes_df['Indicator'].apply(preprocess_text)

# Step 3: Create TF-IDF vectors
tfidf = TfidfVectorizer(stop_words='english')
indicators_tfidf = tfidf.fit_transform(indicators_df['Processed_Indicator'])
classes_tfidf = tfidf.transform(classes_df['Processed_Indicator'])

# Step 4: Perform K-means clustering
n_clusters = 16  # We have 16 predefined classes
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(indicators_tfidf)

# Step 5: Assign cluster labels to indicators
indicators_df['Cluster'] = cluster_labels

# Step 6: Find the closest class for each cluster
cluster_centers = kmeans.cluster_centers_
cluster_class_similarity = cosine_similarity(cluster_centers, classes_tfidf)
closest_class_per_cluster = classes_df.iloc[cluster_class_similarity.argmax(axis=1)]['Indicator'].values

# Step 7: Assign class labels to indicators based on their cluster
indicators_df['Assigned_Class'] = indicators_df['Cluster'].map(lambda x: closest_class_per_cluster[x])

# Step 8: Evaluation
silhouette_avg = silhouette_score(indicators_tfidf, cluster_labels)
print(f"Silhouette Score: {silhouette_avg:.4f}")

# Step 9: Analyze cluster composition
for cluster in range(n_clusters):
    cluster_indicators = indicators_df[indicators_df['Cluster'] == cluster]
    print(f"\nCluster {cluster}:")
    print(f"  Size: {len(cluster_indicators)}")
    print(f"  Assigned Class: {closest_class_per_cluster[cluster]}")
    print("  Top terms:")
    cluster_tfidf = indicators_tfidf[cluster_indicators.index]
    top_terms_indices = cluster_tfidf.sum(axis=0).argsort()[0, -5:]
    top_terms = [tfidf.get_feature_names_out()[i] for i in top_terms_indices.tolist()[0]]
    print(f"    {', '.join(top_terms)}")

# Step 10: Analyze class assignments
print("\nClass Assignment Analysis:")
class_counts = indicators_df['Assigned_Class'].value_counts()
print(class_counts)

# Step 11: Save results
indicators_df.to_csv('clustered_indicators.csv', index=False)

# Step 12: Analyze indicators in each cluster
for cluster in range(n_clusters):
    cluster_indicators = indicators_df[indicators_df['Cluster'] == cluster]
    print(f"\nSample indicators in Cluster {cluster} (Assigned Class: {closest_class_per_cluster[cluster]}):")
    print(cluster_indicators['Indicator'].head().to_string(index=False))