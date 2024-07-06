import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Step 1: Read CSV files
def read_data(indicators_path, classes_path):
    indicators_df = pd.read_csv(indicators_path)
    classes_df = pd.read_csv(classes_path)
    return indicators_df, classes_df

# Step 2: Text preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = word_tokenize(str(text).lower())
    return ' '.join(lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words)

def preprocess_data(indicators_df, classes_df):
    indicators_df['Processed_Indicator'] = indicators_df['Indicator'].apply(preprocess_text)
    classes_df['Processed_Indicator'] = classes_df['Indicator'].apply(preprocess_text)
    return indicators_df, classes_df

# Step 3: Text representation methods
def compute_tfidf(indicators_df, classes_df):
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
    tfidf_vectors = tfidf.fit_transform(indicators_df['Processed_Indicator'])
    tfidf_classes = tfidf.transform(classes_df['Processed_Indicator'])
    return tfidf_vectors, tfidf_classes

def compute_sbert(indicators_df, classes_df):
    sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    sbert_vectors = sbert_model.encode(indicators_df['Processed_Indicator'].tolist())
    sbert_classes = sbert_model.encode(classes_df['Processed_Indicator'].tolist())
    return sbert_vectors, sbert_classes

def compute_doc2vec(indicators_df, classes_df):
    documents = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(indicators_df['Processed_Indicator'])]
    doc2vec_model = Doc2Vec(documents, vector_size=100, window=2, min_count=1, workers=4)
    doc2vec_vectors = np.array([doc2vec_model.infer_vector(doc.split()) for doc in indicators_df['Processed_Indicator']])
    doc2vec_classes = np.array([doc2vec_model.infer_vector(doc.split()) for doc in classes_df['Processed_Indicator']])
    return doc2vec_vectors, doc2vec_classes

# Step 4: Clustering methods
def perform_clustering(vectors, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    dbscan = DBSCAN(eps=0.5, min_samples=3)
    
    return {
        'KMeans': kmeans.fit_predict(vectors),
        'Agglomerative': agglomerative.fit_predict(vectors),
        'DBSCAN': dbscan.fit_predict(vectors)
    }

# Step 5: Evaluate clustering results
def evaluate_clustering(vectors, labels):
    if len(np.unique(labels)) < 2:
        return float('-inf'), 0  # Return negative infinity for invalid clusterings
    
    silhouette = silhouette_score(vectors, labels)
    calinski = calinski_harabasz_score(vectors, labels)
    
    return silhouette, calinski

# Step 6: Plotting functions
def plot_clustering_results(results):
    methods = list(results.keys())
    silhouette_scores = [results[method][0] for method in methods]
    calinski_scores = [results[method][1] for method in methods]
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    ax1.set_title('Clustering Performance Comparison')
    ax1.set_xlabel('Methods')
    ax1.set_ylabel('Silhouette Score', color='tab:blue')
    ax1.bar(methods, silhouette_scores, color='tab:blue', alpha=0.6, label='Silhouette Score')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Calinski-Harabasz Score', color='tab:red')
    ax2.plot(methods, calinski_scores, color='tab:red', marker='o', label='Calinski-Harabasz Score')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    fig.tight_layout()
    plt.show()

def plot_heatmap(df):
    plt.figure(figsize=(14, 7))
    sns.heatmap(df, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Heatmap of Indicator-Classes Probabilities')
    plt.xlabel('Classes')
    plt.ylabel('Indicators')
    plt.show()

def plot_snaky_relation(best_vectors, best_classes):
    plt.figure(figsize=(14, 7))
    
    for i in range(best_classes.shape[1]):
        plt.plot(range(len(best_vectors)), best_vectors[:, i], label=f"Class {i}")
    
    plt.title('Snaky Plot of Indicators to Classes')
    plt.xlabel('Indicators')
    plt.ylabel('Classes')
    plt.legend()
    plt.show()

def plot_probability_distribution(output_df):
    plt.figure(figsize=(14, 7))
    sns.histplot(output_df['Max_Probability'], bins=20, kde=True)
    plt.title('Distribution of Max Probabilities')
    plt.xlabel('Max Probability')
    plt.ylabel('Frequency')
    plt.show()

def plot_class_distribution(output_df):
    plt.figure(figsize=(14, 7))
    sns.countplot(data=output_df, y='Most_Likely_Class', order=output_df['Most_Likely_Class'].value_counts().index)
    plt.title('Distribution of Most Likely Classes')
    plt.xlabel('Count')
    plt.ylabel('Class')
    plt.show()

def main():
    # Read data
    indicators_df, classes_df = read_data('indicators.csv', 'sdg11.csv')
    
    # Preprocess data
    indicators_df, classes_df = preprocess_data(indicators_df, classes_df)
    
    # Compute vectors
    tfidf_vectors, tfidf_classes = compute_tfidf(indicators_df, classes_df)
    sbert_vectors, sbert_classes = compute_sbert(indicators_df, classes_df)
    doc2vec_vectors, doc2vec_classes = compute_doc2vec(indicators_df, classes_df)
    
    # Number of clusters
    n_clusters = len(classes_df)
    
    # Perform clustering
    tfidf_clusters = perform_clustering(tfidf_vectors.toarray(), n_clusters)
    sbert_clusters = perform_clustering(sbert_vectors, n_clusters)
    doc2vec_clusters = perform_clustering(doc2vec_vectors, n_clusters)
    
    # Evaluate all combinations
    results = {}
    for vectors, clusters, name in [
        (tfidf_vectors.toarray(), tfidf_clusters, 'TF-IDF'),
        (sbert_vectors, sbert_clusters, 'SBERT'),
        (doc2vec_vectors, doc2vec_clusters, 'Doc2Vec')
    ]:
        for cluster_method, labels in clusters.items():
            silhouette, calinski = evaluate_clustering(vectors, labels)
            results[f"{name}_{cluster_method}"] = (silhouette, calinski)
            print(f"Method: {name}_{cluster_method}")
            print(f"  Silhouette Score: {silhouette:.4f}")
            print(f"  Calinski-Harabasz Score: {calinski:.4f}")
            print()
    
    # Plot clustering results
    plot_clustering_results(results)
    
    # Find the best method based on silhouette score
    best_method = max(results, key=lambda k: results[k][0])
    print(f"Best method: {best_method}")
    
    # Use the best method for final clustering and analysis
    best_vectors, best_classes = {
        'TF-IDF': (tfidf_vectors.toarray(), tfidf_classes.toarray()),
        'SBERT': (sbert_vectors, sbert_classes),
        'Doc2Vec': (doc2vec_vectors, doc2vec_classes)
    }[best_method.split('_')[0]]
    
    best_labels = {
        'TF-IDF': tfidf_clusters,
        'SBERT': sbert_clusters,
        'Doc2Vec': doc2vec_clusters
    }[best_method.split('_')[0]][best_method.split('_')[1]]
    
    # Calculate similarities and probabilities
    similarities = cosine_similarity(best_vectors, best_classes)
    probabilities = similarities / similarities.sum(axis=1, keepdims=True)
    
    # Create output DataFrame
    output_df = pd.DataFrame({'Indicator': np.arange(len(indicators_df))})
    for i in range(len(classes_df)):
        output_df[f'Prob_Class_{i}'] = probabilities[:, i]
    
    # Add most likely class and its probability
    output_df['Most_Likely_Class'] = probabilities.argmax(axis=1)
    output_df['Max_Probability'] = probabilities.max(axis=1)
    
    # Reorder columns
    cols = ['Indicator', 'Most_Likely_Class', 'Max_Probability'] + [col for col in output_df.columns if col.startswith('Prob_Class_')]
    output_df = output_df[cols]
    
    # Save results
    output_df.to_csv('indicator_sdg_analysis.csv', index=False)
    print("\nResults saved to 'indicator_sdg_analysis.csv'")
    
    # Print overall statistics
    print("\nOverall Statistics:")
    print(f"Total number of indicators: {len(indicators_df)}")
    print(f"Number of SDG classes: {len(classes_df)}")
    print(f"Average max probability: {output_df['Max_Probability'].mean():.4f}")
    
    class_distribution = output_df['Most_Likely_Class'].value_counts()
    print("\nSDG Class Distribution:")
    print(class_distribution)
    
    print("\nTop 5 most confident assignments:")
    print(output_df.nlargest(5, 'Max_Probability')[['Indicator', 'Most_Likely_Class', 'Max_Probability']])
    
    print("\nTop 5 least confident assignments:")
    print(output_df.nsmallest(5, 'Max_Probability')[['Indicator', 'Most_Likely_Class', 'Max_Probability']])
    
    # # Plot heatmap
    # probability_df = output_df.set_index('Indicator')[output_df.columns.difference(['Most_Likely_Class', 'Max_Probability'])]
    # plot_heatmap(probability_df)
    
    # Plot snaky plot
    plot_snaky_relation(best_vectors, best_classes)
    
    # Plot probability distribution
    plot_probability_distribution(output_df)
    
    # Plot class distribution
    plot_class_distribution(output_df)

if __name__ == "__main__":
    main()
