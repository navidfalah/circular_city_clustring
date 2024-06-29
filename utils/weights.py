import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from itertools import permutations, product

# NLTK downloads
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
    return ' '.join(word for word in lemmatized if word not in stop_words)

def load_and_process_csv(file_name, weight_dict):
    try:
        df = pd.read_csv(file_name)
        # Apply preprocessing to each element individually
        for col in df.columns:
            df[col] = df[col].astype(str).apply(preprocess_text)
        
        # Construct weighted text for each row
        weighted_texts = []
        for _, row in df.iterrows():
            row_text = ' '.join(row[col] * weight for col, weight in weight_dict.items())
            weighted_texts.append(row_text)
        return weighted_texts
    except FileNotFoundError:
        print(f"Error: File {file_name} not found.")
        return []

def all_permutations():
    return list(permutations([1, 2, 3, 4]))

def experiment_with_weights(indicator_file, feature_file):
    indicator_permutations = all_permutations()
    feature_permutations = all_permutations()

    # Try each combination of indicator and feature weight permutations
    for ind_weights, feat_weights in product(indicator_permutations, feature_permutations):
        indicator_weights = dict(zip(['Category', 'Subcategory', 'Indicator', 'Description'], ind_weights))
        feature_weights = dict(zip(['Dimension', 'Core Features', 'Examples of Indicators', 'Key Initiatives'], feat_weights))

        # Load and process the texts
        indicator_texts = load_and_process_csv(indicator_file, indicator_weights)
        feature_texts = load_and_process_csv(feature_file, feature_weights)
        texts = indicator_texts + feature_texts

        if texts:
            # Perform TF-IDF Vectorization and NMF
            vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=0.01, ngram_range=(1, 2))
            X = vectorizer.fit_transform(texts)
            nmf = NMF(n_components=3, random_state=0, init='nndsvd')
            W = nmf.fit_transform(X)
            W_normalized = normalize(W, norm='l1', axis=1)

            # Analyze and display results
            print(f"Results for Indicator Weights: {ind_weights}, Feature Weights: {feat_weights}")
            features = vectorizer.get_feature_names_out()
            top_terms = 10
            print("\nTop terms per cluster:")
            for i, comp in enumerate(nmf.components_):
                terms_comp = zip(features, comp)
                sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:top_terms]
                print(f"Cluster {i+1}: " + ", ".join(t[0] for t in sorted_terms))

def evaluate_weights(indicator_file, feature_file):
    best_score = -1
    best_weights = None
    
    indicator_permutations = all_permutations()
    feature_permutations = all_permutations()

    # Load features.csv for evaluation
    try:
        features_df = pd.read_csv(feature_file)
    except FileNotFoundError:
        print(f"Error: File {feature_file} not found.")
        return

    for ind_weights, feat_weights in product(indicator_permutations, feature_permutations):
        indicator_weights = dict(zip(['Category', 'Subcategory', 'Indicator', 'Description'], ind_weights))
        feature_weights = dict(zip(['Dimension', 'Core Features', 'Examples of Indicators', 'Key Initiatives'], feat_weights))

        # Load and process the texts using indicator weights only
        indicator_texts = load_and_process_csv(indicator_file, indicator_weights)

        if indicator_texts:
            # Combine indicator_texts with feature texts from features.csv
            feature_texts = [preprocess_text(row) for row in features_df.iloc[:, 2:].fillna('').apply(lambda x: ' '.join(x), axis=1).tolist()]
            texts = indicator_texts + feature_texts

            # Perform TF-IDF Vectorization and NMF
            vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=0.01, ngram_range=(1, 2))
            X = vectorizer.fit_transform(texts)
            nmf = NMF(n_components=3, random_state=0, init='nndsvd')
            W = nmf.fit_transform(X)
            W_normalized = normalize(W, norm='l1', axis=1)

            # Evaluate the quality of clusters based on predefined features
            # Example: Calculate a simple score based on overlap of top terms with predefined features
            score = 0
            features_list = features_df.iloc[:, 2:].values.flatten().tolist()
            features_set = set(preprocess_text(feature) for feature in features_list if pd.notna(feature))
            features = vectorizer.get_feature_names_out()
            top_terms = 10

            for i, comp in enumerate(nmf.components_):
                terms_comp = zip(features, comp)
                sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:top_terms]
                cluster_terms = set(t[0] for t in sorted_terms)
                score += len(cluster_terms.intersection(features_set))  # Example scoring metric

            if score > best_score:
                best_score = score
                best_weights = (ind_weights, feat_weights)

            print(f"Results for Indicator Weights: {ind_weights}, Feature Weights: {feat_weights}")
            print(f"Score: {score}")
            print("\nTop terms per cluster:")
            for i, comp in enumerate(nmf.components_):
                terms_comp = zip(features, comp)
                sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:top_terms]
                print(f"Cluster {i+1}: " + ", ".join(t[0] for t in sorted_terms))
            print("\n")

    print(f"Best weights found: Indicator {best_weights[0]}, Feature {best_weights[1]} with score {best_score}")

# Example usage
evaluate_weights('indicators.csv', 'features.csv')
