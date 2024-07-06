import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
from matplotlib_venn import venn3

# Ensure NLTK downloads 'punkt' if not already done
nltk.download('punkt')

# Load the data
df = pd.read_csv("indicators.csv")

# Load the additional labeled data
labeled_df = pd.read_csv("features.csv")

# Combine the labeled data with the original data
combined_df = pd.concat([df, labeled_df], ignore_index=True)

# Remove rows with NaN values in the 'Indicator' column
combined_df = combined_df.dropna(subset=['Indicator'])

# Alternatively, you could fill NaN values with a placeholder string:
# combined_df['Indicator'] = combined_df['Indicator'].fillna("No indicator")

# Preprocess the text data
documents = combined_df["Indicator"]

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_tfidf = tfidf_vectorizer.fit_transform(documents)

# LDA Model for Topic Modeling
lda = LatentDirichletAllocation(n_components=10, random_state=42)
X_lda = lda.fit_transform(X_tfidf)

# BERT Model for Embeddings
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize and encode sequences
tokenized = documents.apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
max_len = max(map(len, tokenized))
padded = torch.tensor([i + [0] * (max_len - len(i)) for i in tokenized])
with torch.no_grad():
    embeddings = model(padded)[0][:, 0, :].numpy()

# Word2Vec Model for Word Embeddings
tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
word2vec_model = Word2Vec(tokenized_docs, vector_size=100, window=5, min_count=1, workers=4)

# Doc2Vec Model for Document Embeddings
tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()), tags=[str(i)]) for i, doc in enumerate(documents)]
doc2vec_model = Doc2Vec(tagged_data, vector_size=100, window=5, min_count=1, workers=4)

# Define the target variables
y_economic = combined_df["Economic"].fillna(0)
y_environmental = combined_df["Environmental"].fillna(0)
y_social = combined_df["Social"].fillna(0)

# Function to train and evaluate models
def train_and_evaluate_models(X, y, label):
    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    results = {}

    # Train and evaluate each model
    for model_name, model in models.items():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)

        # Store results
        results[model_name] = {
            "Accuracy": accuracy,
            "F1 Score": f1,
            "Precision": precision,
            "Recall": recall,
            "ROC AUC": roc_auc,
            "FPR": fpr,
            "TPR": tpr
        }

        # Print classification report and confusion matrix
        print(f"Model: {model_name} - {label}")
        print(classification_report(y_test, y_pred))
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        print()

    return results

# Evaluate models for each feature extraction method
print("===== TF-IDF Features =====")
tfidf_results = train_and_evaluate_models(X_tfidf, y_economic, "TF-IDF")

print("===== LDA Features =====")
lda_results = train_and_evaluate_models(X_lda, y_economic, "LDA")

print("===== BERT Embeddings =====")
bert_results = train_and_evaluate_models(embeddings, y_economic, "BERT")

print("===== Word2Vec Embeddings =====")
word2vec_results = train_and_evaluate_models(
    np.array([np.mean([word2vec_model.wv[word] for word in doc], axis=0) for doc in tokenized_docs]),
    y_economic, "Word2Vec")

print("===== Doc2Vec Embeddings =====")
doc2vec_results = train_and_evaluate_models(
    np.array([doc2vec_model.dv[i] for i in range(len(documents))]),
    y_economic, "Doc2Vec")

# Function to find the best model
def find_best_model(results_dict):
    best_model = None
    best_score = 0
    best_feature = None
    
    for feature, models in results_dict.items():
        for model, metrics in models.items():
            if metrics['F1 Score'] > best_score:
                best_score = metrics['F1 Score']
                best_model = model
                best_feature = feature
    
    return best_feature, best_model, best_score

# Combine all results
all_results = {
    "TF-IDF": tfidf_results,
    "LDA": lda_results,
    "BERT": bert_results,
    "Word2Vec": word2vec_results,
    "Doc2Vec": doc2vec_results
}

best_feature, best_model, best_score = find_best_model(all_results)

print(f"Best Model: {best_model}")
print(f"Best Feature: {best_feature}")
print(f"Best F1 Score: {best_score:.4f}")

# Use the best model and feature extraction method
if best_feature == "TF-IDF":
    X = X_tfidf
elif best_feature == "LDA":
    X = X_lda
elif best_feature == "BERT":
    X = embeddings
elif best_feature == "Word2Vec":
    X = np.array([np.mean([word2vec_model.wv[word] for word in doc], axis=0) for doc in tokenized_docs])
else:  # Doc2Vec
    X = np.array([doc2vec_model.dv[i] for i in range(len(documents))])

# Train the best model on all data
if best_model == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
elif best_model == "Decision Tree":
    model = DecisionTreeClassifier()
else:  # Random Forest
    model = RandomForestClassifier()

# Train and predict for each class
classes = ["Economic", "Environmental", "Social"]
predictions = {}

for class_name in classes:
    y = combined_df[class_name].fillna(0)
    model.fit(X, y)
    predictions[class_name] = model.predict(X)

# Create sets of indicators for each class based on predictions
economic_indicators = set(combined_df[predictions["Economic"] == 1]["Indicator"])
environmental_indicators = set(combined_df[predictions["Environmental"] == 1]["Indicator"])
social_indicators = set(combined_df[predictions["Social"] == 1]["Indicator"])

# Create Venn diagram
plt.figure(figsize=(10, 10))
venn3([economic_indicators, environmental_indicators, social_indicators], 
      ('Economic', 'Environmental', 'Social'))
plt.title("Venn Diagram of Predicted Indicator Classes")
plt.show()

# Print statistics
print(f"Number of Economic Indicators: {len(economic_indicators)}")
print(f"Number of Environmental Indicators: {len(environmental_indicators)}")
print(f"Number of Social Indicators: {len(social_indicators)}")

# Calculate overlaps
eco_env = len(economic_indicators.intersection(environmental_indicators))
eco_soc = len(economic_indicators.intersection(social_indicators))
env_soc = len(environmental_indicators.intersection(social_indicators))
all_three = len(economic_indicators.intersection(environmental_indicators, social_indicators))

print(f"\nOverlap between Economic and Environmental: {eco_env}")
print(f"Overlap between Economic and Social: {eco_soc}")
print(f"Overlap between Environmental and Social: {env_soc}")
print(f"Overlap among all three classes: {all_three}")
