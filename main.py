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
nltk.download('punkt')

# Load the data
df = pd.read_csv("indicators.csv")

# Preprocess the text data
documents = df["Indicator"]

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
y_economic = df["Economic"].fillna(0)
y_environmental = df["Environmental"].fillna(0)
y_social = df["Social"].fillna(0)

# Train-test split for each feature extraction method
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(X_tfidf, y_economic, test_size=0.2, random_state=42)
X_train_lda, X_test_lda, y_train_lda, y_test_lda = train_test_split(X_lda, y_economic, test_size=0.2, random_state=42)
X_train_bert, X_test_bert, y_train_bert, y_test_bert = train_test_split(embeddings, y_economic, test_size=0.2, random_state=42)
X_train_word2vec, X_test_word2vec, y_train_word2vec, y_test_word2vec = train_test_split(
    np.array([np.mean([word2vec_model.wv[word] for word in doc], axis=0) for doc in tokenized_docs]),
    y_economic, test_size=0.2, random_state=42)
X_train_doc2vec = np.array([doc2vec_model.dv[i] for i in range(len(documents))])
X_train_doc2vec, X_test_doc2vec, y_train_doc2vec, y_test_doc2vec = train_test_split(X_train_doc2vec, y_economic, test_size=0.2, random_state=42)

# Function to train and evaluate models
def train_and_evaluate_models(X_train, X_test, y_train, y_test, label):
    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    results = {}

    # Train and evaluate each model
    for model_name, model in models.items():
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

# Evaluate models for TF-IDF features
print("===== TF-IDF Features =====")
tfidf_results = train_and_evaluate_models(X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf, "TF-IDF")
for model, metrics in tfidf_results.items():
    print(f"Model: {model}")
    print(f"  Accuracy: {metrics['Accuracy']:.4f}")
    print(f"  F1 Score: {metrics['F1 Score']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall: {metrics['Recall']:.4f}")
    print(f"  ROC AUC: {metrics['ROC AUC']:.4f}")
    plt.plot(metrics['FPR'], metrics['TPR'], label=f'{model} (area = {metrics["ROC AUC"]:.2f})')

plt.title('ROC Curve for TF-IDF Features')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

# Evaluate models for LDA features
print("===== LDA Features =====")
lda_results = train_and_evaluate_models(X_train_lda, X_test_lda, y_train_lda, y_test_lda, "LDA")
for model, metrics in lda_results.items():
    print(f"Model: {model}")
    print(f"  Accuracy: {metrics['Accuracy']:.4f}")
    print(f"  F1 Score: {metrics['F1 Score']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall: {metrics['Recall']:.4f}")
    print(f"  ROC AUC: {metrics['ROC AUC']:.4f}")
    plt.plot(metrics['FPR'], metrics['TPR'], label=f'{model} (area = {metrics["ROC AUC"]:.2f})')

plt.title('ROC Curve for LDA Features')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

# Evaluate models for BERT embeddings
print("===== BERT Embeddings =====")
bert_results = train_and_evaluate_models(X_train_bert, X_test_bert, y_train_bert, y_test_bert, "BERT")
for model, metrics in bert_results.items():
    print(f"Model: {model}")
    print(f"  Accuracy: {metrics['Accuracy']:.4f}")
    print(f"  F1 Score: {metrics['F1 Score']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall: {metrics['Recall']:.4f}")
    print(f"  ROC AUC: {metrics['ROC AUC']:.4f}")
    plt.plot(metrics['FPR'], metrics['TPR'], label=f'{model} (area = {metrics["ROC AUC"]:.2f})')

plt.title('ROC Curve for BERT Embeddings')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

# Evaluate models for Word2Vec embeddings
print("===== Word2Vec Embeddings =====")
word2vec_results = train_and_evaluate_models(X_train_word2vec, X_test_word2vec, y_train_word2vec, y_test_word2vec, "Word2Vec")
for model, metrics in word2vec_results.items():
    print(f"Model: {model}")
    print(f"  Accuracy: {metrics['Accuracy']:.4f}")
    print(f"  F1 Score: {metrics['F1 Score']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall: {metrics['Recall']:.4f}")
    print(f"  ROC AUC: {metrics['ROC AUC']:.4f}")
    plt.plot(metrics['FPR'], metrics['TPR'], label=f'{model} (area = {metrics["ROC AUC"]:.2f})')

plt.title('ROC Curve for Word2Vec Embeddings')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

# Evaluate models for Doc2Vec embeddings
print("===== Doc2Vec Embeddings =====")
doc2vec_results = train_and_evaluate_models(X_train_doc2vec, X_test_doc2vec, y_train_doc2vec, y_test_doc2vec, "Doc2Vec")
for model, metrics in doc2vec_results.items():
    print(f"Model: {model}")
    print(f"  Accuracy: {metrics['Accuracy']:.4f}")
    print(f"  F1 Score: {metrics['F1 Score']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall: {metrics['Recall']:.4f}")
    print(f"  ROC AUC: {metrics['ROC AUC']:.4f}")
    plt.plot(metrics['FPR'], metrics['TPR'], label=f'{model} (area = {metrics["ROC AUC"]:.2f})')

plt.title('ROC Curve for Doc2Vec Embeddings')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()
