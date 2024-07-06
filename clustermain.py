import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.calibration import CalibratedClassifierCV

# Load the labeled dataset
labeled_data = pd.read_csv('features.csv')

# Prepare the features and labels
X = labeled_data['Indicator']
y = labeled_data['label'].str.split(',')  # Assuming labels are comma-separated

# Convert labels to multi-label format
mlb = MultiLabelBinarizer()
y_encoded = mlb.fit_transform(y)

# Convert text data to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_encoded, test_size=0.2, random_state=42)

# Train a multi-label classifier with probability calibration
base_classifier = LinearSVC(dual=False)
calibrated_classifier = CalibratedClassifierCV(base_classifier, cv=5)
classifier = OneVsRestClassifier(calibrated_classifier)
classifier.fit(X_train, y_train)

# Evaluate the model
y_pred = classifier.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=mlb.classes_))

# Print confusion matrix for each class
for i, class_name in enumerate(mlb.classes_):
    print(f"\nConfusion Matrix for {class_name}:")
    print(multilabel_confusion_matrix(y_test, y_pred)[i])

# Load the unlabeled dataset
unlabeled_data = pd.read_csv('indicators.csv')

# Transform the unlabeled data using the same vectorizer
X_unlabeled = vectorizer.transform(unlabeled_data['Indicator'])

# Predict labels for the unlabeled data
predicted_labels = classifier.predict(X_unlabeled)

# Convert binary predictions back to label strings
predicted_labels_decoded = mlb.inverse_transform(predicted_labels)

# Add predicted labels to the unlabeled dataset
unlabeled_data['Predicted_Labels'] = [','.join(labels) for labels in predicted_labels_decoded]

# Add prediction probabilities
prediction_proba = classifier.predict_proba(X_unlabeled)
for i, class_name in enumerate(mlb.classes_):
    unlabeled_data[f'Probability_{class_name}'] = prediction_proba[:, i]

# Save the labeled dataset
unlabeled_data.to_csv('labeled_data_output.csv', index=False)
print("Labeling complete. Results saved to 'labeled_data_output.csv'")
