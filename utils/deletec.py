import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


df = pd.read_csv("indicators.csv")

# Define features and target variable
X = df[["Economic", "Environmental", "Social"]]
y = df["Economic"]  # Assuming 'Economic' as the target variable for demonstration

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
log_reg = LogisticRegression()
dec_tree = DecisionTreeClassifier()
rand_forest = RandomForestClassifier()

# Train models
log_reg.fit(X_train, y_train)
dec_tree.fit(X_train, y_train)
rand_forest.fit(X_train, y_train)

# Predict
log_reg_pred = log_reg.predict(X_test)
dec_tree_pred = dec_tree.predict(X_test)
rand_forest_pred = rand_forest.predict(X_test)

# Calculate metrics
def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    return accuracy, f1, precision, recall

log_reg_metrics = evaluate_model(y_test, log_reg_pred)
dec_tree_metrics = evaluate_model(y_test, dec_tree_pred)
rand_forest_metrics = evaluate_model(y_test, rand_forest_pred)

# Display results
results = {
    "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
    "Accuracy": [log_reg_metrics[0], dec_tree_metrics[0], rand_forest_metrics[0]],
    "F1 Score": [log_reg_metrics[1], dec_tree_metrics[1], rand_forest_metrics[1]],
    "Precision": [log_reg_metrics[2], dec_tree_metrics[2], rand_forest_metrics[2]],
    "Recall": [log_reg_metrics[3], dec_tree_metrics[3], rand_forest_metrics[3]]
}

results_df = pd.DataFrame(results)
results_df
