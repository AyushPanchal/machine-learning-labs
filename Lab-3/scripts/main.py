from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

data = pd.read_csv(r"dataset\drug200.csv")

# Define features and target (No encoding for categorical features)
X = data.drop('Drug', axis=1)
y = data['Drug']

# Split the dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy on the test set
accuracy_without_encoding = accuracy_score(y_test, y_pred)
print(f"Accuracy without encoding: {accuracy_without_encoding:.2f}")
