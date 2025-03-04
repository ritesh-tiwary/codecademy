import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import RFE

# Load the dataset
df = pd.read_csv("https://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition+/data/obesity.csv")

# Split into X (features) and y (target)
X = df.drop(columns=["NObeyesdad"])  # Replace with actual target column
y = df["NObeyesdad"]

# Encode categorical target
y = pd.factorize(y)[0]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {accuracy:.2f}")

# Sequential Forward Selection (SFS)
sfs = SFS(model, k_features=5, forward=True, floating=False, scoring='accuracy', cv=5)
sfs.fit(X_train_scaled, y_train)
print("Chosen Features (SFS):", list(X.columns[list(sfs.k_feature_idx_)]))

# Fit and evaluate model with selected features
X_train_sfs = sfs.transform(X_train_scaled)
X_test_sfs = sfs.transform(X_test_scaled)
model.fit(X_train_sfs, y_train)
y_pred_sfs = model.predict(X_test_sfs)
accuracy_sfs = accuracy_score(y_test, y_pred_sfs)
print(f"SFS Model Accuracy: {accuracy_sfs:.2f}")

# Sequential Backward Selection (SBS)
sbs = SFS(model, k_features=5, forward=False, floating=False, scoring='accuracy', cv=5)
sbs.fit(X_train_scaled, y_train)
print("Chosen Features (SBS):", list(X.columns[list(sbs.k_feature_idx_)]))

# Fit and evaluate model with selected features
X_train_sbs = sbs.transform(X_train_scaled)
X_test_sbs = sbs.transform(X_test_scaled)
model.fit(X_train_sbs, y_train)
y_pred_sbs = model.predict(X_test_sbs)
accuracy_sbs = accuracy_score(y_test, y_pred_sbs)
print(f"SBS Model Accuracy: {accuracy_sbs:.2f}")

# Recursive Feature Elimination (RFE)
rfe = RFE(model, n_features_to_select=5)
rfe.fit(X_train_scaled, y_train)
selected_features_rfe = X.columns[rfe.support_]
print("Chosen Features (RFE):", list(selected_features_rfe))

# Fit and evaluate model with selected features
X_train_rfe = rfe.transform(X_train_scaled)
X_test_rfe = rfe.transform(X_test_scaled)
model.fit(X_train_rfe, y_train)
y_pred_rfe = model.predict(X_test_rfe)
accuracy_rfe = accuracy_score(y_test, y_pred_rfe)
print(f"RFE Model Accuracy: {accuracy_rfe:.2f}")

# Visualization
plt.bar(["Logistic", "SFS", "SBS", "RFE"], [accuracy, accuracy_sfs, accuracy_sbs, accuracy_rfe])
plt.ylabel("Accuracy")
plt.title("Feature Selection Model Accuracy")
plt.show()

