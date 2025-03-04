import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the data
aaron_judge = pd.read_csv('aaron_judge.csv')

# Preprocess data
aaron_judge = aaron_judge[['plate_x', 'plate_z', 'type']]

# Map type to numerical values and drop NaNs
aaron_judge['type'] = aaron_judge['type'].map({'S': 1, 'B': 0})
aaron_judge = aaron_judge.dropna(subset=['plate_x', 'plate_z', 'type'])

# Remove infinite values (if any)
aaron_judge = aaron_judge.replace([np.inf, -np.inf], np.nan).dropna()

# Verify unique values in 'type'
print("Unique values in 'type':", aaron_judge['type'].unique())

# Ensure dtype compatibility
aaron_judge = aaron_judge.astype({'plate_x': 'float64', 'plate_z': 'float64', 'type': 'int'})

# Split the data
training_set, validation_set = train_test_split(aaron_judge, random_state=1)

# Train SVM classifier
classifier = SVC(kernel='rbf', gamma=100, C=100)
classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type'])

# Evaluate the model
accuracy = classifier.score(validation_set[['plate_x', 'plate_z']], validation_set['type'])
print(f'Accuracy: {accuracy:.2f}')

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(aaron_judge['plate_x'], aaron_judge['plate_z'], c=aaron_judge['type'], cmap=plt.cm.coolwarm, alpha=0.25)
plt.xlabel('Plate X')
plt.ylabel('Plate Z')
plt.title('Strike Zone Visualization')
plt.show()

