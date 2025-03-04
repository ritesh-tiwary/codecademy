import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

# Load digits dataset
digits = datasets.load_digits()
print(digits.DESCR)

# Print dataset details
print(digits.data)
print(digits.target)

# Visualize one image
plt.gray()
plt.matshow(digits.images[100])
plt.show()
print(digits.target[100])

# K-Means Clustering
k = 10  # Number of clusters
model = KMeans(n_clusters=k)
model.fit(digits.data)

# Visualizing cluster centers
fig = plt.figure(figsize=(8, 3))
fig.suptitle("Cluster Centers")
for i in range(10):
    ax = fig.add_subplot(2, 5, 1 + i)
    ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
plt.show()

# Predict new samples (example input, replace with actual test data)
new_samples = np.array([
    [0, 0, 10, 15, 14, 1, 0, 0, 0, 4, 16, 11, 14, 6, 0, 0, 0, 5, 15, 1, 12, 7, 0, 0, 0, 4, 12, 1, 16, 5, 0, 0, 0, 0, 0, 3, 15, 2, 0, 0, 0, 0, 0, 10, 8, 0, 0, 0, 0, 0, 0, 12, 5, 0, 0, 0, 0, 0, 0, 14, 4, 0, 0, 0],
    [0, 0, 3, 15, 12, 2, 0, 0, 0, 0, 12, 9, 14, 5, 0, 0, 0, 4, 16, 2, 16, 5, 0, 0, 0, 4, 16, 1, 16, 5, 0, 0, 0, 0, 1, 5, 15, 1, 0, 0, 0, 0, 0, 13, 7, 0, 0, 0, 0, 0, 0, 16, 5, 0, 0, 0, 0, 0, 0, 15, 3, 0, 0, 0]
])

new_labels = model.predict(new_samples)

# Mapping clusters to digits
for i in range(len(new_labels)):
    if new_labels[i] == 0:
        print(0, end='')
    elif new_labels[i] == 1:
        print(9, end='')
    elif new_labels[i] == 2:
        print(2, end='')
    elif new_labels[i] == 3:
        print(1, end='')
    elif new_labels[i] == 4:
        print(6, end='')
    elif new_labels[i] == 5:
        print(8, end='')
    elif new_labels[i] == 6:
        print(4, end='')
    elif new_labels[i] == 7:
        print(5, end='')
    elif new_labels[i] == 8:
        print(7, end='')
    elif new_labels[i] == 9:
        print(3, end='')

