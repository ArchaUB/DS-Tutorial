import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Sample 2x2 matrix (features) and corresponding labels
X = np.array([[4, 2],
              [2, 4],
              [3, 1],
              [5, 3]])

y = np.array([0, 1, 0, 1])  # Class labels

# Method 1: Using Standard Library
lda = LinearDiscriminantAnalysis(n_components=1)
X_transformed_lib = lda.fit_transform(X, y)
print("LDA using sklearn:\n", X_transformed_lib)

# Method 2: Using Normal Matrix Multiplication
mean_vectors = []
class_labels = np.unique(y)
for label in class_labels:
    mean_vectors.append(np.mean(X[y == label], axis=0))

# Compute the Scatter Matrices
S_W = np.zeros((2, 2))  # Within-class scatter matrix
for label, mv in zip(class_labels, mean_vectors):
    class_scatter = np.cov(X[y == label].T, bias=True)
    S_W += class_scatter

mean_overall = np.mean(X, axis=0)
S_B = np.zeros((2, 2))  # Between-class scatter matrix
for label, mv in zip(class_labels, mean_vectors):
    n = X[y == label].shape[0]
    mv = mv.reshape(2, 1)
    mean_diff = (mv - mean_overall.reshape(2, 1))
    S_B += n * (mean_diff @ mean_diff.T)

# Compute Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

# Sort eigenvectors by highest eigenvalue
eigenvector = eigenvectors[:, np.argmax(eigenvalues)]

# Project data onto the new axis
X_transformed_manual = X.dot(eigenvector)
print("\nLDA using Normal Matrix Multiplication:\n", X_transformed_manual)
