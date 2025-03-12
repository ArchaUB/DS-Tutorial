#This code follows the step-by-step reduction method for LDA on a 2x2 matrix
import numpy as np

# Step 1: Define the matrices 
X = np.array([[2, 3], [5, 7]])  # Class 1
Y = np.array([[8, 6], [4, 10]]) # Class 2

# Step 2: Compute class means
mu_X = np.mean(X, axis=0).reshape(-1, 1)
mu_Y = np.mean(Y, axis=0).reshape(-1, 1)

# Step 3: Compute Within-Class Scatter Matrix
S_W = np.dot((X - mu_X), (X - mu_X).T) + np.dot((Y - mu_Y), (Y - mu_Y).T)

# Step 4: Compute Between-Class Scatter Matrix
mu_diff = mu_X - mu_Y
S_B = np.dot(mu_diff, mu_diff.T)

# Step 5: Compute the Discriminant Vector w
S_W_inv = np.linalg.inv(S_W)
w = np.dot(S_W_inv, mu_diff)

# Step 6: Project the Data
z_X = np.dot(X, w)  
z_Y = np.dot(Y, w)  

# Results
print(f"Mean of Class 1:\n{mu_X}")
print(f"Mean of Class 2:\n{mu_Y}")
print(f"Within-Class Scatter Matrix:\n{S_W}")
print(f"Between-Class Scatter Matrix:\n{S_B}")
print(f"Discriminant Vector w:\n{w}")
print(f"Projected Class 1 Data:\n{z_X}")
print(f"Projected Class 2 Data:\n{z_Y}")
