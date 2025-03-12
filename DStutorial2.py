# Imports necessary libraries
import numpy as np  
import pandas as pd  
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score  

# Dataset creation
data = {
    'Hours_Studied': [10, 8, 7, 5, 6, 9, 4, 3, 2, 6],
    'Attendance': [90, 80, 70, 60, 65, 85, 50, 40, 30, 60],
    'Pass': [1, 1, 1, 0, 1, 1, 0, 0, 0, 0]  # 1 = Pass, 0 = Fail
}

# Creating a DataFrame from the dataset
df = pd.DataFrame(data)

# Separating features (X) and target (y)
X = df[['Hours_Studied', 'Attendance']]  # Features
y = df['Pass']  # Target (Pass/Fail)

# Initializing and training the logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Making predictions on the training data
predictions = model.predict(X)

# Calculating and displaying the model's accuracy
accuracy = accuracy_score(y, predictions)

# Predicting the probability of passing for new data
new_data = pd.DataFrame([[7, 75]], columns=['Hours_Studied', 'Attendance'])  
prob = model.predict_proba(new_data)[0, 1]  

# Displaying the results
print("Model Accuracy:", accuracy)
print("Probability of passing for new input [7 hours, 75% attendance]:", prob)
