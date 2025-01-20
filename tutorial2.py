#imports
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Dataset creation
# Hours Studied, Attendance, Pass (1=Yes, 0=No)
data = {
    'Hours_Studied': [10, 8, 7, 5, 6, 9, 4, 3, 2, 6],
    'Attendance': [90, 80, 70, 60, 65, 85, 50, 40, 30, 60],
    'Pass': [1, 1, 1, 0, 1, 1, 0, 0, 0, 0]
}

df = pd.DataFrame(data)

X = df[['Hours_Studied', 'Attendance']]
y = df['Pass']

model = LogisticRegression()
model.fit(X, y)

predictions = model.predict(X)
accuracy = accuracy_score(y, predictions)

new_data = np.array([[7, 75]])  
prob = model.predict_proba(new_data)[0, 1] 

print("Model Accuracy:", accuracy)
print("Probability of passing for new input [7 hours, 75% attendance]:", prob)
