#Implementing LDA Using Standard Library Functions

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample Data (2 features and 2 classes)
data = {
    'Feature1': [2, 3, 5, 7, 9, 1, 6, 8],
    'Feature2': [1, 4, 6, 8, 10, 2, 3, 7],
    'Class': [0, 0, 1, 1, 1, 0, 0, 1]  
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Splitting the data
X = df[['Feature1', 'Feature2']].values
y = df['Class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Applying LDA
lda = LDA()
X_train_lda = lda.fit_transform(X_train, y_train)

# Model evaluation
y_pred = lda.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Transformed Features (LDA):\n', X_train_lda)
