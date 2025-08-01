import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

df=pd.read_csv('titanic.csv')
df=df.dropna(subset=['Age','Sex','Embarked'])

print(df)


#map features
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

#feature selection
X = df[['Pclass', 'Sex', 'Age', 'sibsp', 'Parch', 'Fare', 'Embarked']]
y = df['2urvived']

#train_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#predict always using dependent variable
y_pred = model.predict(X_test)

#generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)