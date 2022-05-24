import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('iris.csv')

print("The head of the dataset is : ")
print(df.head())


print(df.describe())

print(df.isnull().sum())

print(df.info())


# Preparing data for the model 
x = df.iloc[:, 1:5]
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)

# Creating model of naive bayes
model = GaussianNB()
model.fit(x_train, y_train)

# Predictions on the model
y_pred = model.predict(x_test)

print(y)
print(y_pred)

print("The accuracy score is: ")
ac = accuracy_score(y_test, y_pred)*100
print(ac)

print("The confusion Matrix is: ")
cm = confusion_matrix(y_test, y_pred)
print(cm)


# for setosa class
tp = cm[0][0]
tn = cm[1][1]+cm[1][2]+cm[2][1]+cm[2][2]
fp = cm[1][0]+cm[2][0]
fn = cm[0][1]+cm[0][2]

total = tp+tn+fp+fn
print("For setosa class")
print(tp, tn, fp, fn)
print("Error Rate: ")
print((fp+fn)/total)


# for virginica class
tp=cm[1][2]
fn=(cm[2][0])+(cm[2][1])
tn=(cm[0][0])+(cm[0][1])+(cm[1][0])+(cm[1][1])
fp=(cm[0][2])+(cm[1][2])

total = tp+tn+fp+fn
print("For virginica class")
print(tp, tn, fp, fn)
print("Error Rate: ")
print((fp+fn)/total)


# for versicolor class
tp=cm[1][1]
fn=(cm[1][0])+(cm[1][2])
tn=(cm[0][0])+(cm[0][2])+(cm[2][0])+(cm[2][2])
fp=(cm[0][1])+(cm[2][1])

total = tp+tn+fp+fn
print("For versicolor class")
print(tp, tn, fp, fn)
print("Error Rate: ")
print((fp+fn)/total)


print("The classification report is : ")
print(classification_report(y_test, y_pred))



























