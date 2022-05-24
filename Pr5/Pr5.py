import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("Social_Network_Ads.csv")

print("The head data of dataframe: ")
print(df.head())


print(df.describe())


print(df.isnull().sum())


# Creating a model of Logistic Regression

x = df.iloc[:,[2,3]].values
print(x)

y = df.iloc[:,4].values
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)


model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(y_pred)

ac = accuracy_score(y_test, y_pred)*100
print("The accuracy score is: ")
print(ac)

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix is: ")
print(cm)

tp = cm[0][0]
tn = cm[1][1]
fp = cm[1][0]
fn = cm[0][1]

total = tp+tn+fp+fn

error_rate = (fp+fn)/total
print("The error rate is: ")
print(error_rate)

print("The classification report is: \n")
print(classification_report(y_test, y_pred))






















