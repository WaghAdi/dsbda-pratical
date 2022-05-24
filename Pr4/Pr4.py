import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

df = pd.read_csv("HousingData.csv")

print("The head of the dataset: ")
print(df.head())

print("The datatypes of the attributes are: ")
print(df.dtypes)

print("The count of records present in the each column are: ")
print(df.count())

print("The information of the dataset is: ")
print(df.info())

print("Checking null values present in each attribute of dataset: ")
print(df.isnull().sum())

print("Droping null values present in the datset: ")
df = df.dropna()
print(df.isnull().sum())


# Creating a traing and testing datasets
x = df.iloc[:, :-1]
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)

# Creating a Linear Regression model
model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

y_pred = pd.DataFrame(y_pred)

print("The actual values: ")
print(y_test)

print("The predicted values: ")
print(y_pred)
























