import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("student_marks.csv")

print(df.head())

print("The missing values in the dataset are: \n")
print(df.isnull())
print(df.isnull().sum())



print("The missing values present in the dataset:\n")
print(df.isnull().sum())

print("The datatypes of the attributes present: \n")
print(df.dtypes)


# Replace the missing value from math score column with it's mean value

mean_math = df['math score'].mean()

df['math score'].fillna(mean_math, inplace = True)

print(df)
print("The missing values present in the dataset:\n")
print(df.isnull().sum())



# Deleting the missing values attributes records from the dataset
df = df.dropna()


print("The missing values present in the dataset:\n")
print(df.isnull().sum())


# Converting the datatypes of attributes into the suitable datatypes
df['math score'] = df['math score'].astype('int64')

print(df.dtypes)
df['reading score'] = df['reading score'].astype('int64')

print(df.dtypes)
df['writing score'] = df['writing score'].astype('int64')

print(df.dtypes)

print("The data coloum for math score column is: \n")
print(df['math score'])

print("The minimum value of math score column is: \n")
print(df['math score'].min())

print("The maximum value of math score column is: \n")
print(df['math score'].max())

# Checking if outliers are present in math score column
print(df['math score'].plot(kind='box'))

#print(df['math score'].plot(kind='hist'))

print(df.head())



# change the scale of column writing score for better understanding of the variable
scaler = MinMaxScaler(feature_range = (0, 10))

df[['writing score']] = scaler.fit_transform(df[['writing score']])
df['writing score'] = df['writing score'].round(1)

print(df.head())



















