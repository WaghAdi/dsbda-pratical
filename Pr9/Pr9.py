import pandas as pd
import seaborn as sns

df = pd.read_csv("train.csv")

print(df.head())
print(df.info())
print(df.describe())
print(df.count())

print(df.isnull().sum())
print(df.shape)

df = df.dropna()

print(df.isnull().sum())
print(df.shape)

print(df.columns)


df = df.drop(['PassengerId', 'Pclass', 'Name', 'SibSp', 'Parch', 'Cabin', 'Embarked'], axis = 1)


print(df.head())
print(df.dtypes)

#print(sns.countplot(x = 'Sex', data = df))

#print(sns.countplot(x = 'Survived', data = df))

#print(sns.histplot(data = df, x = 'Fare'))
#print(sns.histplot(data = df, x = 'Fare', y = 'Ticket'))

#print(sns.boxplot(x = 'Sex', y = 'Age', data = df))


#print(sns.boxplot(x = 'Survived', y = 'Age', data = df))


print(sns.boxplot(x = 'Sex', y = 'Survived', data = df))

