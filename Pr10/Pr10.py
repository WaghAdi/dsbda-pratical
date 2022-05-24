import pandas as pd
import seaborn as sns


df = pd.read_csv("iris.csv")

print(df.head())

print(df.dtypes)

print(df.info())

print(df.describe())

#print(sns.histplot(data = df, x = 'Id'))
#print(sns.histplot(data = df, x = 'SepalLengthCm'))
#print(sns.histplot(data = df, x = 'SepalWidthCm'))
#print(sns.histplot(data = df, x = 'PetalLengthCm'))
#print(sns.histplot(data = df, x = 'PetalWidthCm'))
#print(sns.histplot(data = df, x = 'Species'))


#print(sns.boxplot(data = df, x = 'Id'))
#print(sns.boxplot(data = df, x = 'SepalLengthCm'))
#print(sns.boxplot(data = df, x = 'SepalWidthCm'))
#print(sns.boxplot(data = df, x = 'PetalLengthCm'))
print(sns.boxplot(data = df, x = 'PetalWidthCm'))

