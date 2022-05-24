import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('heart.csv')
print(f"The Head of the dataset: \n{df.head()}\n")

print(f'Missing values in the datset: \n{df.isnull()} \n')

print(f'Sum of Missing values in the datset: \n{df.isnull().sum()} \n')

print(f'The description of the dataset: \n{df.describe} \n')

print(f'The information of the dataset: \n{df.info} \n')

print(f'The dimensition of the datset are: \n{df.shape} \n')

print(f"The datatyes of variables are: \n{df.dtypes} \n")

print(f"The scatter plot of Cholesterol and HeartDisease is: \n{plt.scatter(df.Cholesterol, df.HeartDisease, color = 'red')}")

print("Conver float datatype into integer of Oldpeak attribute: \n")

df['Oldpeak'] = df['Oldpeak'].astype('int64')

print(f"The datatyes of variables are: \n{df.dtypes} \n")

print("Converting Categorical values into quantitive variables: \n")

print(f"The dummies values in dataset are: \n{pd.get_dummies(df, dtype='int')}\n")

