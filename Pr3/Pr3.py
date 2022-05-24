import pandas as pd

df = pd.read_csv('iris.csv')

print("The head data of the dataset is: ")
print(df.head())

print("The summary statistics of dataset is : \n")
print(df.describe())

print("The unique species present in the dataset are: \n")
print(df['Species'].unique())

# Creating a groups of unique species.
group = df.groupby('Species')

# Printing the group iris-setosa statistical summary
iris_setosa = group.get_group('Iris-setosa')
print("The group of iris-setosa is: ")
print(iris_setosa)
print(iris_setosa.describe())

# Printing the group iris-versicolor statistical summary
iris_versicolor = group.get_group('Iris-versicolor')
print("The group of iris_versicolor is: ")
print(iris_versicolor)
print(iris_versicolor.describe())

# Printing the group iris-virginica statistical summary
iris_virginica = group.get_group('Iris-virginica')
print("The group of iris_virginica is: ")
print(iris_virginica)
print(iris_virginica.describe())

