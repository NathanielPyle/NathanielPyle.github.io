from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import re

column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

df = pd.read_csv("D:/Downloads/iris/iris.data", names=column_names, header=None)

print(df.head())

plt.scatter(df['class'],df['sepal_length'])
plt.show()

print(df.columns)

print(df['class'])

print(df.iloc[0:4])

# for index,row in df.iterrows():
#     print(index,row)


print(df.loc[df['sepal_length'] >=5.0])
print(df.loc[df['petal_width'] <= 1.0])

print(df.shape)

print(df.count())

value_counts = df['class'].value_counts()
print(value_counts)

df['Total'] = df['sepal_length'] + df['sepal_width'] + df['petal_length'] + df['petal_width']


print(df.head())

df = df.drop(columns=['Total'])

print(df.head())

df['Total'] = df.iloc[:, 0:4].sum(axis=1)

print(df.head())

df = df[['Total', 'class', 'sepal_length', 'sepal_width', 'petal_width', 'petal_length']]

print(df.head())

cols = list(df.columns.values)
df = df[cols[0:4] + [cols[-1]]]
print(df.head())

df.to_csv("modified.csv")



# df.to_excel("modified.xlsx", index = False)
# df.to_csv("modified.csv", index = False)
# df.to_csv("modified.txt", index = False, sep='\t')



new_df = df.loc[(df['class'] == 'Iris-setosa') &
       (df['sepal_length'] >= 5.0)]


print(new_df.head())

new_df = new_df.reset_index(drop=True, inplace=True)

df = df.loc[~df["class"].str.contains("Iris")]


print(df.head())

new_df.loc[new_df['class'].str.contains('Iris|setosa',regex=True)]

# plt.bar(Y['class'],X['sepal length'])
# plt.title("Class of flowers compared with Sepal Length")
# plt.xlabel("Class")
# plt.ylabel("Sepal length")
# plt.show()
#
#

#

# sepal_length = np.loadtxt(file_path, delimiter=',', usecols=0, dtype=float)
#
# sepal_width = np.loadtxt(file_path, delimiter=',', usecols=1, dtype=float)
#
# petal_length = np.loadtxt(file_path, delimiter=',', usecols=2, dtype=float)
#
# petal_width = np.loadtxt(file_path, delimiter=',', usecols=3, dtype=float)
#
# classes = np.loadtxt(file_path, delimiter=',', usecols=4, dtype=str)
#
# class_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
# class_values = np.vectorize(class_mapping.get)(classes)
#
# colors = ['red', 'green', 'blue']
#
# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(sepal_length, sepal_width, c=class_values, cmap=plt.get_cmap('viridis', 3), s=10, alpha=0.7)
#
# plt.xlabel('Sepal Length (cm)')
# plt.ylabel('Sepal Width (cm)')
# plt.title('Sepal Dimensions by Iris Class')
#
# # Add legend for classes
# plt.legend(handles=scatter.legend_elements()[0], labels=class_mapping.keys())
#
#
# # Display the plot
# plt.show()
#
# iris_df = pd.DataFrame({
#     'sepal_length': sepal_length,
#     'sepal_width': sepal_width,
#     'petal_length': petal_length,
#     'petal_width': petal_width,
#     'species': classes
# })
#
# # Generate pairplot using seaborn
# sns.pairplot(iris_df, hue='species')
# plt.show()
#
# sns.pairplot(iris_df[['sepal_length', 'sepal_width']])
# plt.show()