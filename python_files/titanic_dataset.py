import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns

titanic = pd.read_csv("D:/Downloads/titanic.csv", index_col=0)

os.chdir('D:/NathanielPyle.github.io/python_files')

#basic overview of the titanic pandas dataframe
## the dataframe consists of 891 rows with 12 columns
print(titanic)
print(titanic.describe())
print(titanic.shape)

## the columns consist of a combination of floats, ints and strings/objects
print(titanic.dtypes)

##check for missing values in pandas dataframe
df = pd.DataFrame(titanic)
check_nan_in_df = df.isnull().values.any()
print(check_nan_in_df)
# # what is intriguing is that dropping any rows which contain N/A values will drastically reduce the dataframe from
# # 891 rows to 183
titanic.dropna(how="any", inplace=True)
print(titanic.shape)

# # reload of the titanic dataframe
titanic = pd.read_csv("D:/Downloads/titanic.csv", index_col=0)

# # using a basic threshold of 2, there are still 891 rows which contain at least 2 missing values
# # most rows contain more than 2 valid non-N/A values
threshold_2 = titanic.dropna(thresh=2)
print(threshold_2.shape)

## various attempts were made to filter the number of N/A values using different thresholds
threshold_3 = titanic.dropna(thresh=3)
print(threshold_3.shape)

threshold_9 = titanic.dropna(thresh=9)
print(threshold_9.shape)

threshold_10 = titanic.dropna(thresh=10)
print(threshold_10.shape)

threshold_11 = titanic.dropna(thresh=11)
print(threshold_11.shape)

## using the following code, it is evident that Age, cabin and embarked contain missing values
missing_data = titanic.isnull().sum()
print(missing_data)


titanic_cleaned = titanic.dropna(how="all")

print(titanic_cleaned.describe())
print(titanic_cleaned.columns)


print(titanic_cleaned.shape)

# the following code was used to drop the following columns as they did not provide any additional information for
# exploratory data analysis
titanic_cleaned = titanic_cleaned.drop(["PassengerId","Ticket","Name", "Cabin"], axis=1)

print(titanic_cleaned.describe())

print(titanic_cleaned.shape)

titanic_cleaned = titanic.dropna(thresh=11)
print(titanic_cleaned.shape)

missing_data = titanic_cleaned.isnull().sum()
print(missing_data)


# seven subgroups were formulated, showing survivors as the dependent variable and pairing that with the independent variables
class_of_survivors = titanic_cleaned.groupby(["Survived", "Pclass"]).size()
age_of_survivors = titanic_cleaned.groupby(["Survived", "Age"]).size()
sibling_or_spouses_aboard = titanic.groupby(["Survived", "SibSp"]).size()
sex_of_survivors = titanic_cleaned.groupby(["Survived", "Sex"]).size()
fare_of_survivors = titanic_cleaned.groupby(["Survived", "Fare"]).size()
embarked_of_survivors = titanic_cleaned.groupby(["Survived", "Embarked"]).size()
parents_children = titanic_cleaned.groupby(["Survived", "Parch"]).size()

# a basic summary of all six subgroups was printed
print(class_of_survivors)
print(age_of_survivors)
print(sibling_or_spouses_aboard)
print(sex_of_survivors)
print(fare_of_survivors)
print(embarked_of_survivors)
print(parents_children)

# histogram showing the age of survivors
plt.hist(age_of_survivors, bins=20, color="skyblue", edgecolor="black")
plt.xlabel("Age of Survivors")
plt.ylabel("Number of Survivors")
plt.xticks(np.linspace(0,80,9))
plt.title("Age of Survivors")
plt.show()

# pie chart showing the class of the survivors
mylabels = ["Deceased 1st class","Deceased 2nd class","Deceased 3rd class",
            "Survivor 1st class", "Survivor 2nd class","Survivor 3rd class"]
plt.title("Pie Chart the Distribution of Survivors as per Passenger Class",fontsize = 12)

plt.pie(class_of_survivors, labels=mylabels, startangle=48, autopct='%1.1f%%')
plt.show()

# stacked bar chart showing survivors with the number siblings or spouses aboard
sibling_or_spouses_aboard = sibling_or_spouses_aboard.unstack()

sibling_or_spouses_aboard.plot(kind='bar', stacked=True)

#stacked bar chart showing sibling_or_spouses_aboard
plt.xticks(rotation=0)
plt.xlabel("Groups Aboard (SibSp)")
plt.ylabel("Number of Passengers")
plt.title("Survival Based on Siblings/Spouses Aboard")
plt.show()


# barchart/countplot using the seaborn package to show the number of survivors based on siblings or spouses aboard
sns.countplot(data=titanic_cleaned, x="SibSp", hue="Survived")
plt.title("Survival Based on Siblings/Spouses Aboard")
plt.legend(["Deceased (0)", "Survival (1)"])
plt.show()

# stacked bar chart showing the sex of survivors based on sex
sex_of_survivors = sex_of_survivors.unstack()
sex_of_survivors.plot(kind="barh", stacked=True)
plt.xlabel("Number of Survivors")
plt.ylabel("Demographics on Sex", labelpad=15)
plt.title("Number of Survivors compared by sex")
plt.show()


# histogram showing the fare distribution among survivors
fare_of_survivors_df = fare_of_survivors.reset_index()

# Filter survivors where Survived == 1
fare_of_survivors_df = fare_of_survivors_df[fare_of_survivors_df["Survived"] == 1]

# Print the DataFrame structure
print(fare_of_survivors_df.describe())
print(fare_of_survivors_df.columns)


fare_of_survivors_df.columns = ["Survived", "Fare", "Count"]
bins = np.linspace(0, 513, 40)

fare_of_survivors_df['Fare'].plot(kind="hist", bins=bins, density=False, color="orange", edgecolor="black")
plt.title("Distribution of Fare Among Survivors")
plt.xlabel("Fare of Survivors")
plt.ylabel("Count of Survivors")
plt.show()

# violinplot using the fare of fare of survivors
sns.violinplot(x="Survived", y="Fare", data=fare_of_survivors_df)
plt.title("Fare Distribution by Survival")
plt.show()

# bar chart showing survivors with embarked location
embarked_of_survivors = embarked_of_survivors.unstack()

embarked_of_survivors.plot(kind='bar')

#bar chart showing embarked location
plt.xticks(rotation=0)
plt.xlabel("Groups depending on Embarked Location")
plt.ylabel("Number of Passengers")
plt.legend(["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"])
plt.title("Survival Based on Embarked Location")
plt.show()

# # bar chart based on number of parents and children
parents_children = parents_children.unstack()
parents_children.plot(kind="bar")
plt.xlabel("Groups based on Survival and number of children/parents")
plt.ylabel("Number of Survivors", labelpad=5)
plt.title("Number of Survivors based on Children/Parent status")
plt.xticks(rotation=0)
plt.show()