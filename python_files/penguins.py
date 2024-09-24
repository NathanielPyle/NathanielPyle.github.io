import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# using a basic dataset to explore the functionality of Pandas
penguins = sns.load_dataset('penguins')

# Basic description of all columns and all variables
print(penguins.describe)

# # there are three species of penguins with unequal distributions
value_counts = penguins['species'].value_counts()
print(value_counts)

# analysing the shape of the dataset, there are 7 columns with 344 rows
print(penguins.shape)

# this did not remove the specified columns if they had NaN values
# therefore, any was used instead of all
##valid_data = penguins.dropna(how ='all')

valid_data = penguins.dropna(how ='any')


#throught the removal of messy rows, the structure of the data
# is now 333 rows with 7 columns
print(valid_data.shape)
print(valid_data.count())

valid_data.plot()
plt.title("Variation upon Penguins physical features")
plt.xlabel("Count of Penguins")
plt.ylabel("Measurement")
plt.show()

print(valid_data.describe)

### analysing the different number of penguin species for
## the clean dataset
value_counts = valid_data['species'].value_counts()
print(value_counts)

# number of penguins as per island
value_counts = valid_data['island'].value_counts()
print(value_counts)

# number of male versus female penguins
value_counts = (valid_data['sex'] == "Male").value_counts()
print(value_counts)

##analysing the number of penguins with flipper length vs
## body mass, there are two major clusters for male and female
## peguins. male penguins who are under 5000 grams and female
## penguins under 4000 grams
penguins_male = valid_data[valid_data['sex'] == 'Male']
print(penguins_male.describe())

plt.scatter(penguins_male['flipper_length_mm'], penguins_male['body_mass_g'], color = "blue")
plt.xlabel('flipper length (mm)')
plt.ylabel('Body Mass (g)')
plt.title('Body Mass vs Flipper Length for Male Penguins')
plt.show()



penguins_female = valid_data[valid_data['sex'] == 'Female']
print(penguins_female.describe())

plt.scatter(penguins_female['flipper_length_mm'], penguins_female['body_mass_g'], color = "red")
plt.xlabel('flipper length (mm)')
plt.ylabel('Body Mass (g)')
plt.title('Body Mass vs Flipper Length for Female Penguins')
plt.show()

## as a result of these clusters, male and female penguins were seperated into two further classes
## male penguins under 5000 grams, and those 5000 grams or over

penguins_male_u5000 = penguins_male[penguins_male["body_mass_g"] < 5000]
penguins_male_o5000 = penguins_male[penguins_male["body_mass_g"] >= 5000]

## using these following strategies was the easiest way of comparing the two samples of
## penguins

print(penguins_male_u5000.describe)
print(penguins_male_o5000.describe)

print(penguins_male_u5000.describe())
print(penguins_male_o5000.describe())

print(penguins_male_u5000.shape)
print(penguins_male_o5000.shape)

## trying to compare two different dataframes with differing shapes, will result in an error. this was attempted and
## unfortunately didn't work
##penguins_male_u5000.compare(penguins_male_o5000, align_axis=1, keep_shape=False, keep_equal=False)

## this also failed to work...
##penguins_male_u5000.compare(penguins_male_o5000)

#boxplots to show the distribution of weight in the two subpopulations
penguins_male_u5000["body_mass_g"].plot.box()
plt.ylabel('Body Mass (g)')
plt.title('Male Penguins Under 5000 grams boxplot')
plt.show()

penguins_male_o5000["body_mass_g"].plot.box()
plt.ylabel('Body Mass (g)')
plt.title('Male Penguins Over 5000 grams boxplot')
plt.show()


## a similar situation was used for female penguins over 4000 grams and those 4000 grams and under

penguins_female_u4000 = penguins_female[penguins_female["body_mass_g"] < 4000]
penguins_female_o4000 = penguins_female[penguins_female["body_mass_g"] >= 4000]


print(penguins_female_u4000.describe())
print(penguins_female_o4000.describe())


print(penguins_female_u4000.describe)
print(penguins_female_o4000.describe)

print(penguins_female_u4000.shape)
print(penguins_female_o4000.shape)


penguins_female_u4000["body_mass_g"].plot.box()
plt.ylabel('Body Mass (g)')
plt.title('Female Penguins Under 4000 grams boxplot')
plt.show()

penguins_female_o4000["body_mass_g"].plot.box()
plt.ylabel('Body Mass (g)')
plt.title('Female Penguins Over 4000 grams boxplot')
plt.show()


## as there is not much information one can obtain from these basic plots, one should concluded that specific samplings
## should be used. These samplings will be based on species, island and the sex of the penguin.

adelie_biscoe_male = valid_data[(valid_data['species'] == "Adelie") &
                                (valid_data["island"] == "Biscoe") &
                                (valid_data["sex"] == "Male")]

print(adelie_biscoe_male.describe())


## this did not provide much information about the distribution of penguin's body mass. this was similar to the previous
## results.
plt.scatter(adelie_biscoe_male['flipper_length_mm'], adelie_biscoe_male['body_mass_g'], color = "blue")
plt.xlabel('flipper length (mm)')
plt.ylabel('Body Mass (g)')
plt.title('Specified search Body Mass vs Flipper Length for Male Penguins')
plt.show()

adelie_biscoe_male["body_mass_g"].plot.box()
plt.ylabel('Body Mass (g)')
plt.title('Specified Boxplot Body Mass vs Flipper Length for Male Penguins')
plt.show()

# Generate pairplots using species
sns.pairplot(valid_data, hue='species')
plt.show()

#Pairplots using sex
sns.pairplot(valid_data, hue='sex')
plt.show()

#Pairplots using island
sns.pairplot(valid_data, hue='island')
plt.show()

# Pairplots using flipper length and body mass with hue for species
sns.pairplot(valid_data[['flipper_length_mm', 'body_mass_g', 'species']], hue='species')
plt.show()

# Pairplots using bill length and body mass with hue for species
sns.pairplot(valid_data[['bill_length_mm', 'body_mass_g', 'species']], hue='species')
plt.show()

# Pairplots using bill depth and body mass with hue for species
sns.pairplot(valid_data[['bill_depth_mm', 'body_mass_g', 'species']], hue='species')
plt.show()
