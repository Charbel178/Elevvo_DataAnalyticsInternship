# Task 2 - Titanic Dataset

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Loading the dataset

df = pd.read_csv(r"C:\Users\LAPTOP\Desktop\Non-uni Courses\Elevvo Internship\Task 2\titanic.csv")

# Step 2: Exploring the dataset

print("First 5 rows: ")
print(df.head())
print()
print("Dataset Info: ")
print(df.info())
print()

# Step 3: Cleaning the data

df['Age'] = df['Age'].fillna(df['Age'].median())

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

if 'Cabin' in df.columns:
    df.drop(columns=['Cabin'], inplace=True)

df['Pclass'] = df['Pclass'].astype('category')

# Step 4: Statistics & Insights

print("\n Survival Rate by Gender")
print(df.groupby('Sex')['Survived'].mean())

print("\n Survival Rate by Class")
print(df.groupby('Pclass', observed=True)['Survived'].mean())

print("\n Survival Rate by Gender and Class")
print(df.groupby(['Sex', 'Pclass'], observed=True)['Survived'].mean())

# Step 5: Visualization

plt.figure(figsize=(6,4))
sns.barplot(x='Sex', y='Survived', data=df)
plt.title("Survival Rate by Gender")
plt.show()

plt.figure(figsize=(6,4))
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title("Survival Rate by Class")
plt.show()

# Step 6: Bonus Visualizations

# Correlation Heatmap

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Survival by Gender & Class

sns.catplot(x='Pclass', y='Survived', hue='Sex', kind='bar', data=df, height=5, aspect=1.2)
plt.title("Survival Rate by Gender and Class")
plt.show()