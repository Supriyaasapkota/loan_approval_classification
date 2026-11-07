
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("loan_approval.csv")
print(df.head())
print("Shape of dataset:", df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
print(df["loan_approved"].value_counts())



plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="loan_approved")
plt.title("Loan Approval Distribution")
plt.xlabel("Loan Approved")
plt.ylabel("Count")
plt.show()



plt.figure(figsize=(6, 4))
sns.histplot(data=df, x="income", kde=True)
plt.title("Income Distribution")
plt.xlabel("Income")
plt.show()



plt.figure(figsize=(6, 4))
sns.histplot(data=df, x="credit_score", kde=True)
plt.title("Credit Score Distribution")
plt.xlabel("Credit Score")
plt.show()


plt.figure(figsize=(6, 4))
sns.boxplot(data=df, y="loan_amount")
plt.title("Loan Amount Box Plot")
plt.show()



plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="loan_approved", y="credit_score")
plt.title("Credit Score vs Loan Approval")
plt.show()



plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="loan_approved", y="income")
plt.title("Income vs Loan Approval")
plt.show()


plt.figure(figsize=(8, 5))
sns.heatmap(df.select_dtypes(include="number").corr(),
            annot=True, cmap="RdYlGn")
plt.title("Correlation Heatmap")
plt.show()
