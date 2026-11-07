
import pandas as pd

df = pd.read_csv("loan_approval.csv")

print("Original Dataset Shape:", df.shape)

df = df.drop(columns=["name"])


print("\nMissing Values:")
print(df.isnull().sum())
df = df.drop_duplicates()

print("\nCleaned Dataset Shape:", df.shape)
print("\nDataset Information:")
print(df.info())

df.to_csv("loan_approval_cleaned.csv", index=False)
print("\nCleaned dataset saved successfully!")
