
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "tourism_project/data/tourism.csv"

df = pd.read_csv(RAW_PATH)

# Remove unnecessary columns
df.drop(columns=["Unnamed: 0", "CustomerID"], inplace=True)

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# Stratified split to preserve target distribution
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Save train/test splits
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("Training feature shape:", Xtrain.shape)
print("Testing feature shape:", Xtest.shape)
print("Training target shape:", ytrain.shape)
print("Testing target shape:", ytest.shape)

print("\nProdTaken distribution in training data:")
print(ytrain.value_counts())

print("\nProdTaken distribution in testing data:")
print(ytest.value_counts())
