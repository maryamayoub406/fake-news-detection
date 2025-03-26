import pandas as pd

# Load datasets
try:
    fake_news = pd.read_csv("dataset/Fake.csv")
    real_news = pd.read_csv("dataset/True.csv")

    print("✅ Successfully loaded Fake.csv and True.csv")
except FileNotFoundError:
    print("❌ Fake.csv or True.csv not found! Check dataset folder.")
    exit()

# Add labels (1 = Real, 0 = Fake)
fake_news["label"] = 0
real_news["label"] = 1

# Combine the datasets
df = pd.concat([fake_news, real_news])

# Shuffle data
df = df.sample(frac=1).reset_index(drop=True)

# Save cleaned dataset
csv_path = "dataset/cleaned_data.csv"
try:
    df.to_csv(csv_path, index=False)
    print(f"✅ File saved at: {csv_path}")
except Exception as e:
    print(f"❌ Error saving file: {e}")

print(df.head())  # Display first few rows
print(df.info())  # Display dataset info
