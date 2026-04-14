import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ---------------- CONFIG ----------------
CSV_PATH = r"E:\FinalData\7-images_tags_reach_cleaned.csv"

# ---------------- LOAD DATA ----------------
df = pd.read_csv(CSV_PATH)
print("✅ Data Loaded Successfully!")
print(f"Shape of Dataset: {df.shape}")
print(df.head())

# ---------------- CHECK MISSING & DUPLICATES ----------------
print("\n📌 Missing Values:")
print(df.isnull().sum())

print("\n📌 Duplicate Rows:", df.duplicated().sum())

# ---------------- CHECK HASHTAGS ----------------
if "Hashtags" in df.columns:
    df["Hashtags"] = df["Hashtags"].apply(lambda x: str(x).split(", "))
    df["HashtagCount"] = df["Hashtags"].apply(len)
    print(f"\n📌 Average Hashtags per Post: {df['HashtagCount'].mean():.2f}")
    print(f"📌 Max Hashtags in a Post: {df['HashtagCount'].max()}")

    # Distribution of hashtag counts
    plt.figure(figsize=(8, 4))
    sns.histplot(df["HashtagCount"], bins=20, kde=False, color='blue')
    plt.title("Distribution of Hashtags per Post")
    plt.xlabel("Hashtag Count")
    plt.ylabel("Frequency")
    plt.show()

    # Top hashtags
    all_tags = [tag for tags in df["Hashtags"] for tag in tags]
    top_tags = Counter(all_tags).most_common(20)
    tags_df = pd.DataFrame(top_tags, columns=["Hashtag", "Frequency"])

    print("\n📌 Top 20 Hashtags:")
    print(tags_df)

    # Plot Top hashtags
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Frequency", y="Hashtag", data=tags_df, palette="Blues_r")
    plt.title("Top 20 Most Frequent Hashtags")
    plt.show()

# ---------------- IMBALANCE CHECK ----------------
print("\n📌 Checking Hashtag Imbalance...")
unique_hashtags = set(all_tags)
tag_counts = Counter(all_tags)
imbalance_ratio = max(tag_counts.values()) / min(tag_counts.values())
print(f"Imbalance Ratio (Max/Min): {imbalance_ratio:.2f}")
print(f"Total Unique Hashtags: {len(unique_hashtags)}")

# Visualize full hashtag distribution
plt.figure(figsize=(12, 5))
sns.histplot(list(tag_counts.values()), bins=50, color='orange')
plt.title("Hashtag Frequency Distribution")
plt.xlabel("Frequency")
plt.ylabel("Number of Hashtags")
plt.show()

# ---------------- NUMERIC COLUMN DISTRIBUTION ----------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("\n📌 Numeric Columns:", numeric_cols)

for col in numeric_cols:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True, color='green')
    plt.title(f"Distribution of {col}")

    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[col], color='lightblue')
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.show()

# ---------------- OUTLIER DETECTION ----------------
print("\n📌 OUTLIERS DETECTION (IQR method):")
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"{col}: {len(outliers)} outliers")

# ---------------- CORRELATION HEATMAP ----------------
plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

print("\n✅ Data Quality Analysis Completed!")
