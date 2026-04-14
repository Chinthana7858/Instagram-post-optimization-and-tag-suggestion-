import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from collections import defaultdict, Counter
from tqdm import tqdm

# --- File Paths ---
input_file = r"E:\ProcessedV3\6-images_tags_reach.csv"
allowed_hashtag_file =  r"E:\ProcessedV3\7-hashtags_occurrences_100_plus.xlsx"
output_file = r"E:\ProcessedV3\User_Hashtag_Frequency_Matrix"

# --- Load Hashtags to Keep ---
print("📥 Loading allowed hashtags...")
allowed_df = pd.read_excel(allowed_hashtag_file)
allowed_hashtags = set(allowed_df['Hashtag'].astype(str).str.strip())
print(f"✅ Loaded {len(allowed_hashtags)} allowed hashtags.")

# --- Load Dataset ---
print("📥 Loading post dataset...")
df = pd.read_csv(input_file)
df["Hashtags"] = df["Hashtags"].apply(lambda x: str(x).split(', '))
print(f"✅ Loaded {len(df)} rows.")

# --- Filter hashtags per post ---
print("🔍 Filtering hashtags to allowed set...")
df["Hashtags"] = df["Hashtags"].apply(lambda tags: [tag for tag in tags if tag in allowed_hashtags])

# --- Build user-hashtag dictionary ---
print("🔄 Building user-hashtag dictionary...")
user_hashtag_dict = defaultdict(list)
for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
    user_hashtag_dict[row['Username']].extend(row['Hashtags'])

print(f"✅ Found {len(user_hashtag_dict)} unique users.")

# --- Prepare frequency matrix manually ---
print("🔢 Counting hashtag frequencies per user...")
usernames = list(user_hashtag_dict.keys())
hashtags_per_user = list(user_hashtag_dict.values())
hashtag_list = sorted(allowed_hashtags)
user_hashtag_matrix = []

for tags in tqdm(hashtags_per_user, desc="Building frequency rows"):
    tag_counts = Counter(tags)
    row = [tag_counts.get(tag, 0) for tag in hashtag_list]
    user_hashtag_matrix.append(row)

# --- Create DataFrame ---
print("🧱 Building DataFrame...")
user_hashtag_df = pd.DataFrame(user_hashtag_matrix, columns=hashtag_list)
user_hashtag_df.insert(0, "Username", usernames)

# --- Save to CSV ---
print("💾 Saving to CSV...")
user_hashtag_df.to_csv(output_file + ".csv", index=False)
print(f"✅ User-Hashtag Frequency Matrix saved to: {output_file}.csv")
