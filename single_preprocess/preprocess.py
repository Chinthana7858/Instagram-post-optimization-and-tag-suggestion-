import os
import re
import json
import shutil
import ast

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter

# ================== CONFIGURATION ==================
BASE_DIR = r"E:\FinalData"
DATASET_DIR = r"E:\Dataset"
IMAGE_DIR = r"E:\Dataset\images"
JSON_DIR = r"E:\Dataset\posts\posts"

FILES = {
    "influencers": os.path.join(DATASET_DIR, "influencers.csv"),
    "mapping": os.path.join(DATASET_DIR, "JSON-Image_files_mapping.txt"),
    "selected_users": os.path.join(BASE_DIR, "1-selected_users.csv"),
    "selected_posts_mapping": os.path.join(BASE_DIR, "2-selected_posts_mapping.csv"),
    "enriched_posts": os.path.join(BASE_DIR, "3-selected_posts_enriched.csv"),
    "cleaned_posts": os.path.join(BASE_DIR, "4-cleaned_selected_posts.csv"),
    "images_with_tags": os.path.join(BASE_DIR, "5-images_with_tags.csv"),
    "images_tags_reach": os.path.join(BASE_DIR, "6-images_tags_reach.csv"),
    "selected_images_dir": os.path.join(BASE_DIR, "selected_images"),
    "top_hashtags_xlsx": os.path.join(BASE_DIR, "7-hashtags_occurrences_100_plus.xlsx"),
    "user_hashtag_matrix": os.path.join(BASE_DIR, "User_Hashtag_Frequency_Matrix.csv"),
}

NUMBER_OF_USERS_PER_CATEGORY=500;
NUMBER_OF_POSTS_PER_USER=100;
# ===================================================

def step1_select_users():
    df = pd.read_csv(FILES["influencers"])
    df_selected = df.groupby("Category", group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), NUMBER_OF_USERS_PER_CATEGORY), random_state=42)
    ).reset_index(drop=True)
    df_selected.to_csv(FILES["selected_users"], index=False)
    print(f"[Step 1] ✅ Saved: {FILES['selected_users']}")


def step2_select_posts():
    if os.path.exists(FILES["selected_posts_mapping"]):
        try:
            os.remove(FILES["selected_posts_mapping"])
        except PermissionError:
            print("❌ Close the selected_posts_mapping file and retry.")
            return
    df_users = pd.read_csv(FILES["selected_users"])
    df_posts = pd.read_csv(FILES["mapping"], sep="\t",
                           names=["influencer_name", "JSON_PostMetadata_file_name", "Image_file_name"], skiprows=1)
    df_merged = df_posts.merge(df_users, left_on="influencer_name", right_on="Username", how="inner")
    df_selected = df_merged.groupby("influencer_name", group_keys=False).apply(
        lambda x: x.sample(n=min(NUMBER_OF_POSTS_PER_USER, len(x)), random_state=42), include_groups=False
    ).reset_index(drop=True)
    df_selected.to_csv(FILES["selected_posts_mapping"], index=False)
    print(f"[Step 2] ✅ Saved: {FILES['selected_posts_mapping']}")


def step3_enrich_posts():
    df = pd.read_csv(FILES["selected_posts_mapping"])
    likes, comments, hashtags = [], [], []

    for _, row in df.iterrows():
        uname = row["Username"]
        post_id = row["JSON_PostMetadata_file_name"].replace(".info", "")
        json_file = os.path.join(JSON_DIR, f"{uname}-{post_id}.info")

        if os.path.exists(json_file) and os.path.getsize(json_file) > 0:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                likes.append(data.get("edge_media_preview_like", {}).get("count", 0))
                comments.append(data.get("edge_media_preview_comment", {}).get("count", 0))
                caption = data.get("edge_media_to_caption", {}).get("edges", [{}])
                text = caption[0].get("node", {}).get("text", "") if caption else ""
                tags = [w for w in text.split() if w.startswith("#")]
                hashtags.append(", ".join(tags))
            except:
                likes.append(0);
                comments.append(0);
                hashtags.append("")
        else:
            likes.append(0);
            comments.append(0);
            hashtags.append("")

    df["Likes"] = likes
    df["Comments"] = comments
    df["Hashtags"] = hashtags
    df.to_csv(FILES["enriched_posts"], index=False)
    print(f"[Step 3] ✅ Saved: {FILES['enriched_posts']}")


def step4_clean_hashtags():
    df = pd.read_csv(FILES["enriched_posts"])

    def process_hashtags(text):
        if pd.isna(text) or text.strip() == "": return None
        hashtags = [ht.strip() for ht in text.split(",") if re.match(r"^#[A-Za-z]+$", ht.strip())]
        return hashtags if hashtags else None

    df["Hashtags"] = df["Hashtags"].astype(str).apply(process_hashtags)
    df_clean = df.dropna(subset=["Hashtags"])
    df_clean.to_csv(FILES["cleaned_posts"], index=False)
    print(f"[Step 4] ✅ Saved: {FILES['cleaned_posts']} | {len(df_clean)} rows retained")


def step5_expand_image_tags():
    df = pd.read_csv(FILES["cleaned_posts"])
    rows = []
    for _, row in df.iterrows():
        try:
            images = ast.literal_eval(row["Image_file_name"])
            hashtags = ast.literal_eval(str(row["Hashtags"]))
            for img in images:
                rows.append([img, row["Username"], row["Category"], row["#Followers"], row["#Followees"], row["Likes"],
                             row["Comments"], ", ".join(hashtags)])
        except:
            continue
    columns = ["Image_file_name", "Username", "Category", "#Followers", "#Followees", "Likes", "Comments", "Hashtags"]
    df_expanded = pd.DataFrame(rows, columns=columns)
    df_expanded.to_csv(FILES["images_with_tags"], index=False)
    print(f"[Step 5] ✅ Saved: {FILES['images_with_tags']}")


def count_hashtags(ht_text):
    if pd.isna(ht_text):
        return 0
    try:
        if ht_text.startswith("["):  # JSON list format
            tags = ast.literal_eval(ht_text)
        else:
            tags = ht_text.split(", ")
        return len([t for t in tags if t.strip()])
    except:
        return 0


def step6_compute_reach():
    df = pd.read_csv(FILES["images_with_tags"])
    df["HashtagCount"] = df["Hashtags"].apply(count_hashtags)

    # Apply the improved formula
    df["IHC_h"] = ((df["Likes"] + 2 * df["Comments"]) / df["#Followers"]) * \
                  (1 + 0.2 * np.log1p(df["HashtagCount"])) * 100

    df.to_csv(FILES["images_tags_reach"], index=False)
    print(f"[Step 6] ✅ Saved: {FILES['images_tags_reach']}")


def step7_copy_selected_images():
    df = pd.read_csv(FILES["images_tags_reach"])
    os.makedirs(FILES["selected_images_dir"], exist_ok=True)

    def extract_num(name):
        match = re.search(r"(\d+)\.jpg", name)
        return match.group(1) if match else None

    image_dict = {
        extract_num(f): f for f in os.listdir(IMAGE_DIR) if extract_num(f)
    }

    missing = []
    for _, row in df.iterrows():
        filename = row["Image_file_name"]
        num = extract_num(filename)
        actual = image_dict.get(num)
        if actual:
            src = os.path.join(IMAGE_DIR, actual)
            dst = os.path.join(FILES["selected_images_dir"], filename)
            if os.path.exists(src): shutil.copy(src, dst)
        else:
            missing.append(filename)

    if missing:
        with open(os.path.join(FILES["selected_images_dir"], "missing_images.txt"), "w") as f:
            for m in missing: f.write(m + "\n")
    print(f"[Step 7] ✅ Copied available images to: {FILES['selected_images_dir']}")


def step8_generate_top_hashtags():
    df = pd.read_csv(FILES["images_tags_reach"])
    df["Hashtags"] = df["Hashtags"].astype(str)
    counter = Counter()
    for h in df["Hashtags"]:
        try:
            tags = ast.literal_eval(h) if h.startswith("[") else h.split(", ")
            for tag in tags:
                tag = tag.strip()
                if tag: counter[tag] += 1
        except:
            continue

    filtered = {k: v for k, v in counter.items() if v > 100}
    df_tags = pd.DataFrame(list(filtered.items()), columns=["Hashtag", "Occurrence"]).sort_values(by="Occurrence",
                                                                                                  ascending=False)
    df_tags.to_excel(FILES["top_hashtags_xlsx"], index=False)
    print(f"[Step 8] ✅ Saved: {FILES['top_hashtags_xlsx']}")


def step9_generate_user_hashtag_matrix():
    allowed_df = pd.read_excel(FILES["top_hashtags_xlsx"])
    allowed = set(allowed_df["Hashtag"].astype(str).str.strip())
    df = pd.read_csv(FILES["images_tags_reach"])
    df["Hashtags"] = df["Hashtags"].apply(lambda x: [t for t in str(x).split(", ") if t in allowed])

    user_tags = defaultdict(list)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building user-tag matrix"):
        user_tags[row["Username"]].extend(row["Hashtags"])

    tags = sorted(allowed)
    matrix = []
    for user in tqdm(user_tags, desc="Counting hashtags"):
        count = Counter(user_tags[user])
        row = [count.get(tag, 0) for tag in tags]
        matrix.append([user] + row)

    final_df = pd.DataFrame(matrix, columns=["Username"] + tags)
    final_df.to_csv(FILES["user_hashtag_matrix"], index=False)
    print(f"[Step 9] ✅ Saved: {FILES['user_hashtag_matrix']}")


def main():
    step1_select_users()
    step2_select_posts()
    step3_enrich_posts()
    step4_clean_hashtags()
    step5_expand_image_tags()
    step6_compute_reach()
    step7_copy_selected_images()
    step8_generate_top_hashtags()
    step9_generate_user_hashtag_matrix()


if __name__ == "__main__":
    main()
