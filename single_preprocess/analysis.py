import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from matplotlib.backends.backend_pdf import PdfPages
from itertools import combinations
import re

sns.set(style="whitegrid")

# =======================  Load Dataset  =======================
csv_file_path = r"E:\FinalData\6-images_tags_reach.csv"
df = pd.read_csv(csv_file_path)

# Validate Columns
required_cols = ['Username', 'Category', '#Followers', '#Followees', 'Likes', 'Comments', 'Hashtags']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in CSV: {missing_cols}")

# =======================  PDF Setup  =======================
pdf_path = r"E:\FinalData\analysis_report.pdf"
pdf = PdfPages(pdf_path)

try:
    # ======================= 0. Dataset Overview =======================
    fig, ax = plt.subplots(figsize=(10, 6))
    text = f"""
    Dataset Overview
    -------------------------
    - Number of Images      : {len(df)}
    - Number of Unique Users: {df['Username'].nunique()}
    - Number of Categories  : {df['Category'].nunique()}
    - Categories            : {', '.join(df['Category'].dropna().unique())}
    """
    ax.text(0.01, 0.6, text, fontsize=12, fontfamily="monospace")
    ax.axis('off')
    pdf.savefig(fig)
    plt.close()

    # ======================= 1. Category Distribution (Pie) =======================
    fig = plt.figure(figsize=(6, 6))
    category_counts = df['Category'].value_counts()
    category_counts.plot.pie(autopct='%1.1f%%', startangle=140, shadow=True)
    plt.title("Category Distribution (Pie Chart)")
    plt.ylabel('')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # ======================= 2. Number of Images per Category =======================
    fig = plt.figure(figsize=(8, 5))
    sns.countplot(x='Category', data=df, order=category_counts.head(20).index)
    plt.title("Number of Images per Category (Top 20)")
    plt.xlabel("Category")
    plt.ylabel("Image Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # ======================= Clean Numerical Columns =======================
    for col in ['#Followers', '#Followees', 'Likes', 'Comments']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # ======================= 3. Followers Distribution =======================
    followers_filtered = df['#Followers'].dropna()
    followers_filtered = followers_filtered[followers_filtered < followers_filtered.quantile(1)]
    fig = plt.figure(figsize=(8, 5))
    sns.histplot(followers_filtered, kde=True, bins=30)
    plt.title("Distribution of Followers")
    plt.xlabel("Number of Followers")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()


    # ======================= 7. Hashtag Extraction =======================
    hashtag_list = df['Hashtags'].dropna().apply(lambda x: re.findall(r'#\w+', str(x)))
    df['Hashtag_Count'] = hashtag_list.apply(len)

    # ======================= 8. Hashtag Count Distribution =======================
    fig = plt.figure(figsize=(8, 5))
    sns.histplot(df['Hashtag_Count'], kde=False, bins=20)
    plt.title("Number of Hashtags per Post")
    plt.xlabel("Hashtag Count")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # ======================= 11. Correlation: Hashtag Count vs Likes/Comments =======================
    fig = plt.figure(figsize=(8, 5))
    sns.scatterplot(x='Hashtag_Count', y='Likes', data=df)
    plt.title("Hashtag Count vs Likes")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    fig = plt.figure(figsize=(8, 5))
    sns.scatterplot(x='Hashtag_Count', y='Comments', data=df)
    plt.title("Hashtag Count vs Comments")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # ======================= 12. User-wise Hashtag Diversity =======================
    user_diversity = hashtag_list.groupby(df['Username']).apply(lambda tags: len(set(tag for lst in tags for tag in lst)))
    fig = user_diversity.sort_values(ascending=False)[:20].plot(kind='bar', figsize=(12, 6), title="User-wise Hashtag Diversity (Top 20)").get_figure()
    plt.xlabel("Username")
    plt.ylabel("Unique Hashtags Used")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # ======================= 13. Hashtag Co-occurrence Matrix =======================
    co_matrix = defaultdict(Counter)
    for tags in hashtag_list:
        for tag1, tag2 in combinations(set(tags), 2):
            co_matrix[tag1][tag2] += 1
            co_matrix[tag2][tag1] += 1

    # Top 20 hashtags
    all_tags = [tag for tags in hashtag_list for tag in tags]
    top_hashtags = Counter(all_tags).most_common(20)
    top_tags = [tag for tag, _ in top_hashtags]

    # Build co-occurrence DataFrame
    co_df = pd.DataFrame(index=top_tags, columns=top_tags).fillna(0)
    for t1 in top_tags:
        for t2 in top_tags:
            co_df.loc[t1, t2] = co_matrix[t1][t2]

    # Plot heatmap
    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(co_df.astype(int), cmap="Blues", annot=False)
    plt.title("Co-occurrence Heatmap of Top 20 Hashtags")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

finally:
    pdf.close()
    print(f"✅ PDF report saved at: {pdf_path}")
