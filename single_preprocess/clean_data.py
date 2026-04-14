import pandas as pd
import numpy as np

# -------------------- CONFIG --------------------
INPUT_CSV = r"E:\FinalData\6-images_tags_reach.csv"  # Original dataset
OUTPUT_CSV = r"E:\FinalData\7-images_tags_reach_cleaned.csv"  # Cleaned dataset path

NUMERIC_COLUMNS = ["#Followers", "#Followees", "Likes", "Comments", "HashtagCount", "IHC_h"]

# -------------------- FUNCTIONS --------------------
def remove_duplicates(df):
    initial_count = len(df)
    df = df.drop_duplicates()
    print(f"✅ Removed {initial_count - len(df)} duplicate rows")
    return df

def handle_outliers_iqr(df, numeric_cols):
    """
    Handles outliers by Winsorizing using the IQR method.
    Caps values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] to the boundary values.
    """
    for col in numeric_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

            # Cap values instead of removing rows
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

            print(f"✅ {col}: {outliers_count} outliers capped (Bounds: {lower_bound:.2f}, {upper_bound:.2f})")

    return df

# -------------------- MAIN PROCESS --------------------
if __name__ == "__main__":
    print("📂 Loading dataset...")
    df = pd.read_csv(INPUT_CSV)
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Remove duplicates
    df = remove_duplicates(df)

    # Handle numeric outliers
    df = handle_outliers_iqr(df, NUMERIC_COLUMNS)

    # Save cleaned dataset
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Cleaned dataset saved at: {OUTPUT_CSV}")
    print(f"🔍 Final shape: {df.shape}")
