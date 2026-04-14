import os
import itertools

import numpy as np
import torch
import pickle
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
import random

from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42)
torch.manual_seed(42)

# ---------------------- Configuration ----------------------
CSV_PATH = r"E:\ProcessedV3\6-images_tags_reach.csv"
IMG_PATH = r"E:\ProcessedV3\selected_images"
USER_CSV = r"E:\ProcessedV3\User_Hashtag_Frequency_Matrix.csv"
MLB_PATH = r"E:\ProcessedV3\mlb.pkl"
MODEL_A_PATH = r"E:\ProcessedV3\MultiModels\resnet50_multimodel.pth"
MODEL_B_PATH = r"E:\ProcessedV3\MultiModels\ihc_dualbranch_model.pth"
OUTPUT_CSV = r"E:\ProcessedV3\evaluation_output.csv"
TOP_K = 5
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 7

# ---------------------- Transforms ----------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------------------- Load Metadata with Filtering ----------------------
MIN_TAG_COUNT = 100


def prepare_data(csv_path, min_count=100, mlb_path=None):
    df = pd.read_csv(csv_path)

    # Store original hashtags
    df["Original_Hashtags"] = df["Hashtags"]

    # Convert hashtags to list
    df["Hashtags"] = df["Hashtags"].apply(lambda x: str(x).split(", "))

    # Apply filtering
    all_tags = [tag for tags in df["Hashtags"] for tag in tags]
    tag_counts = pd.Series(all_tags).value_counts()
    selected_tags = tag_counts[tag_counts >= min_count].index.tolist()
    df["Hashtags"] = df["Hashtags"].apply(lambda tags: [tag for tag in tags if tag in selected_tags])
    df = df[df["Hashtags"].map(len) > 0]

    # MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    mlb.fit(df["Hashtags"])
    if mlb_path:
        with open(mlb_path, "wb") as f:
            pickle.dump(mlb, f)

    # Scale IHC_h
    scaler = StandardScaler()
    df["IHC_h"] = scaler.fit_transform(df[["IHC_h"]])
    with open("ihc_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return df, mlb, scaler


user_matrix_df = pd.read_csv(USER_CSV).set_index("Username")
user_dim = user_matrix_df.shape[1]


# ---------------------- Dataset ----------------------
class HashtagIhcDataset(Dataset):
    def __init__(self, df, mlb, scaler):
        self.df = df.reset_index(drop=True)
        self.mlb = mlb
        self.scaler = scaler

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tag_vector = torch.tensor(self.mlb.transform([row["Hashtags"]])[0], dtype=torch.float32)
        ihc_score = torch.tensor(row["IHC_h"], dtype=torch.float32)
        return tag_vector, ihc_score


class ImageUserDataset(Dataset):
    def __init__(self, df, user_matrix, img_path, transform, mlb):
        self.df = df.reset_index(drop=True)
        self.user_matrix = user_matrix
        self.img_path = img_path
        self.transform = transform
        self.mlb = mlb

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["Image_file_name"]
        username = row["Username"]
        hashtags = row["Hashtags"]

        try:
            image = Image.open(os.path.join(self.img_path, img_name)).convert("RGB")
            image = self.transform(image)
        except:
            image = torch.zeros((3, IMG_SIZE, IMG_SIZE))  # fallback

        user_vec = torch.tensor(self.user_matrix.loc[username].values, dtype=torch.float32)
        label = torch.tensor(self.mlb.transform([hashtags])[0], dtype=torch.float32)
        return image, user_vec, label



def train_model_a(train_df, val_df, mlb, num_tags):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = ImageUserDataset(train_df, user_matrix_df, IMG_PATH, transform, mlb)
    val_ds = ImageUserDataset(val_df, user_matrix_df, IMG_PATH, transform, mlb)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = ResNetWithUserEmbedding(user_dim, num_tags).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        # Training
        model.train()
        total_train_loss = 0
        for img, user_vec, labels in train_loader:
            img, user_vec, labels = img.to(device), user_vec.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(img, user_vec)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for img, user_vec, labels in val_loader:
                img, user_vec, labels = img.to(device), user_vec.to(device), labels.to(device)
                output = model(img, user_vec)
                loss = criterion(output, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save Model
    torch.save(model.state_dict(), MODEL_A_PATH)
    print(f"✅ Hashtag prediction model saved to {MODEL_A_PATH}")

    # Plot Loss Curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, EPOCHS + 1), train_losses, marker='o', label='Train Loss', color='blue')
    plt.plot(range(1, EPOCHS + 1), val_losses, marker='s', label='Validation Loss', color='orange')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("model_a_loss_curve.png")
    print("📊 Loss curve saved as model_a_loss_curve.png")

    return model


# ---------------------- Model B (IHC Predictor) ----------------------
class DualBranchIHCModel(nn.Module):
    def __init__(self, num_tags, num_numeric=2):
        super().__init__()

        # Hashtag branch
        self.hashtag_branch = nn.Sequential(
            nn.Linear(num_tags, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Numeric branch (followers + hashtag_count)
        self.numeric_branch = nn.Sequential(
            nn.Linear(num_numeric, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )

        # Combined branch
        self.combined = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, tags, numeric):
        tag_out = self.hashtag_branch(tags)
        num_out = self.numeric_branch(numeric)
        combined = torch.cat([tag_out, num_out], dim=1)
        return self.combined(combined)

# ---------------------- Model A (Hashtag Predictor) ----------------------
class CooccurrenceHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cooc_matrix = nn.Parameter(torch.eye(num_classes))

    def forward(self, logits):
        return logits @ self.cooc_matrix

class ResNetWithUserEmbedding(nn.Module):
    def __init__(self, user_out, num_classes, user_weight=1.0):
        super().__init__()
        base = models.resnet50(weights=None)
        base.fc = nn.Identity()
        self.resnet = base
        self.user_proj = nn.Sequential(
            nn.Linear(user_out, 256),
            nn.ReLU(), nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
        self.cooc = CooccurrenceHead(num_classes)
        self.user_weight = user_weight

    def forward(self, img, user_feat):
        img_feat = self.resnet(img)
        user_feat = self.user_proj(user_feat) * self.user_weight
        x = torch.cat([img_feat, user_feat], dim=1)
        return self.cooc(self.classifier(x))

# ---------------------- Inference + Evaluation ----------------------
def evaluate(model_a, model_b, val_df, TAGS, num_tags, follower_scaler, hashtag_count_scaler, ihc_scaler):
    device = DEVICE
    model_a.eval()
    model_b.eval()
    results = []

    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="🔍 Evaluating"):
        username = row["Username"]
        image_file = row["Image_file_name"]
        original_tags = row["Original_Hashtags"]
        filtered_tags = row["Hashtags"]
        prev_ihc = ihc_scaler.inverse_transform([[row["IHC_h"]]])[0][0]  # Convert back to original scale

        try:
            img = Image.open(os.path.join(IMG_PATH, image_file)).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
        except:
            continue

        if username not in user_matrix_df.index:
            continue

        user_vector = torch.tensor(user_matrix_df.loc[username].values, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            # Predict hashtag probabilities
            probs = torch.sigmoid(model_a(img_tensor, user_vector))[0]
            topk_indices = torch.topk(probs, TOP_K).indices.cpu().tolist()

            # Generate all subsets of top-k hashtags
            all_combos = []
            for r in range(1, len(topk_indices) + 1):
                all_combos.extend(itertools.combinations(topk_indices, r))

            combo_tensors = torch.zeros((len(all_combos), num_tags), device=device)
            for i, combo in enumerate(all_combos):
                combo_tensors[i, list(combo)] = 1

            # Prepare numeric features
            followers = np.log1p(row["#Followers"])
            followers = follower_scaler.transform(pd.DataFrame([[followers]], columns=["#Followers"]))[0][0]

            numeric_features = []
            for combo in all_combos:
                hashtag_count = len(combo)
                hashtag_count = np.log1p(hashtag_count)
                hashtag_count = hashtag_count_scaler.transform(pd.DataFrame([[hashtag_count]], columns=["HashtagCount"]))[0][0]
                numeric_features.append([followers, hashtag_count])

            numeric_tensor = torch.tensor(numeric_features, dtype=torch.float32, device=device)

            # Predict scores
            ihc_scores = model_b(combo_tensors, numeric_tensor).squeeze().detach().cpu().numpy()

            best_idx = np.argmax(ihc_scores)
            best_score = ihc_scaler.inverse_transform([[ihc_scores[best_idx]]])[0][0]
            # Top-K hashtags predicted by Model A
            predicted_hashtags = [TAGS[i] for i in topk_indices]

            # Best subset chosen by IHC optimization
            optimized_subset = [TAGS[i] for i in all_combos[best_idx]]

            results.append({
                "Image": image_file,
                "Username": username,
                "Original_Hashtags": original_tags,
                "Predicted_Hashtags": ", ".join(predicted_hashtags),  # Full Top-K predicted
                "Filtered_Hashtags": ", ".join(optimized_subset),  # Subset that maximized IHC
                "Previous_IHC_h": prev_ihc,
                "Predicted_IHC_h": best_score
            })

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Results saved to {OUTPUT_CSV}")



# ---------------------- Main Entry ----------------------
def main():

    df, mlb, scaler = prepare_data(CSV_PATH, MIN_TAG_COUNT, MLB_PATH)
    TAGS = mlb.classes_
    NUM_TAGS = len(TAGS)
    print(f"✅ Number of filtered hashtags: {NUM_TAGS}")

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🚀 Loading or Training Model A (Hashtag Predictor)...")
    model_a = ResNetWithUserEmbedding(user_dim, NUM_TAGS).to(device)
    if os.path.exists(MODEL_A_PATH):
        model_a.load_state_dict(torch.load(MODEL_A_PATH, map_location=device))
        print("📦 Loaded pre-trained Model A")
    else:
        model_a = train_model_a(train_df, val_df, mlb, NUM_TAGS)

    print("🔁 Loading Model B (IHC_h Predictor)...")
    ihc_model = DualBranchIHCModel(num_tags=NUM_TAGS).to(device)
    # Load numeric scalers
    with open("follower_scaler.pkl", "rb") as f:
        follower_scaler = pickle.load(f)
    with open("hashtag_count_scaler.pkl", "rb") as f:
        hashtag_count_scaler = pickle.load(f)
    with open("ihc_scaler.pkl", "rb") as f:
        ihc_scaler = pickle.load(f)

    if os.path.exists(MODEL_B_PATH):
        ihc_model.load_state_dict(torch.load(MODEL_B_PATH, map_location=device))
        print("📦 Loaded pre-trained Model B")
    else:
        raise FileNotFoundError(f"❌ Model B not found at {MODEL_B_PATH}. Please train it separately and try again.")

    print("🧪 Evaluating final system...")
    evaluate(model_a, ihc_model, val_df, TAGS, NUM_TAGS, follower_scaler, hashtag_count_scaler, ihc_scaler)





if __name__ == "__main__":
    main()

