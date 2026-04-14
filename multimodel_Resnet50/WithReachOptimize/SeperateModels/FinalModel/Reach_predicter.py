import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
POST_CSV = r"E:\FinalData\6-images_tags_reach.csv"
MLB_PATH = r"E:\FinalData\mlb.pkl"
MODEL_PATH = r"E:\FinalData\Models\ihc_dualbranch_model.pth"
MIN_TAG_COUNT = 100
BATCH_SIZE = 64
EPOCHS = 20
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- DATASET ----------------
class HashtagIHCScoreDataset(Dataset):
    def __init__(self, df, mlb):
        self.df = df.reset_index(drop=True)
        self.mlb = mlb

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Multi-hot vector for hashtags
        tag_vector = torch.tensor(self.mlb.transform([row["Hashtags"]])[0], dtype=torch.float32)

        # Numeric features (already normalized)
        followers = torch.tensor(row["#Followers"], dtype=torch.float32).unsqueeze(0)
        hashtag_count = torch.tensor(row["HashtagCount"], dtype=torch.float32).unsqueeze(0)

        ihc_score = torch.tensor(row["IHC_h"], dtype=torch.float32)

        numeric_features = torch.cat([followers, hashtag_count], dim=0)
        return tag_vector, numeric_features, ihc_score


# ---------------- MODEL ----------------
class DualBranchModel(nn.Module):
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


# ---------------- DATA PREP ----------------
def prepare_data(csv_path, min_count=100, mlb_path=None):
    df = pd.read_csv(csv_path)

    # Split hashtags into list
    df["Hashtags"] = df["Hashtags"].apply(lambda x: str(x).split(", "))

    # Filter hashtags based on min_count
    all_tags = [tag for tags in df["Hashtags"] for tag in tags]
    tag_counts = pd.Series(all_tags).value_counts()
    selected_tags = tag_counts[tag_counts >= min_count].index.tolist()
    df["Hashtags"] = df["Hashtags"].apply(lambda tags: [t for t in tags if t in selected_tags])
    df = df[df["Hashtags"].map(len) > 0]  # Remove rows with empty hashtags

    # Add Hashtag Count feature
    df["HashtagCount"] = df["Hashtags"].apply(len)

    # Apply log transformation for large ranges
    df["#Followers"] = np.log1p(df["#Followers"])
    df["HashtagCount"] = np.log1p(df["HashtagCount"])

    # Encode hashtags
    mlb = MultiLabelBinarizer()
    mlb.fit(df["Hashtags"])
    if mlb_path:
        with open(mlb_path, "wb") as f:
            pickle.dump(mlb, f)

    # Normalize features
    ihc_scaler = StandardScaler()
    df["IHC_h"] = ihc_scaler.fit_transform(df[["IHC_h"]])

    follower_scaler = StandardScaler()
    df["#Followers"] = follower_scaler.fit_transform(df[["#Followers"]])

    hashtag_count_scaler = StandardScaler()
    df["HashtagCount"] = hashtag_count_scaler.fit_transform(df[["HashtagCount"]])

    # Save scalers
    with open("ihc_scaler.pkl", "wb") as f:
        pickle.dump(ihc_scaler, f)
    with open("follower_scaler.pkl", "wb") as f:
        pickle.dump(follower_scaler, f)
    with open("hashtag_count_scaler.pkl", "wb") as f:
        pickle.dump(hashtag_count_scaler, f)

    return df, mlb


# ---------------- TRAIN ----------------
def train_model(model, train_loader, val_loader, epochs=50):
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    loss_fn = nn.MSELoss()
    model.to(DEVICE)

    best_val_loss = float("inf")
    early_stop_counter = 0
    patience = 5

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for tags, numeric, target in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            tags, numeric, target = tags.to(DEVICE), numeric.to(DEVICE), target.to(DEVICE)

            pred = model(tags, numeric).squeeze()
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for tags, numeric, target in val_loader:
                tags, numeric, target = tags.to(DEVICE), numeric.to(DEVICE), target.to(DEVICE)
                pred = model(tags, numeric).squeeze()
                val_loss = loss_fn(pred, target)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break

    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, marker='o', label="Train Loss", color='blue')
    plt.plot(range(1, len(val_losses)+1), val_losses, marker='s', label="Val Loss", color='orange')
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ihc_dualbranch_loss_curve.png")
    print("✅ Loss plot saved as ihc_dualbranch_loss_curve.png")


# ---------------- EVALUATE ----------------
def evaluate_model(model, dataloader):
    model.to(DEVICE)  # Ensure model on DEVICE
    model.eval()
    true_vals, pred_vals = [], []

    with torch.no_grad():
        for tags, numeric, target in dataloader:
            tags, numeric = tags.to(DEVICE), numeric.to(DEVICE)  # Move inputs to GPU/CPU
            preds = model(tags, numeric).squeeze().cpu()  # Move predictions back to CPU
            target = target.cpu()  # For comparison

            true_vals.extend(target.tolist())
            pred_vals.extend(preds.tolist())

    # Inverse transform predictions
    with open("ihc_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    true_vals = scaler.inverse_transform(np.array(true_vals).reshape(-1, 1)).flatten()
    pred_vals = scaler.inverse_transform(np.array(pred_vals).reshape(-1, 1)).flatten()

    rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2))
    mae = np.mean(np.abs(true_vals - pred_vals))
    r2 = r2_score(true_vals, pred_vals)

    print(f"\n📊 Final Evaluation on Validation Set:")
    print(f"   RMSE  = {rmse:.4f}")
    print(f"   MAE   = {mae:.4f}")
    print(f"   R²    = {r2:.4f}")

    plot_evaluation_metrics(rmse, mae, r2)



# ---------------- PLOT METRICES ----------------
def plot_evaluation_metrics(rmse, mae, r2):
    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2
    }

    plt.figure(figsize=(8, 5))
    keys, values = list(metrics.keys()), list(metrics.values())
    bars = plt.barh(keys, values, color='skyblue')
    plt.title("Evaluation Metrics")
    for bar in bars:
        plt.text(bar.get_width() + 0.01, bar.get_y() + 0.3, f"{bar.get_width():.4f}")
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("ihc_dualbranch_eval_metrics.png")
    plt.show()


# ---------------- MAIN ----------------
if __name__ == "__main__":
    df, mlb = prepare_data(POST_CSV, MIN_TAG_COUNT, MLB_PATH)
    dataset = HashtagIHCScoreDataset(df, mlb)

    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, drop_last=True)

    model = DualBranchModel(num_tags=len(mlb.classes_))
    train_model(model, train_loader, val_loader, EPOCHS)

    model.load_state_dict(torch.load(MODEL_PATH))
    print(f"✅ Best model loaded from {MODEL_PATH}")

    evaluate_model(model, val_loader)
