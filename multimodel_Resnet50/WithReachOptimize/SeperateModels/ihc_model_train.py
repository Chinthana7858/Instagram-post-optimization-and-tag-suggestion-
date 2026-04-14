import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
POST_CSV = r"E:\ProcessedV3\6-images_tags_reach.csv"
MLB_PATH = r"E:\ProcessedV3\mlb.pkl"
IHC_MODEL_PATH = r"E:\ProcessedV3\ihc_predictor.pth"
MIN_TAG_COUNT = 100
BATCH_SIZE = 32
EPOCHS = 7
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
        tag_vector = torch.tensor(self.mlb.transform([row["Hashtags"]])[0], dtype=torch.float32)
        ihc_score = torch.tensor(row["IHC_h"], dtype=torch.float32)
        return tag_vector, ihc_score


# ---------------- MODEL ----------------
class IHCScorePredictor(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_tags, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


# ---------------- DATA PREP ----------------
def prepare_data(csv_path, min_count=100, mlb_path=None):
    df = pd.read_csv(csv_path)
    df["Hashtags"] = df["Hashtags"].apply(lambda x: str(x).split(", "))

    all_tags = [tag for tags in df["Hashtags"] for tag in tags]
    tag_counts = pd.Series(all_tags).value_counts()
    selected_tags = tag_counts[tag_counts >= min_count].index.tolist()
    df["Hashtags"] = df["Hashtags"].apply(lambda tags: [t for t in tags if t in selected_tags])
    df = df[df["Hashtags"].map(len) > 0]

    mlb = MultiLabelBinarizer()
    mlb.fit(df["Hashtags"])

    if mlb_path:
        with open(mlb_path, "wb") as f:
            pickle.dump(mlb, f)

    return df, mlb


# ---------------- TRAIN ----------------
def train_model(model, train_loader, val_loader, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)  # L2 regularization
    loss_fn = nn.MSELoss()
    model.to(DEVICE)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for tag_vec, ihc_score in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            tag_vec = tag_vec.to(DEVICE)
            ihc_score = ihc_score.to(DEVICE)

            pred = model(tag_vec).squeeze()
            loss = loss_fn(pred, ihc_score)

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
            for tag_vec, ihc_score in val_loader:
                tag_vec = tag_vec.to(DEVICE)
                ihc_score = ihc_score.to(DEVICE)
                pred = model(tag_vec).squeeze()
                val_loss = loss_fn(pred, ihc_score)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1} → Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # 📊 Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), train_losses, marker='o', label="Train Loss", color='blue')
    plt.plot(range(1, epochs+1), val_losses, marker='s', label="Val Loss", color='orange')
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ihc_loss_curve.png")
    print("✅ Loss plot saved as ihc_loss_curve.png")


# ---------------- EVALUATE ----------------
def evaluate_model(model, dataloader):
    model.eval()
    true_vals, pred_vals = [], []

    with torch.no_grad():
        for tag_vec, ihc_score in dataloader:
            tag_vec = tag_vec.to(DEVICE)
            preds = model(tag_vec).squeeze().cpu()
            ihc_score = ihc_score.cpu()

            true_vals.extend(ihc_score.tolist())
            pred_vals.extend(preds.tolist())

    true_vals = torch.tensor(true_vals)
    pred_vals = torch.tensor(pred_vals)
    rmse = torch.sqrt(torch.mean((true_vals - pred_vals) ** 2)).item()
    mae = torch.mean(torch.abs(true_vals - pred_vals)).item()
    r2 = r2_score(true_vals.numpy(), pred_vals.numpy())  # ← R² score calculation

    print(f"\n📊 Final Evaluation on Validation Set:")
    print(f"   RMSE  = {rmse:.4f}")
    print(f"   MAE   = {mae:.4f}")
    print(f"   R²    = {r2:.4f}")  # ← Print R² score


# ---------------- MAIN ----------------
if __name__ == "__main__":
    df, mlb = prepare_data(POST_CSV, MIN_TAG_COUNT, MLB_PATH)
    dataset = HashtagIHCScoreDataset(df, mlb)

    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, drop_last=True)

    model = IHCScorePredictor(num_tags=len(mlb.classes_))
    train_model(model, train_loader, val_loader, EPOCHS)
    torch.save(model.state_dict(), IHC_MODEL_PATH)
    print(f"✅ IHC model saved to {IHC_MODEL_PATH}")

    model.load_state_dict(torch.load(IHC_MODEL_PATH))
    evaluate_model(model, val_loader)
