import os
import itertools

import numpy as np
import torch
import pickle
import pandas as pd
from PIL import Image
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
MODEL_B_PATH = r"E:\ProcessedV3\MultiModels\ihc_predictor.pth"
OUTPUT_CSV = r"E:\ProcessedV3\evaluation_output.csv"
TOP_K = 6
MAX_COMBO = 3
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
    df["Hashtags"] = df["Hashtags"].apply(lambda x: str(x).split(", "))

    all_tags = [tag for tags in df["Hashtags"] for tag in tags]
    tag_counts = pd.Series(all_tags).value_counts()
    selected_tags = tag_counts[tag_counts >= min_count].index.tolist()
    df["Hashtags"] = df["Hashtags"].apply(lambda tags: [tag for tag in tags if tag in selected_tags])
    df = df[df["Hashtags"].map(len) > 0]

    mlb = MultiLabelBinarizer()
    mlb.fit(df["Hashtags"])
    if mlb_path:
        with open(mlb_path, "wb") as f:
            pickle.dump(mlb, f)

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


def train_model_a(train_df, mlb, num_tags):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = ImageUserDataset(train_df, user_matrix_df, IMG_PATH, transform, mlb)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = ResNetWithUserEmbedding(user_dim, num_tags).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for img, user_vec, labels in train_loader:
            img, user_vec, labels = img.to(device), user_vec.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(img, user_vec)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Model A Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), MODEL_A_PATH)
    print(f"✅ Hashtag prediction model saved to {MODEL_A_PATH}")
    return model


# ---------------------- Model B (IHC Predictor) ----------------------
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
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


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

# ---------------------- Train IHC Model ----------------------
def train_ihc_model(train_df, val_df, mlb, scaler, num_tags):

    train_ds = HashtagIhcDataset(train_df, mlb, scaler)
    val_ds = HashtagIhcDataset(val_df, mlb, scaler)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = IHCScorePredictor(num_tags).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    early_stop_counter = 0
    patience = 5

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        for tag_vec, ihc_score in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            tag_vec, ihc_score = tag_vec.to(DEVICE), ihc_score.to(DEVICE)
            pred = model(tag_vec).squeeze()
            loss = loss_fn(pred, ihc_score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for tag_vec, ihc_score in val_loader:
                tag_vec, ihc_score = tag_vec.to(DEVICE), ihc_score.to(DEVICE)
                pred = model(tag_vec).squeeze()
                val_loss = loss_fn(pred, ihc_score)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), MODEL_B_PATH)
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("⛔ Early stopping triggered.")
                break

    print(f"✅ IHC model saved to {MODEL_B_PATH}")
    return model

# ---------------------- Inference + Evaluation ----------------------
def evaluate(model_a, model_b, val_df, TAGS, num_tags):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_a.eval()
    model_b.eval()
    results = []

    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="🔍 Evaluating"):
        username = row["Username"]
        image_file = row["Image_file_name"]
        prev_tags = row["Hashtags"]
        prev_ihc = row["IHC_h"]

        try:
            img = Image.open(os.path.join(IMG_PATH, image_file)).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
        except:
            continue

        if username not in user_matrix_df.index:
            continue
        user_vector = torch.tensor(user_matrix_df.loc[username].values, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.sigmoid(model_a(img_tensor, user_vector))[0]
            topk_indices = torch.topk(probs, TOP_K).indices.cpu().tolist()
            best_score = -float("inf")
            best_tags = []
            for r in range(1, MAX_COMBO + 1):
                for combo in itertools.combinations(topk_indices, r):
                    tag_vec = torch.zeros(num_tags).to(device)
                    tag_vec[list(combo)] = 1
                    pred_ihc = model_b(tag_vec.unsqueeze(0)).item()
                    if pred_ihc > best_score:
                        best_score = pred_ihc
                        best_tags = combo

        pred_tags = [TAGS[i] for i in best_tags]
        results.append({
            "Image": image_file,
            "Username": username,
            "Previous_Hashtags": ", ".join(prev_tags),
            "Predicted_Hashtags": ", ".join(pred_tags),
            "Previous_IHC_h": prev_ihc,
            "Predicted_IHC_h": best_score
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_CSV, index=False)

    y_true = result_df["Previous_IHC_h"].astype(float).values
    y_pred = result_df["Predicted_IHC_h"].astype(float).values
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("\n📊 Evaluation Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R²  : {r2:.4f}")
    print(f"✅ Evaluation saved to {OUTPUT_CSV}")

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
        model_a = train_model_a(train_df, mlb, NUM_TAGS)

    print("🔁 Loading or Training Model B (IHC_h Predictor)...")
    ihc_model = IHCScorePredictor(NUM_TAGS).to(device)
    if os.path.exists(MODEL_B_PATH):
        ihc_model.load_state_dict(torch.load(MODEL_B_PATH, map_location=device))
        print("📦 Loaded pre-trained Model B")
    else:
        ihc_model = train_ihc_model(train_df, val_df, mlb, scaler,NUM_TAGS).to(device)

    print("🧪 Evaluating final system...")
    evaluate(model_a, ihc_model, val_df, TAGS, NUM_TAGS)



if __name__ == "__main__":
    main()

