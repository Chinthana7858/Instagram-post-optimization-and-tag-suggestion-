import os
import gc
import argparse
import pickle

import timm
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, hamming_loss, precision_score, recall_score, f1_score, average_precision_score
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# ------------------------------ Configuration ------------------------------
IMG_FOLDER = r"E:\FinalData\selected_images"
POST_CSV = r"E:\FinalData\6-images_tags_reach.csv"
USER_CSV = r"E:\FinalData\User_Hashtag_Frequency_Matrix.csv"
MLB_PATH = r"E:\FinalData\mlb.pkl"
MODEL_PATH = r"E:\FinalData\Models\vit_base_patch16_224_userbiased_model.pth"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
MIN_TAG_COUNT = 100
THRESHOLD = 0.5

# ------------------------------ Dataset ------------------------------
class MultimodalHashtagDataset(Dataset):
    def __init__(self, df, mlb, user_matrix_df, img_folder, img_size, transform=None):
        self.df = df.reset_index(drop=True)
        self.mlb = mlb
        self.user_matrix_df = user_matrix_df.set_index("Username")
        self.img_folder = img_folder
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        username = row["Username"]
        image_path = os.path.join(self.img_folder, row["Image_file_name"])
        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except:
            image = torch.zeros((3, self.img_size, self.img_size))

        user_vector = torch.tensor(self.user_matrix_df.loc[username].values, dtype=torch.float32)
        labels = torch.tensor(self.mlb.transform([row["Hashtags"]])[0], dtype=torch.float32)
        return image, user_vector, labels

# ------------------------------ Model ------------------------------
class ViTWithUserEmbedding(nn.Module):
    def __init__(self, user_out, num_classes, vit_model='vit_base_patch16_224'):
        super().__init__()
        self.vit = timm.create_model(vit_model, pretrained=True)
        vit_out = self.vit.head.in_features
        self.vit.reset_classifier(0)  # Remove classification layer

        self.user_proj = nn.Sequential(
            nn.Linear(user_out, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(vit_out + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, img, user_feat):
        img_feat = self.vit(img)
        user_feat = self.user_proj(user_feat)
        x = torch.cat([img_feat, user_feat], dim=1)
        return self.classifier(x)

# ------------------------------ Data Prep ------------------------------
def prepare_data(post_csv, user_matrix_csv, min_count=100, mlb_path=None):
    df = pd.read_csv(post_csv)
    df["Hashtags"] = df["Hashtags"].apply(lambda x: str(x).split(", "))
    user_matrix_df = pd.read_csv(user_matrix_csv)

    # Filter infrequent tags
    all_tags = [tag for tags in df["Hashtags"] for tag in tags]
    selected_tags = pd.Series(all_tags).value_counts()
    selected_tags = selected_tags[selected_tags >= min_count].index.tolist()
    df["Hashtags"] = df["Hashtags"].apply(lambda tags: [tag for tag in tags if tag in selected_tags])
    df = df[df["Hashtags"].map(len) > 0]

    mlb = MultiLabelBinarizer()
    mlb.fit(df["Hashtags"])

    if mlb_path:
        with open(mlb_path, "wb") as f:
            pickle.dump(mlb, f)

    return df, user_matrix_df, mlb

def get_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# ------------------------------ Evaluation ------------------------------
def calculate_metrics(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs >= threshold).astype(int)
    metrics = {
        "mAP": average_precision_score(y_true, y_probs, average='macro'),
        "Macro F1 Score": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "Micro F1 Score": f1_score(y_true, y_pred, average='micro', zero_division=0),
        "Micro Recall": recall_score(y_true, y_pred, average='micro', zero_division=0),
        "Micro Precision": precision_score(y_true, y_pred, average='micro', zero_division=0),
        "Subset Accuracy": accuracy_score(y_true, y_pred)
    }

    print("\n📌 Final Evaluation Metrics")
    for k, v in metrics.items():
        print(f"{k:<20}: {v:.4f}")

    return metrics

def plot_eval_bar(metrics):
    plt.figure(figsize=(10, 6))
    keys, values = list(metrics.keys()), list(metrics.values())
    bars = plt.barh(keys, values, color='skyblue')
    plt.xlim(0, 1.0)
    plt.title('Evaluation Metrics')
    for bar in bars:
        plt.text(bar.get_width() + 0.01, bar.get_y() + 0.3, f"{bar.get_width():.4f}")
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.show()


# ------------------------------ Training ------------------------------
def train_model(model, train_loader, optimizer, criterion, device, scaler, epoch):
    model.train()
    total_loss = 0
    for images, user_vecs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images, user_vecs, labels = images.to(device), user_vecs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images, user_vecs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# ------------------------------ Main ------------------------------
def main(mode):
    gc.collect()
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device: {device}")

    # Load data
    df, user_matrix_df, mlb = prepare_data(POST_CSV, USER_CSV, min_count=MIN_TAG_COUNT, mlb_path=MLB_PATH)
    transform = get_transforms(IMG_SIZE)
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)

    user_dim = user_matrix_df.shape[1] - 1 if "Username" in user_matrix_df.columns else user_matrix_df.shape[1]
    model = ViTWithUserEmbedding(user_dim, len(mlb.classes_)).to(device)



    if mode == "train":
        print("🚀 Training mode...")
        train_ds = MultimodalHashtagDataset(train_df, mlb, user_matrix_df, IMG_FOLDER, IMG_SIZE, transform)
        val_ds = MultimodalHashtagDataset(val_df, mlb, user_matrix_df, IMG_FOLDER, IMG_SIZE, transform)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        scaler = torch.cuda.amp.GradScaler()

        best_loss = float("inf")
        for epoch in range(EPOCHS):
            loss = train_model(model, train_loader, optimizer, criterion, device, scaler, epoch)
            print(f"📉 Epoch {epoch+1} Loss: {loss:.4f}")
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), MODEL_PATH)
                print("✅ Model saved!")

    print("🧪 Evaluating model...")
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    val_ds = MultimodalHashtagDataset(val_df, mlb, user_matrix_df, IMG_FOLDER, IMG_SIZE, transform)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    y_true, y_probs = [], []
    with torch.no_grad():
        for images, user_vecs, labels in val_loader:
            images, user_vecs, labels = images.to(device), user_vecs.to(device), labels.to(device)
            outputs = torch.sigmoid(model(images, user_vecs))
            y_probs.append(outputs.cpu())
            y_true.append(labels.cpu())

    y_true = torch.cat(y_true).numpy()
    y_probs = torch.cat(y_probs).numpy()
    metrics = calculate_metrics(y_true, y_probs, threshold=THRESHOLD)
    plot_eval_bar(metrics)


# ------------------------------ CLI Entry ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train", help="Mode: train or test")
    args = parser.parse_args()
    main(args.mode)

