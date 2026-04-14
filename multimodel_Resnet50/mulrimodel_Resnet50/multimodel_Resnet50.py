import os
import gc
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    accuracy_score, hamming_loss, precision_score, recall_score,
    f1_score, average_precision_score
)
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import matplotlib.pyplot as plt
# ------------------------------ Configuration ------------------------------
IMG_FOLDER = r"E:\FinalData\selected_images"
POST_CSV = r"E:\FinalData\6-images_tags_reach.csv"
USER_CSV = r"E:\FinalData\User_Hashtag_Frequency_Matrix.csv"
MLB_PATH = r"E:\FinalData\mlb.pkl"
MODEL_PATH = r"E:\FinalData\Models\resnet50_userbiased_model.pth"
IMG_SIZE = 128
BATCH_SIZE = 128
EPOCHS = 10
MIN_TAG_COUNT = 100
THRESHOLD = 0.4


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
class ResNetWithUserEmbedding(nn.Module):
    def __init__(self, user_out, num_classes):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        resnet.fc = nn.Identity()
        self.resnet_backbone = resnet
        self.user_proj = nn.Sequential(
            nn.Linear(user_out, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, img, user_feat):
        img_feat = self.resnet_backbone(img)
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


def get_transforms(img_size=128):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# ------------------------------ Metrics ------------------------------
def calculate_metrics(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs >= threshold).astype(int)
    metrics = {
        "Subset Accuracy": accuracy_score(y_true, y_pred),
        "Micro Precision": precision_score(y_true, y_pred, average='micro', zero_division=0),
        "Micro Recall": recall_score(y_true, y_pred, average='micro', zero_division=0),
        "Micro F1 Score": f1_score(y_true, y_pred, average='micro', zero_division=0),
        "Macro F1 Score": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "mAP": average_precision_score(y_true, y_probs, average='macro')
    }
    print("\n📌 Final Evaluation Metrics")
    for k, v in metrics.items():
        print(f"{k:<20}: {v:.4f}")
    return metrics


# ------------------------------ Visualization ------------------------------
def plot_training_curves(train_losses, val_losses, val_f1):
    plt.figure(figsize=(12, 5))

    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Val Loss', marker='o')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    # F1 curve
    plt.subplot(1, 2, 2)
    plt.plot(val_f1, label='Val F1 Score', color='green', marker='o')
    plt.title('Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


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
def train_and_validate(model, train_loader, val_loader, optimizer, criterion, device, scaler, epochs):
    train_losses, val_losses, val_f1_scores = [], [], []
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, user_vecs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, user_vecs, labels = images.to(device), user_vecs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images, user_vecs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss, all_labels, all_probs = 0, [], []
        with torch.no_grad():
            for images, user_vecs, labels in val_loader:
                images, user_vecs, labels = images.to(device), user_vecs.to(device), labels.to(device)
                outputs = model(images, user_vecs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                all_labels.append(labels.cpu())
                all_probs.append(torch.sigmoid(outputs).cpu())

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        y_true = torch.cat(all_labels).numpy()
        y_probs = torch.cat(all_probs).numpy()
        val_f1 = f1_score(y_true, (y_probs >= THRESHOLD).astype(int), average='micro', zero_division=0)
        val_f1_scores.append(val_f1)

        print(f"📊 Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val F1={val_f1:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print("✅ Model saved!")

    # Plot curves
    plot_training_curves(train_losses, val_losses, val_f1_scores)


# ------------------------------ Main ------------------------------
def main(mode):
    gc.collect()
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device: {device}")

    df, user_matrix_df, mlb = prepare_data(POST_CSV, USER_CSV, min_count=MIN_TAG_COUNT, mlb_path=MLB_PATH)
    transform = get_transforms(IMG_SIZE)
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)

    user_dim = user_matrix_df.shape[1] - 1 if "Username" in user_matrix_df.columns else user_matrix_df.shape[1]
    model = ResNetWithUserEmbedding(user_dim, len(mlb.classes_)).to(device)

    if mode == "train":
        train_ds = MultimodalHashtagDataset(train_df, mlb, user_matrix_df, IMG_FOLDER, IMG_SIZE, transform)
        val_ds = MultimodalHashtagDataset(val_df, mlb, user_matrix_df, IMG_FOLDER, IMG_SIZE, transform)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        scaler = torch.cuda.amp.GradScaler()

        train_and_validate(model, train_loader, val_loader, optimizer, criterion, device, scaler, EPOCHS)

    # Final Evaluation
    print("🧪 Evaluating...")
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


# ------------------------------ CLI ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="test", help="Mode: train or test")
    args = parser.parse_args()
    main(args.mode)


# def test_different_thresholds(y_true, y_probs, thresholds=None):
#     """
#     Test model performance across different thresholds for multi-label classification.
#
#     Args:
#         y_true (ndarray): Ground truth multi-label matrix.
#         y_probs (ndarray): Predicted probabilities for each class.
#         thresholds (list): List of threshold values to test. Default: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#
#     Returns:
#         results (list): List of tuples (threshold, Micro F1, Precision, Recall, mAP)
#     """
#     from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
#
#     if thresholds is None:
#         thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#
#     results = []
#
#     print("\n📊 Testing Different Thresholds...")
#     print("-" * 70)
#     print(f"{'Threshold':<10}{'Micro F1':<12}{'Precision':<12}{'Recall':<12}{'mAP':<10}")
#     print("-" * 70)
#
#     for t in thresholds:
#         y_pred = (y_probs >= t).astype(int)
#         micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
#         precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
#         recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
#         map_score = average_precision_score(y_true, y_probs, average='macro')
#
#         results.append((t, micro_f1, precision, recall, map_score))
#         print(f"{t:<10.2f}{micro_f1:<12.4f}{precision:<12.4f}{recall:<12.4f}{map_score:<10.4f}")
#
#     # Plot Micro F1, Precision, Recall vs Threshold
#     plt.figure(figsize=(10, 6))
#     plt.plot([r[0] for r in results], [r[1] for r in results], marker='o', label="Micro F1")
#     plt.plot([r[0] for r in results], [r[2] for r in results], marker='s', label="Precision")
#     plt.plot([r[0] for r in results], [r[3] for r in results], marker='^', label="Recall")
#     plt.xlabel("Threshold")
#     plt.ylabel("Score")
#     plt.title("Effect of Threshold on Metrics")
#     plt.legend()
#     plt.grid()
#     plt.show()
#
#     return results
