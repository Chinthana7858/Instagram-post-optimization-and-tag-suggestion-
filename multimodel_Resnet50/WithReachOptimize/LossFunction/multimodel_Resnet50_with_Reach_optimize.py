import itertools
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
from sklearn.metrics import accuracy_score, hamming_loss, precision_score, recall_score, f1_score, average_precision_score
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# ------------------------------ Configuration ------------------------------
IMG_FOLDER = r"E:\ProcessedV3\selected_images"
POST_CSV = r"E:\ProcessedV3\6-images_tags_reach.csv"
USER_CSV = r"E:\ProcessedV3\User_Hashtag_Frequency_Matrix.csv"
MLB_PATH = r"E:\ProcessedV3\mlb.pkl"
MODEL_PATH = r"E:\ProcessedV3\resnet50_Reach_Optimize.pth"
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 1
MIN_TAG_COUNT = 100
THRESHOLD = 0.4
BALANCE = 0.5  # 0 = only user/image preference, 1 = only reach (IHC_h)

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
        ihc_score = torch.tensor(row["IHC_h"], dtype=torch.float32)
        return image, user_vector, labels, ihc_score


# ------------------------------ CooccurrenceMatrix ------------------------------
class CooccurrenceHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cooc_matrix = nn.Parameter(torch.eye(num_classes))  # initialize with identity

    def forward(self, logits):
        return logits @ self.cooc_matrix  # apply co-occurrence adjustment
# ------------------------------ Model ------------------------------
class ResNetWithUserEmbedding(nn.Module):
    def __init__(self, user_out, num_classes, user_weight=1.0):
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

        self.cooccurrence_head = CooccurrenceHead(num_classes)
        self.ihc_head = nn.Sequential(
            nn.Linear(num_classes, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.user_weight = user_weight

    def forward(self, img, user_feat):
        img_feat = self.resnet_backbone(img)
        user_feat = self.user_proj(user_feat)
        user_feat = self.user_weight * user_feat
        x = torch.cat([img_feat, user_feat], dim=1)
        logits = self.classifier(x)
        return self.cooccurrence_head(logits)

    def predict_ihc(self, tag_vec):
        return self.ihc_head(tag_vec)

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

# ------------------------------ Evaluation ------------------------------
def calculate_metrics(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs >= threshold).astype(int)
    print("\n📌 Final Evaluation Metrics")
    print(f"🔹 Subset Accuracy     : {accuracy_score(y_true, y_pred):.4f} (Exact match ratio)")
    print(f"🔹 Hamming Loss        : {hamming_loss(y_true, y_pred):.4f}")
    print(f"🔹 Micro Precision     : {precision_score(y_true, y_pred, average='micro', zero_division=0):.4f}")
    print(f"🔹 Micro Recall        : {recall_score(y_true, y_pred, average='micro', zero_division=0):.4f}")
    print(f"🔹 Micro F1 Score      : {f1_score(y_true, y_pred, average='micro', zero_division=0):.4f}")
    print(f"🔹 Macro F1 Score      : {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"🔹 mAP                 : {average_precision_score(y_true, y_probs, average='macro'):.4f}")

# ------------------------------ Training ------------------------------
def train_model(model, train_loader, optimizer, criterion, device, scaler, epoch):
    model.train()
    total_loss = 0
    for images, user_vecs, labels, ihc_scores in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images, user_vecs, labels = images.to(device), user_vecs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images, user_vecs)
            ihc_preds = model.predict_ihc(torch.sigmoid(outputs))  # use probs not logits
            tag_loss = criterion(outputs, labels)
            ihc_loss = nn.MSELoss()(ihc_preds.squeeze(), ihc_scores.to(device))
            loss = tag_loss + 0.3 * ihc_loss  # Adjust 0.3 weight as needed

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
    # Change waight
    model = ResNetWithUserEmbedding(user_dim, len(mlb.classes_), user_weight=1.0).to(device)

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

    if mode == "test":
        print("🧪 Evaluating model...")
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()

        val_ds = MultimodalHashtagDataset(val_df, mlb, user_matrix_df, IMG_FOLDER, IMG_SIZE, transform)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

        y_true, y_probs = [], []
        with torch.no_grad():
            for images, user_vecs, labels in tqdm(val_loader):
                images, user_vecs, labels = images.to(device), user_vecs.to(device), labels.to(device)
                logits = model(images, user_vecs)
                probs = torch.sigmoid(logits)

                topk_vals, topk_indices = probs.topk(k=10, dim=1)
                best_score = -float("inf")
                best_vec = torch.zeros_like(probs)

                for k in range(1, 4):
                    for combo in itertools.combinations(topk_indices[0].tolist(), k):
                        candidate = torch.zeros_like(probs[0])
                        candidate[list(combo)] = 1
                        ihc_pred = model.predict_ihc(candidate.unsqueeze(0).to(device)).item()

                        blend_score = (1 - BALANCE) * probs[0][list(combo)].mean().item() + BALANCE * ihc_pred

                        if blend_score > best_score:
                            best_score = blend_score
                            best_vec = candidate

                y_probs.append(best_vec.cpu())
                y_true.append(labels.cpu())

        y_true = torch.cat(y_true).numpy()
        y_probs = torch.stack(y_probs).numpy()
        calculate_metrics(y_true, y_probs, threshold=THRESHOLD)


# ------------------------------ CLI Entry ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train", help="Mode: train or test")
    args = parser.parse_args()
    main(args.mode)
#
# if __name__ == "__main__":
#     main("test")