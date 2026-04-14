import os, gc, pickle, multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, hamming_loss, accuracy_score, average_precision_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# === Configuration ===
IMAGE_FOLDER = r"E:\FinalData\selected_images"
CSV_FILE = r"E:\FinalData\6-images_tags_reach.csv"
MODEL_PATH = r"E:\FinalData\Models\resnet50_image_only_model.pth"
MLB_PATH = r"E:\FinalData\mlb.pkl"
IMG_SIZE = 128
BATCH_SIZE = 256
EPOCHS = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Dataset ===
class HashtagDataset(Dataset):
    def __init__(self, image_paths, hashtags, mlb, image_folder, img_size, transform=None):
        self.image_paths = image_paths
        self.hashtags = hashtags
        self.image_folder = image_folder
        self.img_size = img_size
        self.transform = transform
        self.mlb = mlb

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_paths[idx])
        try:
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
        except:
            print(f"⚠️ Error loading image: {img_path}. Skipping...")
            img = torch.zeros((3, self.img_size, self.img_size))
        label = torch.tensor(self.mlb.transform([self.hashtags[idx]])[0], dtype=torch.float32)
        return img, label


# === Model ===
class HashtagClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# === Training Function ===
def train(model, train_loader, val_loader, criterion, optimizer, scaler, device, epochs, model_path):
    train_losses, val_losses, f1_scores = [], [], []
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"🔄 Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device.type):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        val_loss, f1, _, _, _ = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        f1_scores.append(f1)

        print(f"📊 Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | F1: {f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"✅ Saved best model at epoch {epoch+1}")

    # ✅ Plot training curves
    plot_training_curves(train_losses, val_losses, f1_scores)

    return model, train_losses, val_losses, f1_scores


# === Evaluation Function ===
def evaluate(model, data_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.3).float()

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

    avg_val_loss = val_loss / len(data_loader)
    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_probs = torch.cat(all_probs).numpy()

    f1 = f1_score(y_true, y_pred, average="micro")
    return avg_val_loss, f1, y_true, y_pred, y_probs


# === Visualization: Training Curves ===
def plot_training_curves(train_losses, val_losses, f1_scores):
    plt.figure(figsize=(12, 5))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Val Loss", marker='o')
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    # F1 score curve
    plt.subplot(1, 2, 2)
    plt.plot(f1_scores, label="Validation F1", color='green', marker='o')
    plt.title("Validation F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


# === Visualization: Evaluation Metrics ===
def plot_evaluation_metrics(metrics_dict):
    keys = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    plt.figure(figsize=(10, 6))
    bars = plt.barh(keys, values, color='skyblue')
    plt.xlabel("Score")
    plt.title("📊 Evaluation Metrics (Multilabel Hashtag Prediction)")
    plt.xlim(0, 1.0)

    for bar in bars:
        plt.text(bar.get_width() + 0.01, bar.get_y() + 0.3, f"{bar.get_width():.4f}")

    plt.tight_layout()
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.show()


# === Data Prep ===
def load_data():
    df = pd.read_csv(CSV_FILE)
    df["Hashtags"] = df["Hashtags"].apply(lambda x: x.split(", "))
    image_files = df["Image_file_name"].values
    hashtags = df["Hashtags"].values

    min_occurrences = 100
    hashtag_counts = Counter(tag for tags in hashtags for tag in tags)
    filtered_hashtags = [[tag for tag in tags if hashtag_counts[tag] >= min_occurrences] for tags in hashtags]

    mlb = MultiLabelBinarizer()
    mlb.fit(filtered_hashtags)
    with open(MLB_PATH, "wb") as f:
        pickle.dump(mlb, f)
    num_classes = len(mlb.classes_)

    X_train, X_val, y_train, y_val = train_test_split(image_files, filtered_hashtags, test_size=0.2, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = HashtagDataset(X_train, y_train, mlb, IMAGE_FOLDER, IMG_SIZE, transform)
    val_dataset = HashtagDataset(X_val, y_val, mlb, IMAGE_FOLDER, IMG_SIZE, transform)
    num_workers = min(multiprocessing.cpu_count() // 2, 4)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, num_classes, mlb


# === Testing ===
def test_only():
    gc.collect()
    torch.cuda.empty_cache()
    _, val_loader, num_classes, _ = load_data()
    model = HashtagClassifier(num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    val_loss, _, y_true, y_pred, y_probs = evaluate(model, val_loader, criterion, device)

    micro_f1 = f1_score(y_true, y_pred, average="micro")
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    micro_precision = precision_score(y_true, y_pred, average="micro", zero_division=0)
    micro_recall = recall_score(y_true, y_pred, average="micro", zero_division=0)
    hamming = hamming_loss(y_true, y_pred)
    subset_acc = accuracy_score(y_true, y_pred)
    map_score = average_precision_score(y_true, y_probs, average="micro")

    print("\n📌 Final Evaluation Metrics")
    print(f"Subset Accuracy: {subset_acc:.4f}")
    print(f"Micro Precision: {micro_precision:.4f}")
    print(f"Micro Recall: {micro_recall:.4f}")
    print(f"Micro F1 Score: {micro_f1:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"mAP: {map_score:.4f}")

    metrics = {
        "Subset Accuracy": subset_acc,
        "Micro Precision": micro_precision,
        "Micro Recall": micro_recall,
        "Micro F1 Score": micro_f1,
        "Macro F1 Score": macro_f1,
        "mAP": map_score
    }
    plot_evaluation_metrics(metrics)


# === Main ===
if __name__ == "__main__":
    MODE = "test"  # Change to "test" after training

    if MODE == "train":
        train_loader, val_loader, num_classes, _ = load_data()
        model = HashtagClassifier(num_classes).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scaler = torch.amp.GradScaler()

        model, train_losses, val_losses, f1_scores = train(
            model, train_loader, val_loader,
            criterion, optimizer, scaler,
            device, EPOCHS, MODEL_PATH
        )

    elif MODE == "test":
        test_only()
