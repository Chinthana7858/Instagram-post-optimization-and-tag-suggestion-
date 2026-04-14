import os
import itertools
import torch
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.nn.functional import sigmoid

# ================= CONFIG =================
IMG_SIZE = 128
TOP_K = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
MODEL_A_PATH = r"E:\FinalData\MultiModels\resnet50_multimodel.pth"
MODEL_B_PATH = r"E:\FinalData\MultiModels\ihc_dualbranch_model.pth"
MLB_PATH = r"E:\FinalData\mlb.pkl"
FOLLOWER_SCALER_PATH = r"E:\FinalData\pkls\follower_scaler.pkl"
HASHTAG_SCALER_PATH = r"E:\FinalData\pkls\hashtag_count_scaler.pkl"
IHC_SCALER_PATH = r"E:\FinalData\pkls\ihc_scaler.pkl"
USER_FREQ_PATH = r"E:\FinalData\User_Hashtag_Frequency_Matrix.csv"
USER_META_PATH = r"E:\FinalData\6-images_tags_reach.csv"

# Load user data
user_matrix_df = pd.read_csv(USER_FREQ_PATH).set_index("Username")
user_meta_df = pd.read_csv(USER_META_PATH).set_index("Username")  # Has followers, etc.

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ================= MODELS =================
class CooccurrenceHead(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cooc_matrix = torch.nn.Parameter(torch.eye(num_classes))

    def forward(self, logits):
        return logits @ self.cooc_matrix

class ResNetWithUserEmbedding(torch.nn.Module):
    def __init__(self, user_out, num_classes, user_weight=1.0):
        super().__init__()
        from torchvision import models
        base = models.resnet50(weights=None)
        base.fc = torch.nn.Identity()
        self.resnet = base
        self.user_proj = torch.nn.Sequential(
            torch.nn.Linear(user_out, 256),
            torch.nn.ReLU(), torch.nn.Dropout(0.3)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(2048 + 256, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(512, num_classes)
        )
        self.cooc = CooccurrenceHead(num_classes)
        self.user_weight = user_weight

    def forward(self, img, user_feat):
        img_feat = self.resnet(img)
        user_feat = self.user_proj(user_feat) * self.user_weight
        x = torch.cat([img_feat, user_feat], dim=1)
        return self.cooc(self.classifier(x))

class DualBranchIHCModel(torch.nn.Module):
    def __init__(self, num_tags, num_numeric=2):
        super().__init__()
        self.hashtag_branch = torch.nn.Sequential(
            torch.nn.Linear(num_tags, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU()
        )
        self.numeric_branch = torch.nn.Sequential(
            torch.nn.Linear(num_numeric, 32),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32)
        )
        self.combined = torch.nn.Sequential(
            torch.nn.Linear(128 + 32, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1)
        )

    def forward(self, tags, numeric):
        tag_out = self.hashtag_branch(tags)
        num_out = self.numeric_branch(numeric)
        combined = torch.cat([tag_out, num_out], dim=1)
        return self.combined(combined)

# ================= LOAD RESOURCES =================
print("Loading resources...")
with open(MLB_PATH, "rb") as f:
    mlb = pickle.load(f)
with open(FOLLOWER_SCALER_PATH, "rb") as f:
    follower_scaler = pickle.load(f)
with open(HASHTAG_SCALER_PATH, "rb") as f:
    hashtag_count_scaler = pickle.load(f)
with open(IHC_SCALER_PATH, "rb") as f:
    ihc_scaler = pickle.load(f)

TAGS = mlb.classes_
NUM_TAGS = len(TAGS)
user_dim = user_matrix_df.shape[1]

# Load models
model_a = ResNetWithUserEmbedding(user_dim, NUM_TAGS).to(DEVICE)
model_a.load_state_dict(torch.load(MODEL_A_PATH, map_location=DEVICE))
model_a.eval()

model_b = DualBranchIHCModel(num_tags=NUM_TAGS).to(DEVICE)
model_b.load_state_dict(torch.load(MODEL_B_PATH, map_location=DEVICE))
model_b.eval()

# ================= PREDICT FUNCTION =================
def predict(username, image_path):
    # User vector
    if username in user_matrix_df.index:
        user_vec = torch.tensor(user_matrix_df.loc[username].values, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    else:
        user_vec = torch.zeros((1, user_matrix_df.shape[1]), dtype=torch.float32).to(DEVICE)

    # Followers
    if username in user_meta_df.index:
        followers = user_meta_df.loc[username, "#Followers"]
        if isinstance(followers, pd.Series):
            followers = followers.iloc[0]  # Or followers.mean()
    else:
        followers = 0

    # Image
    img = Image.open("E:\\FinalData\\selected_images\\" + image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # Predict hashtags
    with torch.no_grad():
        probs = sigmoid(model_a(img_tensor, user_vec))[0].cpu().numpy()

    topk_indices = np.argsort(probs)[-TOP_K:][::-1]
    predicted_hashtags = [TAGS[i] for i in topk_indices]

    # Generate all subsets of top-k
    all_combos = []
    for r in range(1, len(topk_indices) + 1):
        all_combos.extend(itertools.combinations(topk_indices, r))

    combo_tensors = torch.zeros((len(all_combos), NUM_TAGS), device=DEVICE)
    for i, combo in enumerate(all_combos):
        combo_tensors[i, list(combo)] = 1

    # Numeric features
    followers_scaled = follower_scaler.transform(
        pd.DataFrame([[np.log1p(followers)]], columns=["#Followers"])
    )[0][0]

    numeric_features = []
    for combo in all_combos:
        hashtag_count = np.log1p(len(combo))
        hashtag_count_scaled = hashtag_count_scaler.transform(
            pd.DataFrame([[hashtag_count]], columns=["HashtagCount"])
        )[0][0]
        numeric_features.append([followers_scaled, hashtag_count_scaled])

    numeric_tensor = torch.tensor(numeric_features, dtype=torch.float32, device=DEVICE)

    # Predict IHC scores
    with torch.no_grad():
        ihc_scores = model_b(combo_tensors, numeric_tensor).squeeze().cpu().numpy()
    best_idx = np.argmax(ihc_scores)
    best_score = ihc_scaler.inverse_transform([[ihc_scores[best_idx]]])[0][0]
    optimized_hashtags = [TAGS[i] for i in all_combos[best_idx]]

    print("\n=== Prediction Result ===")
    print(f"Username: {username}")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Top-{TOP_K} Suggested Hashtags: {predicted_hashtags}")
    print(f"Optimized Hashtags for Reach: {optimized_hashtags}")
    print(f"Predicted IHC_h (Reach): {best_score:.2f}")

# ================= RUN EXAMPLE =================
if __name__ == "__main__":
    username_input = input("Enter username: ")
    image_input = input("Enter image path: ")
    predict(username_input, image_input)
