import torch
import torch.onnx
import pandas as pd
import numpy as np
from torchvision import transforms
from PIL import Image
import pickle
from multimodel_Resnet50 import ResNetWithUserEmbedding


# ------------------ Load model configuration ------------------
MODEL_PATH = r"E:\ProcessedV3\Models\resnet50_userbiased_model.pth"
USER_CSV = r"E:\ProcessedV3\User_Hashtag_Frequency_Matrix.csv"
MLB_PATH = r"E:\ProcessedV3\mlb.pkl"
ONNX_EXPORT_PATH = r"E:\ProcessedV3\Models\resnet50_userbiased_model.onnx"
IMG_SIZE = 128

# ------------------ Load label binarizer ------------------
with open(MLB_PATH, "rb") as f:
    mlb = pickle.load(f)

# ------------------ Load user matrix ------------------
user_matrix_df = pd.read_csv(USER_CSV)
user_matrix_df = user_matrix_df.set_index("Username")
user_dim = user_matrix_df.shape[1]

# ------------------ Create dummy input ------------------
dummy_img = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)  # batch size = 1
dummy_user_vec = torch.randn(1, user_dim)

# ------------------ Load model ------------------
model = ResNetWithUserEmbedding(user_out=user_dim, num_classes=len(mlb.classes_))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ------------------ Export to ONNX ------------------
torch.onnx.export(
    model,
    (dummy_img, dummy_user_vec),                         # inputs (tuple)
    ONNX_EXPORT_PATH,                                    # output path
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["image", "user_vector"],
    output_names=["hashtag_logits"],
    dynamic_axes={
        "image": {0: "batch_size"},
        "user_vector": {0: "batch_size"},
        "hashtag_logits": {0: "batch_size"}
    }
)

print(f"✅ Model successfully exported to {ONNX_EXPORT_PATH}")
