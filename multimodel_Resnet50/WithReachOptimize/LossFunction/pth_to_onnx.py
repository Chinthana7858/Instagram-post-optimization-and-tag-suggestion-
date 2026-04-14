import torch
from torchvision import models
from multimodel_Resnet50_with_Reach_optimize import ResNetWithUserEmbedding  # Adjust if in same file

# ---------------- Configuration ----------------
# ---------------- Configuration ----------------
MODEL_PATH = r"E:\ProcessedV3\resnet50_Reach_Optimize.pth"
ONNX_EXPORT_PATH = r"E:\ProcessedV3\resnet50_Reach_Optimize.onnx"
NUM_CLASSES = 3058
USER_DIM = 3017
IMG_SIZE = 128

BATCH_SIZE = 1     # For dummy input

# ---------------- Load Model ----------------
model = ResNetWithUserEmbedding(USER_DIM, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# ---------------- Dummy Input ----------------
dummy_img = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
dummy_user = torch.randn(BATCH_SIZE, USER_DIM)

# ---------------- Export to ONNX ----------------
torch.onnx.export(
    model,
    (dummy_img, dummy_user),
    ONNX_EXPORT_PATH,
    input_names=["image", "user_vector"],
    output_names=["hashtag_logits"],
    dynamic_axes={
        "image": {0: "batch_size"},
        "user_vector": {0: "batch_size"},
        "hashtag_logits": {0: "batch_size"}
    },
    export_params=True,
    opset_version=11
)

print(f"✅ ONNX model saved at {ONNX_EXPORT_PATH}")
