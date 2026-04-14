import torch
import pickle
from updated_reach_predicter import DualBranchModel  # Ensure your model class is imported

# ---------------- CONFIG ----------------
MODEL_PATH = r"E:\ProcessedV3\ihc_dualbranch_model.pth"
MLB_PATH = r"E:\ProcessedV3\mlb.pkl"
ONNX_PATH = r"E:\ProcessedV3\ihc_dualbranch_model.onnx"

# ---------------- LOAD MODEL ----------------
# Load MultiLabelBinarizer to get num_tags
with open(MLB_PATH, "rb") as f:
    mlb = pickle.load(f)
num_tags = len(mlb.classes_)

# Create model instance and load weights
model = DualBranchModel(num_tags=num_tags)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# ---------------- PREPARE DUMMY INPUT ----------------
dummy_tags = torch.randn(1, num_tags)  # Example: 1 batch, num_tags input
dummy_numeric = torch.randn(1, 2)     # Example: 1 batch, 2 numeric features (#Followers, HashtagCount)

# ---------------- EXPORT TO ONNX ----------------
torch.onnx.export(
    model,
    (dummy_tags, dummy_numeric),
    ONNX_PATH,
    input_names=["tags", "numeric"],
    output_names=["ihc_score"],
    dynamic_axes={
        "tags": {0: "batch_size"},
        "numeric": {0: "batch_size"},
        "ihc_score": {0: "batch_size"}
    },
    opset_version=17
)

print(f"✅ Model successfully converted to ONNX and saved at {ONNX_PATH}")
