# ultralytics version 
!pip install ultralytics -q

# IMPORTS

import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics import YOLO
from pathlib import Path

# LOAD YOLO11n MODEL
model = YOLO("yolo11n.pt")
print("YOLO11n model loaded successfully.")

# DATASET YAML CREATION
dataset_base = r"D:/drone_dataset"
class_names = ["drone"]
# YOLO data configuration
data = {
    "train": f"{dataset_base}/images/train",
    "val": f"{dataset_base}/images/val",
    "nc": len(class_names),
    "names": class_names
}
# Local YAML save path
yaml_path = "data.yaml"
# Create YAML file
with open(yaml_path, "w") as f:
    yaml.dump(data, f, default_flow_style=False)
print(" data.yaml created successfully.")
# COORDINATE ATTENTION MODULE ONLY
class CoordAttention(nn.Module):
    def __init__(self, in_ch, reduction=32):
        super().__init__()
        mip = max(8, in_ch // reduction)
        self.conv1 = nn.Conv2d(in_ch, mip, 1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, in_ch, 1)
        self.conv_w = nn.Conv2d(mip, in_ch, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        B, C, H, W = x.size()
        fh = F.avg_pool2d(x, (1, W))
        fw = F.avg_pool2d(x, (H, 1))
        fh = self.act(self.bn1(self.conv1(fh)))
        fw = self.act(self.bn1(self.conv1(fw)))
        ah = self.sigmoid(self.conv_h(fh))
        aw = self.sigmoid(self.conv_w(fw))
        return x * ah * aw
# FIND LAYERS FOR ATTENTION INJECTION
def find_candidates(module):
    for name, child in module.named_children():
        cls = child.__class__.__name__.lower()
        # Candidate layers
        if (
            "c3" in cls
            or "bottleneck" in cls
            or "conv" in cls
            or "c2f" in cls
        ):
            yield module, name, child
        # Recursive search
        for item in find_candidates(child):
            yield item
# ATTENTION INJECTION FUNCTION
def inject_attention(model_module, factory, max_inject=6):
    candidates = list(find_candidates(model_module))
    if not candidates:
        print(" No candidates found.")
        return
    total = len(candidates)
  # Select evenly spaced layers
    indices = (
        list(range(total))
        if total <= max_inject
        else [int(i * total / max_inject) for i in range(max_inject)]
    )
    injected = 0
    for idx in indices:

        parent, name, child = candidates[idx]
        out_ch = None
        # Find output channels
        for m in child.modules():
            if isinstance(m, nn.Conv2d):
                out_ch = m.out_channels
                break
        if not out_ch:
            continue
        # Inject Coordinate Attention
        setattr(
            parent,
            name,
            nn.Sequential(
                child,
                factory(out_ch)
            )
        )
        injected += 1
        print(f" CoordAttention injected after `{name}` | channels={out_ch}")

    print(f"\nTotal injected: {injected}/{len(indices)}")

# INJECT COORDINATE ATTENTION
print("\n Injecting Coordinate Attention...\n")
inject_attention(
    model.model,
    lambda ch: CoordAttention(ch),
    max_inject=6
)

# SAVE MODIFIED MODEL
# Local save folder
out_dir = Path("runs/attn_experiments")
out_dir.mkdir(parents=True, exist_ok=True)
# Save modified weights
torch.save(
    model.model.state_dict(),
    out_dir / "yolo11n_ca_state.pt"
)
print("Modified model weights saved.")
# TRAINING CONFIGURATION
epochs = 50
batch = 16
imgsz = 640
# Disable WANDB
os.environ["WANDB_MODE"] = "disabled"
os.environ["TENSORBOARD"] = "False"
# OPTIONAL CALLBACKS
class SimpleCB:
    # Print training loss
    def on_train_epoch_end(self, trainer):
        try:
            print(
                f"Epoch {trainer.epoch + 1}/{trainer.epochs} "
                f"| Loss: {trainer.loss_items}"
            )
        except:
            pass
    # Print validation metrics
    def on_val_end(self, v):
        try:
            print(
                f"VAL mAP50: {v.metrics.box.map50:.4f} "
                f"| mAP: {v.metrics.box.map:.4f}"
            )
        except:
            pass
# Register callbacks
model.add_callback(
    "on_train_epoch_end",
    SimpleCB().on_train_epoch_end
)
model.add_callback(
    "on_val_end",
    SimpleCB().on_val_end
)
# TRAIN MODEL
results = model.train(
    data=yaml_path,
    epochs=epochs,
    imgsz=imgsz,
    batch=batch,
    workers=0,
    device=0,
    name="drone_detector_ca",
    mosaic=0.0,
    mixup=0.0,
    translate=0.0,
    scale=0.0,
    degrees=0.0,
    shear=0.0,
    perspective=0.0,
    fliplr=0.5,
    amp=False
)

print("\n Training Finished!")
