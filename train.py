# train.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random

# ---------------------------
# Paths & Config
# ---------------------------
DATASET_DIR = r"E:\Projects\Lung Cancer Detection\Dataset"
CLASSES = ["Bengin cases", "Malignant cases", "Normal cases"]

# Force GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_DEVICE = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

# Hyperparams
BATCH_SIZE = 2
LR = 1e-4
EPOCHS = 10
INPUT_H = 64
INPUT_W = 64
DEPTH = 32  # fake depth (stack single slice DEPTH times)
SEED = 42
NUM_WORKERS = 0   # Windows => 0, Linux => >0
PIN_MEMORY = True if DEVICE.type == "cuda" else False

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------
# CNN Model
# ---------------------------
class LungCancer3DCNN(nn.Module):
    def __init__(self, num_classes=3, base_channels=32):
        super(LungCancer3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, base_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(base_channels)
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = nn.Conv3d(base_channels, base_channels*2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(base_channels*2)
        self.pool2 = nn.MaxPool3d(2)

        d = DEPTH // 4
        h = INPUT_H // 4
        w = INPUT_W // 4
        flat_channels = base_channels * 2 * d * h * w

        self.fc1 = nn.Linear(flat_channels, 256)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x

# ---------------------------
# Dataset Loader (JPG + PNG)
# ---------------------------
class LungCancerDataset(Dataset):
    def __init__(self, files, labels, transform=None):
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        label = self.labels[idx]

        # Load JPG/PNG as grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

        # Normalize [0,1]
        minv, maxv = img.min(), img.max()
        if maxv - minv < 1e-6:
            img_norm = np.zeros_like(img, dtype=np.float32)
        else:
            img_norm = (img - minv) / (maxv - minv)

        # Resize
        img_resized = cv2.resize(img_norm, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)

        # Fake depth
        volume = np.stack([img_resized] * DEPTH, axis=0).astype(np.float32)

        # Augmentation
        if self.transform is not None:
            volume = self.transform(volume)

        # Convert to tensor
        tensor_volume = torch.from_numpy(volume).unsqueeze(0).float()  # (1,D,H,W)
        tensor_label = torch.tensor(label, dtype=torch.long)
        return tensor_volume, tensor_label

# ---------------------------
# Gather files
# ---------------------------
def gather_files_labels(dataset_dir, class_names):
    files, labels = [], []
    exts = (".png", ".jpg", ".jpeg")
    for idx, cls in enumerate(class_names):
        cls_folder = os.path.join(dataset_dir, cls)
        if not os.path.exists(cls_folder):
            print(f"⚠️ Warning: folder not found: {cls_folder}")
            continue
        for root, _, fnames in os.walk(cls_folder):
            for fname in fnames:
                if fname.lower().endswith(exts):
                    files.append(os.path.join(root, fname))
                    labels.append(idx)
    return files, labels

all_files, all_labels = gather_files_labels(DATASET_DIR, CLASSES)
if len(all_files) == 0:
    raise RuntimeError(f"No image files found under {DATASET_DIR}. Check paths and filenames.")

train_files, val_files, train_labels, val_labels = train_test_split(
    all_files, all_labels, test_size=0.2, stratify=all_labels, random_state=SEED
)

# ---------------------------
# Augmentations
# ---------------------------
def train_augment(volume: np.ndarray) -> np.ndarray:
    if np.random.rand() < 0.5:
        volume = volume + np.random.normal(0, 0.01, volume.shape)
    if np.random.rand() < 0.5:
        volume = np.flip(volume, axis=2).copy()
    if np.random.rand() < 0.5:
        volume = np.flip(volume, axis=1).copy()
    return np.clip(volume, 0.0, 1.0).astype(np.float32)

def val_augment(volume: np.ndarray) -> np.ndarray:
    return volume.astype(np.float32)

# ---------------------------
# DataLoaders
# ---------------------------
train_dataset = LungCancerDataset(train_files, train_labels, transform=train_augment)
val_dataset = LungCancerDataset(val_files, val_labels, transform=val_augment)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

print(f"✅ Device: {PRINT_DEVICE}")
print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
print(f"Batch size: {BATCH_SIZE}, Depth: {DEPTH}, HxW: {INPUT_H}x{INPUT_W}")

# ---------------------------
# Model / Loss / Optimizer
# ---------------------------
model = LungCancer3DCNN(num_classes=len(CLASSES)).to(DEVICE)
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_acc = 0.0
best_path = "lung_cancer_model_best.pth"
latest_path = "lung_cancer_model_latest.pth"

# ---------------------------
# Training loop
# ---------------------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} - Train", leave=False)
    for volumes, labels in loop:
        volumes = volumes.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(volumes)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        batch_correct = (preds == labels).sum().item()
        batch_total = labels.size(0)

        running_loss += loss.item() * batch_total
        running_correct += batch_correct
        running_total += batch_total

        loop.set_postfix(loss=running_loss / running_total if running_total else 0.0,
                         acc=running_correct / running_total if running_total else 0.0)

    epoch_train_loss = running_loss / running_total
    epoch_train_acc = running_correct / running_total

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        loop_val = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} - Val", leave=False)
        for volumes, labels in loop_val:
            volumes = volumes.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(volumes)
            loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            val_loss += loss.item() * labels.size(0)

            loop_val.set_postfix(vloss=val_loss / val_total if val_total else 0.0,
                                 vacc=val_correct / val_total if val_total else 0.0)

    epoch_val_loss = val_loss / val_total
    epoch_val_acc = val_correct / val_total

    print(f"Epoch {epoch:02d}/{EPOCHS}  "
          f"Train Loss: {epoch_train_loss:.4f}  Train Acc: {epoch_train_acc:.4f}  "
          f"Val Loss: {epoch_val_loss:.4f}  Val Acc: {epoch_val_acc:.4f}")

    # Save checkpoints
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_loss": epoch_train_loss,
        "train_acc": epoch_train_acc,
        "val_loss": epoch_val_loss,
        "val_acc": epoch_val_acc
    }, latest_path)

    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save(model.state_dict(), best_path)
        print(f"✅ New best model saved (val_acc={best_val_acc:.4f}) -> {best_path}")

print("Training complete. Best val accuracy: {:.4f}".format(best_val_acc))
print(f"Final model checkpoints: latest -> {latest_path}, best -> {best_path}")
