import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import soundfile as sf
from sklearn.model_selection import train_test_split

# ----------------------------
# Parameters
# # ----------------------------
# KEYWORD_DIR = "samples/keyword"        # folder with your wake word recordings
# BACKGROUND_DIR = "samples/background"  # folder with negative/background audio
MODEL_SAVE_PATH = "models/prognosis.onnx"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # train_model folder
KEYWORD_DIR = os.path.join(BASE_DIR, "..", "samples", "keyword")
BACKGROUND_DIR = os.path.join(BASE_DIR, "..", "samples", "background")

# Optional: verify
print("Keyword folder:", KEYWORD_DIR)
print("Files:", os.listdir(KEYWORD_DIR))
SAMPLE_RATE = 16000
NUM_FRAMES = 100        # number of frames per sample
NUM_FEATURES = 40       # number of features per frame (MFCC or log-Mel)
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.001

# ----------------------------
# Simple feature extraction: log-Mel spectrogram
# ----------------------------
import librosa

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=NUM_FEATURES)
    log_mel = librosa.power_to_db(mel)
    
    # Pad/truncate to NUM_FRAMES
    if log_mel.shape[1] < NUM_FRAMES:
        pad_width = NUM_FRAMES - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0,0),(0,pad_width)), mode='constant')
    else:
        log_mel = log_mel[:, :NUM_FRAMES]
    
    return log_mel.T  # shape: (NUM_FRAMES, NUM_FEATURES)

# ----------------------------
# Load dataset
# ----------------------------
X, y = [], []

for f in os.listdir(KEYWORD_DIR):
    if f.endswith(".wav"):
        feats = extract_features(os.path.join(KEYWORD_DIR, f))
        X.append(feats)
        y.append(1)  # keyword

for f in os.listdir(BACKGROUND_DIR):
    if f.endswith(".wav"):
        feats = extract_features(os.path.join(BACKGROUND_DIR, f))
        X.append(feats)
        y.append(0)  # background

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# PyTorch Dataset
# ----------------------------
class WakeWordDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = WakeWordDataset(X_train, y_train)
val_dataset = WakeWordDataset(X_val, y_val)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ----------------------------
# Simple CNN model
# ----------------------------
class SmallWakeWordNet(nn.Module):
    def __init__(self, num_features, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(num_features, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(16, num_classes)
        
    def forward(self, x):
        # x shape: (batch, frames, features)
        x = x.transpose(1,2)  # (batch, features, frames)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

model = SmallWakeWordNet(NUM_FEATURES)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ----------------------------
# Training loop
# ----------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

# ----------------------------
# Export to ONNX
# ----------------------------
dummy_input = torch.randn(1, NUM_FRAMES, NUM_FEATURES)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
torch.onnx.export(model, dummy_input, MODEL_SAVE_PATH, opset_version=11)
print(f"ONNX model saved to {MODEL_SAVE_PATH}")
