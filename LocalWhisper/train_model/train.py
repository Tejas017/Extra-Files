# import torch
# # model = your trained PyTorch model
# dummy_input = torch.randn(1, num_frames, num_features)  # match input shape
# torch.onnx.export(model, dummy_input, "models/prognosis.onnx", opset_version=11)
import torch
import torch.nn as nn

# Example small model
class SmallWakeWordNet(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.fc = nn.Linear(input_features, 2)  # 2 classes: keyword / background

    def forward(self, x):
        # x shape: (batch, frames, features)
        x = x.mean(dim=1)  # simple average over frames
        return self.fc(x)

num_frames = 100
num_features = 40

model = SmallWakeWordNet(num_features)

# Dummy input to match input shape
dummy_input = torch.randn(1, num_frames, num_features)

# Export to ONNX
torch.onnx.export(model, dummy_input, "models/prognosis.onnx", opset_version=11)
