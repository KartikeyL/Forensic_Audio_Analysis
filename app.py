import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa

# --- CRNN MODEL DEFINITION -----------------------------------------------
class CRNNModel(nn.Module):
    def __init__(self, n_mels=128, hidden_size=128, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  
            nn.BatchNorm2d(32),                          
            nn.ReLU(),
            nn.MaxPool2d(2),                             

            nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64),                          
            nn.ReLU(),
            nn.MaxPool2d(2),                             

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),                         
            nn.ReLU()                                    
        )
        feat_dim = (n_mels // 4) * 128  # 128 mel bins pooled twice -> 32 freq
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.conv(x)  
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        out, _ = self.lstm(x)
        feat = out[:, -1, :]  
        return self.fc(feat)

# --- LOAD MODEL (cached) --------------------------------------------------
@st.cache_resource
def load_model(path: str, device: str):
    model = CRNNModel()
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model

# --- FEATURE EXTRACTION ---------------------------------------------------
def extract_melspec(audio_file, sr=22050, n_mels=128, hop_length=512):
    y, _ = librosa.load(audio_file, sr=sr)
    m = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length
    )
    log_m = librosa.power_to_db(m, ref=np.max)
    log_m = (log_m - log_m.mean()) / (log_m.std() + 1e-9)
    return torch.tensor(log_m).unsqueeze(0).unsqueeze(0).float()

# --- STREAMLIT USER INTERFACE ---------------------------------------------
def main():
    st.title("🔍 Audio Tampering Detector")
    st.markdown("""
    Upload two MP3 files:
    1. **Original Audio** (reference)
    2. **Test Audio** (to check)
    The app will show the probability the test audio is untampered.
    """)

    col1, col2 = st.columns(2)
    orig_file = col1.file_uploader("Original (.mp3)", type=["mp3"])
    test_file = col2.file_uploader("Test (.mp3)", type=["mp3"])

    if orig_file and test_file:
        st.audio(orig_file, format="audio/mp3")
        st.audio(test_file, format="audio/mp3")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model(
            r"D:\Major-project\FOR-Dataset\reduced\crnn_reduced_improved.pth",
            device
        )

        # Only test audio is passed through the classifier
        test_tensor = extract_melspec(test_file)
        logits = model(test_tensor.to(device))
        probs = F.softmax(logits, dim=1).cpu().detach().numpy()[0]
        untampered_prob = probs[1]  # index 1 = untampered

        st.metric("Probability Untampered", f"{untampered_prob:.3f}")
        if untampered_prob >= 0.5:
            st.success("Likely Untampered")
        else:
            st.error("Possible Tampering")

if __name__ == "__main__":
    main()
