import streamlit as st
import torch
import torch.nn as nn
import librosa
import numpy as np
import io

# Constants
SAMPLE_RATE = 16000
N_MELS = 128
FIXED_LENGTH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CRNN Model Definition
class CRNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CRNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.3)
        )
        self.lstm = nn.LSTM(input_size=64 * 64, hidden_size=128, num_layers=2,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

# Load CRNN Model
model = CRNN().to(DEVICE)
model.load_state_dict(torch.load("crnn_best.pth", map_location=DEVICE))
model.eval()

# Preprocess Function
def preprocess_audio(file):
    y, sr = librosa.load(file, sr=SAMPLE_RATE)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec = np.pad(mel_spec, ((0, 0), (0, max(0, FIXED_LENGTH - mel_spec.shape[1]))), mode="constant")[:, :FIXED_LENGTH]
    mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    return mel_spec

# Predict Function
def predict(file):
    input_tensor = preprocess_audio(io.BytesIO(file.read()))
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        real_prob = probs[0][0].item() * 100  # Real
        fake_prob = probs[0][1].item() * 100  # Fake
        return real_prob, fake_prob

# Streamlit App
st.title("🎙️ DeepFake Audio Tampering Detection")
st.markdown("Upload an **original (real)** audio and a **test** audio to check if the test audio is tampered (fake)")

# Upload both audios
original_file = st.file_uploader("Upload Original (Real) Audio", type=["wav", "mp3"], key="original")
test_file = st.file_uploader("Upload Test Audio", type=["wav", "mp3"], key="test")

if original_file and test_file:
    st.markdown("---")
    
    st.markdown("**Original Audio:**")
    st.audio(original_file, format='audio/wav')
    
    st.markdown("**Test Audio:**")
    st.audio(test_file, format='audio/wav')

    # Predict only on test audio
    real_prob, fake_prob = predict(test_file)

    st.markdown(f"**Untampered (Real) Probability: {real_prob:.2f}%**")
    st.progress(int(real_prob))

    st.markdown(f"**Tampered (Fake) Probability: {fake_prob:.2f}%**")
    st.progress(int(fake_prob))
    
    # Final larger result message
    if fake_prob > 50:
        st.markdown('<p style="color:red; font-size:36px; font-weight:bold;">This test audio is likely tampered (fake)</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:green; font-size:36px; font-weight:bold;">This test audio is likely untampered (real)</p>', unsafe_allow_html=True)
