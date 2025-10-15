import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import numpy as np
import os

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
MODEL_PATH = os.path.join(BASE_DIR, "../../models/transformer_model")

# -----------------------------
# Device
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load model, tokenizer, label encoder
# -----------------------------
print("Loading model and tokenizer...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
le = joblib.load(os.path.join(MODEL_PATH, "label_encoder.pkl"))

model.to(DEVICE)
model.eval()

print(f"✅ Model loaded on {DEVICE}")

# -----------------------------
# Prediction function
# -----------------------------
def predict(text):
    # Tokenize the input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Forward pass (no gradients)
    with torch.no_grad():
        logits = model(**inputs).logits

    pred_class = np.argmax(logits.cpu().numpy(), axis=1)
    return le.inverse_transform(pred_class)[0]


# -----------------------------
# Test it
# -----------------------------
if __name__ == "__main__":
    test_text = "Customer cannot access their account, please reset password"
    prediction = predict(test_text)
    print(f"🔮 Prediction: {prediction}")
