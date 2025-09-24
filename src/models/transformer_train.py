import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
import mlflow
from sklearn.metrics import accuracy_score, f1_score

# ========================
# Step 1 – Load dataset
# ========================
df = pd.read_csv("data/raw/all_tickets_processed_improved_v3.csv")
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df['Topic_group'], random_state=42
)

# ========================
# Step 2 – Encode labels
# ========================
le = LabelEncoder()
train_labels = le.fit_transform(train_df['Topic_group'])
test_labels = le.transform(test_df['Topic_group'])

# ========================
# Step 3 – Tokenize texts
# ========================
tokenizer = DistilBertTokenizerFast.from_pretrained(
    'distilbert-base-multilingual-cased'
)
train_encodings = tokenizer(list(train_df['Document']), truncation=True, padding=True)
test_encodings = tokenizer(list(test_df['Document']), truncation=True, padding=True)

# ========================
# Step 4 – Create PyTorch dataset
# ========================
class TicketDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = TicketDataset(train_encodings, train_labels)
test_dataset = TicketDataset(test_encodings, test_labels)

# ========================
# Step 5 – Initialize model
# ========================
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-multilingual-cased',
    num_labels=len(le.classes_)
)

# ========================
# Step 6 – Training args
# ========================
training_args = TrainingArguments(
    output_dir='./models/transformer',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs',
    logging_steps=50
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# ========================
# Step 7 – MLflow tracking
# ========================
mlflow.start_run()
mlflow.log_param("model_name", "distilbert-base-multilingual-cased")
mlflow.log_param("num_labels", len(le.classes_))

# ========================
# Step 8 – Train
# ========================
trainer.train()

# ========================
# Step 9 – Evaluate
# ========================
preds = trainer.predict(test_dataset)
y_pred = preds.predictions.argmax(-1)
acc = accuracy_score(test_labels, y_pred)
f1 = f1_score(test_labels, y_pred, average='weighted')

mlflow.log_metric("accuracy", acc)
mlflow.log_metric("f1_score", f1)
mlflow.end_run()

# ========================
# Step 10 – Save model & tokenizer
# ========================
model.save_pretrained("./models/transformer")
tokenizer.save_pretrained("./models/transformer")

print("Training finished! Model and tokenizer saved in ./models/transformer")
print(f"Test Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
