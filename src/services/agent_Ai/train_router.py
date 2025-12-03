"""
Train DistilGPT-2 router model for intelligent service selection
"""

import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================
# Configuration
# ==============================
ROUTER_MODEL_PATH = "models/distilgpt2_router"
MAX_LENGTH = 256
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3

# ==============================
# Data Preparation
# ==============================
def load_and_prepare_data(training_data_path: str):
    """Load training data generated from services"""
    logger.info(f"Loading training data from {training_data_path}")
    df = pd.read_csv(training_data_path)
    
    if 'Document' not in df.columns or 'router_label' not in df.columns:
        raise ValueError("Training data must have 'Document' and 'router_label' columns")
    
    df = df.dropna(subset=['Document', 'router_label'])
    df['router_label'] = df['router_label'].astype(int)
    
    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"Label distribution:\n{df['router_label'].value_counts()}")
    
    return df

def create_train_val_split(df: pd.DataFrame, test_size: float = 0.2):
    """Split data into train and validation sets"""
    train_df, val_df = train_test_split(
        df[['Document', 'router_label']],
        test_size=test_size,
        random_state=42,
        stratify=df['router_label']
    )
    
    logger.info(f"Train set: {len(train_df)} samples")
    logger.info(f"Validation set: {len(val_df)} samples")
    return train_df, val_df

# ==============================
# Tokenization
# ==============================
def tokenize_dataset(df: pd.DataFrame, tokenizer, max_length: int):
    """Tokenize Document data"""
    def tokenize_function(examples):
        return tokenizer(
            examples['Document'],
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
    
    dataset = Dataset.from_pandas(df)
    logger.info("Tokenizing dataset...")
    
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['Document']
    )
    
    tokenized = tokenized.rename_column('router_label', 'labels')
    return tokenized

# ==============================
# Training
# ==============================
def train_router_model(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """Train DistilGPT-2 for routing"""
    logger.info("Initializing DistilGPT-2 model...")
    model_name = "distilgpt2"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        pad_token_id=tokenizer.pad_token_id
    )
    
    model.config.id2label = {0: "TF-IDF", 1: "Transformer"}
    model.config.label2id = {"TF-IDF": 0, "Transformer": 1}
    
    train_dataset = tokenize_dataset(train_df, tokenizer, MAX_LENGTH)
    val_dataset = tokenize_dataset(val_df, tokenizer, MAX_LENGTH)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir=ROUTER_MODEL_PATH,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        logging_dir=f"{ROUTER_MODEL_PATH}/logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        push_to_hub=False,
        report_to="none"
    )
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = (predictions == labels).mean()
        
        tfidf_mask = labels == 0
        transformer_mask = labels == 1
        
        tfidf_acc = (predictions[tfidf_mask] == labels[tfidf_mask]).mean() if tfidf_mask.sum() > 0 else 0
        transformer_acc = (predictions[transformer_mask] == labels[transformer_mask]).mean() if transformer_mask.sum() > 0 else 0
        
        return {
            "accuracy": accuracy,
            "tfidf_accuracy": tfidf_acc,
            "transformer_accuracy": transformer_acc
        }
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    logger.info(f"Saving model to {ROUTER_MODEL_PATH}")
    trainer.save_model(ROUTER_MODEL_PATH)
    tokenizer.save_pretrained(ROUTER_MODEL_PATH)
    
    logger.info("\n" + "="*60)
    logger.info("FINAL EVALUATION")
    logger.info("="*60)
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        logger.info(f"{key}: {value:.4f}")
    
    return trainer, tokenizer

# ==============================
# Evaluation Visualizations
# ==============================
def evaluate_and_visualize(trainer, val_df: pd.DataFrame, tokenizer):
    """Generate evaluation metrics and visualizations"""
    logger.info("Generating evaluation visualizations...")
    
    val_dataset = tokenize_dataset(val_df, tokenizer, MAX_LENGTH)
    predictions = trainer.predict(val_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(
        true_labels,
        pred_labels,
        target_names=['TF-IDF', 'Transformer'],
        digits=4
    ))
    
    cm = confusion_matrix(true_labels, pred_labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['TF-IDF', 'Transformer'],
        yticklabels=['TF-IDF', 'Transformer']
    )
    plt.title('Router Model Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    confusion_matrix_path = os.path.join(ROUTER_MODEL_PATH, "confusion_matrix.png")
    plt.savefig(confusion_matrix_path)
    logger.info(f"✅ Confusion matrix saved to {confusion_matrix_path}")
    plt.close()
    
    probs = np.max(predictions.predictions, axis=1)
    correct = pred_labels == true_labels
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(probs[correct], bins=30, alpha=0.7, label='Correct', color='green')
    plt.hist(probs[~correct], bins=30, alpha=0.7, label='Incorrect', color='red')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.title('Confidence Distribution')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.boxplot([probs[correct], probs[~correct]], labels=['Correct', 'Incorrect'])
    plt.ylabel('Confidence')
    plt.title('Confidence by Correctness')
    
    plt.tight_layout()
    confidence_plot_path = os.path.join(ROUTER_MODEL_PATH, "confidence_analysis.png")
    plt.savefig(confidence_plot_path)
    logger.info(f"✅ Confidence analysis saved to {confidence_plot_path}")
    plt.close()
    
    errors_df = val_df.copy()
    errors_df['predicted'] = pred_labels
    errors_df['correct'] = correct
    errors_df['confidence'] = probs
    errors = errors_df[~errors_df['correct']]
    
    if len(errors) > 0:
        print("\n" + "="*60)
        print(f"ERROR ANALYSIS - {len(errors)} errors ({len(errors)/len(val_df)*100:.2f}%)")
        print("="*60)
        print("\nSample errors:")
        for idx, row in errors.head(5).iterrows():
            print(f"\nDocument: {row['Document'][:100]}...")
            print(f"True: {'Transformer' if row['router_label'] == 1 else 'TF-IDF'}")
            print(f"Predicted: {'Transformer' if row['predicted'] == 1 else 'TF-IDF'}")
            print(f"Confidence: {row['confidence']:.2%}")
    
    results_path = os.path.join(ROUTER_MODEL_PATH, "evaluation_results.csv")
    errors_df.to_csv(results_path, index=False)
    logger.info(f"✅ Detailed results saved to {results_path}")

# ==============================
# Main
# ==============================
def main():
    import argparse
    
    # 👇 moved global line here to fix SyntaxError
    global ROUTER_MODEL_PATH, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE
    
    parser = argparse.ArgumentParser(description='Train DistilGPT-2 router')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training data CSV')
    parser.add_argument('--output', type=str, default=ROUTER_MODEL_PATH,
                       help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    ROUTER_MODEL_PATH = args.output
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    
    os.makedirs(ROUTER_MODEL_PATH, exist_ok=True)
    
    logger.info("="*60)
    logger.info("TRAINING DISTILGPT-2 ROUTER MODEL")
    logger.info("="*60)
    logger.info(f"Training data: {args.data}")
    logger.info(f"Output path: {ROUTER_MODEL_PATH}")
    logger.info(f"Epochs: {NUM_EPOCHS}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Learning rate: {LEARNING_RATE}")
    logger.info("="*60)
    
    df = load_and_prepare_data(args.data)
    train_df, val_df = create_train_val_split(df)
    trainer, tokenizer = train_router_model(train_df, val_df)
    evaluate_and_visualize(trainer, val_df, tokenizer)
    
    logger.info("\n" + "="*60)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("="*60)
    logger.info(f"Model saved to: {ROUTER_MODEL_PATH}")
    logger.info("\nNext steps:")
    logger.info("1. Review evaluation metrics and visualizations")
    logger.info("2. Test the router: python test_router.py")
    logger.info("3. Deploy agent_ai_service.py")

if __name__ == "__main__":
    main()