import os
import torch
from torch.utils.data import DataLoader, random_split
from transformers import BertForSequenceClassification, AdamW
from sklearn.metrics import roc_auc_score, accuracy_score
import optuna

import config
from dataset import SequenceDataset

def train_model(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate_model(model, data_loader, device):
    model.eval()
    preds = []
    true_labels = []
    probs = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            # Compute probabilities of positive class
            prob = torch.softmax(logits, dim=1)[:, 1]
            predictions = torch.argmax(logits, dim=1)
            preds.extend(predictions.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())
            probs.extend(prob.cpu().tolist())
    return true_labels, preds, probs

def objective(trial, train_dataset, val_dataset, device):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 5e-5)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Load model
    model = BertForSequenceClassification.from_pretrained(config.MODEL_NAME, num_labels=2)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training loop (few epochs)
    for epoch in range(config.EPOCHS):
        train_model(model, train_loader, optimizer, device)
    
    # Evaluate on validation data
    true_labels, preds, probs = evaluate_model(model, val_loader, device)
    try:
        auc = roc_auc_score(true_labels, probs)
    except:
        auc = accuracy_score(true_labels, preds)
    return auc

def run_training(cell_line):
    # Ensure results directory exists
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Check GPU
    assert torch.cuda.is_available(), "GPU is required for training."
    device = torch.device(config.DEVICE)
    
    # Load tokenizer
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Load training dataset
    train_path = os.path.join(config.DATA_DIR, cell_line, config.TRAIN_FILE)
    dataset = SequenceDataset(train_path, tokenizer)
    
    # Split dataset into training and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Hyperparameter tuning with Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, train_dataset, val_dataset, device),
                   n_trials=config.OPTUNA_TRIALS)
    
    best_lr = study.best_trial.params['lr']
    best_batch = study.best_trial.params['batch_size']
    best_wd = study.best_trial.params['weight_decay']
    print(f"Best hyperparameters: lr={best_lr}, batch_size={best_batch}, weight_decay={best_wd}")
    
    # Final training on full dataset
    full_loader = DataLoader(dataset, batch_size=best_batch, shuffle=True)
    model = BertForSequenceClassification.from_pretrained(config.MODEL_NAME, num_labels=2)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=best_lr, weight_decay=best_wd)
    
    for epoch in range(config.EPOCHS):
        loss = train_model(model, full_loader, optimizer, device)
        print(f"Epoch {epoch+1}/{config.EPOCHS}, Loss: {loss:.4f}")
    
    # Save the trained model
    model_file = os.path.join(config.RESULTS_DIR, f"{cell_line}_model.pt")
    torch.save(model.state_dict(), model_file)
    print(f"Model saved to {model_file}")
