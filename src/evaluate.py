import os
import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import pandas as pd

import config
from dataset import SequenceDataset

def run_evaluation(cell_line):
    # Ensure results directory exists
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Check GPU
    assert torch.cuda.is_available(), "GPU is required for evaluation."
    device = torch.device(config.DEVICE)
    
    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(config.MODEL_NAME, num_labels=2)
    # Load trained weights
    model_path = os.path.join(config.RESULTS_DIR, f"{cell_line}_model.pt")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Load test dataset
    test_path = os.path.join(config.DATA_DIR, cell_line, config.TEST_FILE)
    dataset = SequenceDataset(test_path, tokenizer)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE)
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
    
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    roc_path = os.path.join(config.RESULTS_DIR, f"{cell_line}_roc_curve.png")
    plt.savefig(roc_path)
    plt.close()
    
    # Compute Precision-Recall curve
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label=f'AUC = {pr_auc:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    pr_path = os.path.join(config.RESULTS_DIR, f"{cell_line}_pr_curve.png")
    plt.savefig(pr_path)
    plt.close()
    
    # Save predictions to CSV
    df = pd.DataFrame({
        'sequence': dataset.sequences,
        'true_label': all_labels,
        'pred_label': all_preds,
        'score': all_probs
    })
    csv_path = os.path.join(config.RESULTS_DIR, f"{cell_line}_predictions.csv")
    df.to_csv(csv_path, index=False)
    print(f"Evaluation complete. ROC curve: {roc_path}, PR curve: {pr_path}, predictions: {csv_path}")
