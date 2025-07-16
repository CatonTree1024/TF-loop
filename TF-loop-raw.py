import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertModel
from tqdm import tqdm
import optuna
from bertviz import head_view, model_view
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, auc
import torch.nn.functional as F
def load_dataset(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    data = []
    current_label = None
    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            label_match = re.match(r">(\d)_\d+", line)
            if label_match:
                current_label = int(label_match.group(1))
        else:
            if current_label is not None and len(line) > 0:
                data.append((current_label, line))
                current_label = None
    return pd.DataFrame(data, columns=['label', 'sequence'])

df_train = load_dataset("./BETA/train_sequences.txt")
df_test = load_dataset("./BETA/test_sequences.txt")
 
def tokenizer_kmer(sequence, k=1):
    return [sequence[i:i + k] for i in range(len(sequence) - k + 1)]

# Apply k-mer tokenization
df_train['tokens'] = df_train['sequence'].apply(lambda x: " ".join(tokenizer_kmer(x)))
df_test['tokens'] = df_test['sequence'].apply(lambda x: " ".join(tokenizer_kmer(x)))

# Save k-mer tokenization and labels to a txt file
df_train[['label', 'tokens']].to_csv('./BETA/kmer_1_train_tokens_labels.txt', index=False, header=False, sep='\t')
df_test[['label', 'tokens']].to_csv('./BETA/kmer_1_test_tokens_labels.txt', index=False, header=False, sep='\t')

# Print the length of the longest and shortest sequences
max_length = max(df_train['sequence'].apply(len).max(), df_test['sequence'].apply(len).max())
min_length = min(df_train['sequence'].apply(len).min(), df_test['sequence'].apply(len).min())
print(f"The length of the longest sequence is: {max_length}")
print(f"The length of the shortest sequence is: {min_length}")

# Dataset Class
class SequenceDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.texts = dataframe['tokens']
        self.labels = dataframe['label']
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        text = str(self.texts[index])
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return inputs, label

# Model Setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def objective(trial):
    # Hyperparameters to tune
    BATCH_SIZE = trial.suggest_categorical("BATCH_SIZE", [8, 16, 32])
    LEARNING_RATE = trial.suggest_loguniform("LEARNING_RATE", 1e-5, 5e-4)
    MAX_LENGTH = trial.suggest_int("MAX_LENGTH", 2, 168)
    
    train_dataset = SequenceDataset(df_train, tokenizer, MAX_LENGTH)
    test_dataset = SequenceDataset(df_test, tokenizer, MAX_LENGTH)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    EPOCHS = 10
    best_auc = 0
    for epoch in range(EPOCHS):
        # Training
        model.train()
        for inputs, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            inputs = {key: val.to(device) for key, val in inputs.items()}
            labels = labels.to(device)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        
        # Evaluation
        model.eval()
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs = {key: val.to(device) for key, val in inputs.items()}
                labels = labels.to(device)
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, axis=1)[:, 1]
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        auc = roc_auc_score(all_labels, all_probs)
        if auc > best_auc:
            best_auc = auc
    
    return best_auc

# Hyperparameter Tuning with Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

# Best Trial
best_trial = study.best_trial
print(f"Best AUC: {best_trial.value}")
print(f"Best Hyperparameters: {best_trial.params}")

# Train the best model
BATCH_SIZE = best_trial.params["BATCH_SIZE"]
LEARNING_RATE = best_trial.params["LEARNING_RATE"]
MAX_LENGTH = best_trial.params["MAX_LENGTH"]

train_dataset = SequenceDataset(df_train, tokenizer, MAX_LENGTH)
test_dataset = SequenceDataset(df_test, tokenizer, MAX_LENGTH)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Training the best model
EPOCHS = 10
best_auc = 0
best_model_state = None

# 保存预测结果
best_results = []

for epoch in range(EPOCHS):
    # Training phase
    model.train()
    train_loss = 0
    for inputs, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
        optimizer.zero_grad()
        inputs = {key: val.to(device) for key, val in inputs.items()}
        labels = labels.to(device)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_dataloader)

    # Evaluation phase
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = {key: val.to(device) for key, val in inputs.items()}
            labels = labels.to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, axis=1)[:, 1]
            preds = torch.argmax(logits, axis=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # 保存预测结果到best_results
            for true_label, pred_label, prob in zip(labels.cpu().numpy(), preds.cpu().numpy(), probs.cpu().numpy()):
                best_results.append((true_label, pred_label, prob))

    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    print(f"Epoch {epoch+1} - Loss: {avg_train_loss:.4f}, Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")

    # Save the best model
    if auc > best_auc:
        best_auc = auc
        best_model_state = model.state_dict()

# Saving the best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), "./BETA/best_model.pth")
    print(f"Best model saved with AUC: {best_auc:.4f}")

    # 保存最好的预测结果
    df_best_results = pd.DataFrame(best_results, columns=['True Label', 'Predicted Label', 'Predicted Probability'])
    df_best_results.to_csv('./best_predictions_and_labels.csv', index=False)

    # 绘制ROC曲线
    fpr, tpr, _ = roc_curve([r[0] for r in best_results], [r[2] for r in best_results])  # r[0]是真实标签，r[2]是预测概率
    roc_auc_value = roc_auc_score([r[0] for r in best_results], [r[2] for r in best_results])

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc_value:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig('./BETA/best_roc_curve.png')  # 保存ROC曲线为PNG文件
    plt.close()

    # 绘制PR曲线
    precision, recall, _ = precision_recall_curve([r[0] for r in best_results], [r[2] for r in best_results])
    pr_auc = average_precision_score([r[0] for r in best_results], [r[2] for r in best_results])

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.savefig('./BETA/best_pr_curve.png')  # 保存PR曲线为PNG文件
    plt.close()

    # 打印AUC值
    print(f'Best ROC AUC: {roc_auc_value:.4f}')
    print(f'Best PR AUC: {pr_auc:.4f}')
