import torch
from torch.utils.data import Dataset
from data_preparation import load_data, kmer_sequence

class SequenceDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.sequences, self.labels = load_data(file_path)
        # 动态设置最大长度
        self.max_length = max_length
        texts = [kmer_sequence(seq, config.KMER) for seq in self.sequences]
        encodings = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']