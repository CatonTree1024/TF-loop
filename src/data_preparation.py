#
import os
from config import KMER

def load_data(file_path):
    """
    Load sequences and labels from a text file.
    Each line should contain a sequence and a label separated by whitespace.
    """
    sequences = []
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            # First column is sequence, second is label
            seq = parts[0]
            label = None
            if len(parts) > 1:
                try:
                    label = int(parts[1])
                except:
                    label = parts[1]
            sequences.append(seq)
            if label is not None:
                labels.append(label)
    return sequences, labels

def kmer_tokenize(seq, k=None):
    """
    Convert a sequence string into a list of k-mers.
    """
    if k is None:
        k = KMER
    if k > len(seq):
        # If k-mer size is larger than sequence, return sequence as a single token
        return [seq]
    kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
    return kmers

def kmer_sequence(seq, k=None):
    """
    Tokenize a sequence into k-mers and join them as a space-separated string.
    Suitable for BERT tokenization of k-mer tokens.
    """
    kmers = kmer_tokenize(seq, k)
    return " ".join(kmers)
