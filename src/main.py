import argparse
import config

from train import run_training
from evaluate import run_evaluation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate BERT classification model for a cell line.")
    parser.add_argument('--cell_line', type=str, required=True, help="Name of the cell line (folder name in data directory).")
    args = parser.parse_args()
    
    cell_line = args.cell_line
    print(f"Starting training for cell line: {cell_line}")
    run_training(cell_line)
    print(f"Training complete. Starting evaluation for cell line: {cell_line}")
    run_evaluation(cell_line)
    print("Evaluation complete.")
