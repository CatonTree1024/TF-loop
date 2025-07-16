import os

# 项目名称
PROJECT_NAME = "TF-loop"

# 数据目录（每个细胞系子文件夹下应包含 TRAIN_FILE 和 TEST_FILE）
DATA_DIR = os.path.join(os.getcwd(), "data")
TRAIN_FILE = "train.txt"
TEST_FILE = "test.txt"

# k-mer 大小
KMER = 1

# BERT 模型名称
MODEL_NAME = "bert-base-uncased"

# Optuna 超参数搜索次数
OPTUNA_TRIALS = 10

# 训练 Epoch 数量
EPOCHS = 10

# Batch size 默认
BATCH_SIZE = 16

# 设备配置，确保使用 GPU
DEVICE = "cuda:0"

# 结果保存目录
RESULTS_DIR = os.path.join(os.getcwd(), "results")
