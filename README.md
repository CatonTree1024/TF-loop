# TF-loop: Deciphering the transcription factor regulatory language for CTCF mediated chromatin loop based on BERT
## 项目简介
TF-loop 是一个基于 BERT 的序列分类模型，专用于不同细胞系的 TF Language 分类任务。该项目提供了数据加载、k-mer 分词、模型训练、超参数调优以及评估流程的完整实现。用户可以在提供的四个细胞系数据集上进行模型训练和评估。
## 安装与运行
### 1.克隆仓库并进入项目目录：
```
git clone https://github.com/CatonTree1024/TF-loop.git
cd TF-loop
```

### 2.安装依赖（确保已安装 Python3）：
```
pip install torch transformers optuna scikit-learn matplotlib pandas
```

### 3.使用 GPU 进行训练：本项目要求 GPU 环境（CUDA 支持）以加速模型训练。

### 4.运行命令示例：
```
python main.py --cell_line <CellLineName>
```
其中 <CellLineName> 是 data/ 目录下某一细胞系文件夹的名称（包含两个数据文件）。

## 数据结构
- 数据目录：data/，每个细胞系文件夹（如 BETA/）包含两个文本文件。  
        train.txt：训练集，每行包含一个序列和对应标签（用空格分隔）。  
        test.txt：测试集，格式同上。  
- 序列文件假设为 DNA 序列，标签为二分类标签（0 或 1）。  

## 模型输出
所有训练和评估结果保存在 results/ 目录下。  
- 训练好的模型文件：<CellLineName>_model.pt  
- 预测结果 CSV：<CellLineName>_predictions.csv，包含序列（sequence）、真实标签（true_label）、预测标签（pred_label）和预测分数（score）。  
- ROC 曲线图：<CellLineName>_roc_curve.png  
- PR 曲线图：<CellLineName>_pr_curve.png  

## 示例命令
在示例命令中，将以 BETA 细胞系为例：
```
python main.py --cell_line BETA
```
该命令将依次进行数据加载、超参数调优、模型训练和评估，并将所有输出文件保存到 results/ 目录中。

## 引用
若在学术论文或报告中使用本项目代码，请引用 TF-loop 项目或相关文档（例如通过 GitHub 链接）。您可以使用如下引用格式：
```
@misc{TF-loop,
  title={TF-loop: Deciphering the transcription factor regulatory language for CTCF mediated chromatin loop based on BERT},
  howpublished={\url{https://github.com/CatonTree1024/TF-loop}},
  year={2025}
}
```

