# Parameter-Efficient Fine-Tuning (PEFT) using LoRA

## Overview

This project explores **Parameter-Efficient Fine-Tuning (PEFT)** using Low-Rank Adaptation (LoRA) for fine-tuning pretrained transformer models.

The project compares traditional Full Fine-Tuning with LoRA-based fine-tuning using BERT (`bert-base-uncased`) on the AG News text classification dataset.

The objective is to understand how LoRA reduces the number of trainable parameters while maintaining competitive classification performance.

## Objectives

- Understand the concept of Parameter-Efficient Fine-Tuning (PEFT).
- Implement Full Fine-Tuning as a baseline.
- Apply LoRA adapters to a pretrained BERT model.
- Compare model performance, training time, and trainable parameters.
- Evaluate the trade-offs between computational efficiency and accuracy.

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT
- Scikit-learn
- Google Colab

## Dataset

**AG News Dataset**

The AG News dataset contains news articles categorized into four classes:

1. World
2. Sports
3. Business
4. Sci/Tech

For this experiment:
- Training samples: 30,000
- Evaluation samples: 5,000
- Maximum tokenization length: 256

## Methodology

### 1. Full Fine-Tuning

- Pretrained model: `bert-base-uncased`
- All model parameters are updated during training.
- Learning rate: `2e-5`
- Batch size: 16
- Epochs: 1
- Weight decay: 0.01

### 2. LoRA Fine-Tuning

LoRA introduces trainable low-rank matrices into selected layers of the pretrained model, allowing adaptation without updating all original model parameters.

Configuration:
- Rank (r): 8
- LoRA Alpha: 16
- LoRA Dropout: 0.1
- Target Modules: Query, Key, Value, Dense
- Learning rate: `2e-4`
- Batch size: 16
- Epochs: 1

Task type: Sequence Classification.

## Experimental Results

| Metric | Full Fine-Tuning | LoRA |
|---|---:|---:|
| Accuracy | 92.90% | 91.96% |
| Macro F1-Score | 92.84% | 91.88% |
| Training Time | 791.1 sec | 543.8 sec |
| Trainable Parameters | 100% | 1.21% |

### Key Observations

- LoRA reduced the trainable parameters to approximately 1.21% of the total model parameters.
- LoRA achieved a 91.96% classification accuracy, compared to 92.90% with Full Fine-Tuning.
- LoRA required less training time in this experiment.
- Full Fine-Tuning achieved slightly better classification performance, while LoRA provided a more parameter-efficient alternative.

## Conclusion

The experiment demonstrates how LoRA can significantly reduce the number of trainable parameters while retaining competitive performance.

Although Full Fine-Tuning achieved slightly higher accuracy, LoRA offered a more efficient approach to adapting pretrained transformer models.

This highlights the potential of PEFT techniques for fine-tuning large language models and transformer-based architectures with reduced computational requirements.

## Future Improvements

- Experiment with different LoRA ranks and alpha values.
- Apply LoRA to larger pretrained models such as RoBERTa.
- Explore quantization techniques such as 8-bit quantization.
- Implement saving and loading of LoRA adapters independently.
- Evaluate performance across different datasets.

## How to Run

1. Open the notebook in Google Colab.
2. Enable GPU acceleration from Runtime → Change runtime type.
3. Execute the installation and import cells.
4. Load and preprocess the AG News dataset.
5. Run the Full Fine-Tuning experiment.
6. Run the LoRA Fine-Tuning experiment.
7. Compare the evaluation metrics.

## Repository Contents

- `PEFT_LoRA_Final.ipynb` – Complete implementation of Full Fine-Tuning and LoRA-based fine-tuning.

---

**Note:** This is an experimental implementation using a subset of the AG News dataset and one training epoch. Results may vary across hardware and execution environments.
