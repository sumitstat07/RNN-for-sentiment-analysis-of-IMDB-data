# RNN for Sentiment Analysis on IMDB Data

A Recurrent Neural Network (RNN), built from scratch with PyTorch, that classifies IMDB movie reviews as **positive** or **negative**. This project walks through the complete NLP pipeline — from raw text to a trained deep learning model — including cleaning, preprocessing, feature extraction, and evaluation.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview

| | |
|---|---|
| **Task** | Binary sentiment classification (positive / negative) |
| **Model** | Single-layer RNN (`torch.nn.RNN`) with a fully connected output layer |
| **Framework** | PyTorch |
| **Feature Extraction** | TF-IDF (top 5,000 features) |
| **Dataset Size** | 50,000 movie reviews |

## Dataset

This project uses the [**IMDB Dataset of 50K Movie Reviews**](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews), a balanced dataset of 25,000 positive and 25,000 negative reviews commonly used as a benchmark for binary sentiment classification.

> **Note:** The dataset (`IMDB Dataset.csv`, ~64 MB) is **not included in this repository** to keep it lightweight. Download it from the Kaggle link above and place it in the project root before running the notebook.

## Pipeline

The notebook follows these stages end to end:

1. **Data Loading** — Read the CSV into a pandas DataFrame and remove duplicate reviews.
2. **Text Preprocessing**
   - Convert text to lowercase
   - Remove URLs
   - Remove punctuation
   - Strip HTML tags
   - Remove stopwords (via NLTK)
   - Apply stemming (Porter Stemmer)
3. **Label Encoding** — Convert sentiment labels (`positive`/`negative`) into binary integers.
4. **Vectorization** — Transform cleaned text into numerical features using **TF-IDF** (top 5,000 terms).
5. **Train/Test Split** — 80/20 split for training and evaluation.
6. **Model Training** — Train the RNN using `BCELoss` and the Adam optimizer.
7. **Evaluation** — Measure classification accuracy on the held-out test set.

## Model Architecture

```
Input (TF-IDF vector, 5000 features)
        ↓
   nn.RNN(input_size=5000, hidden_size=128, num_layers=1)
        ↓
   Fully Connected Layer (128 → 1)
        ↓
   Sigmoid Activation
        ↓
Output (Probability of Positive Sentiment)
```

- **Loss function:** Binary Cross-Entropy Loss (`BCELoss`)
- **Optimizer:** Adam
- **Epochs:** 10
- **Batch size:** 64

## Project Structure

```
.
├── RNN_for_Sentiment_Analysis_on_IMDB_Data.ipynb   # Main notebook (data prep → training → evaluation)
├── README.md                                        # Project documentation
└── .gitignore                                        # Excludes dataset, checkpoints, and cache files
```

## Installation

Clone the repository:

```bash
git clone https://github.com/sumitstat07/RNN-for-sentiment-analysis-of-IMDB-data.git
cd RNN-for-sentiment-analysis-of-IMDB-data
```

Install the required dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn torch nltk
```

Download the NLTK resources used for stopword removal and tokenization (run once):

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
```

## Usage

1. Download `IMDB Dataset.csv` from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place it in the project root.
2. Launch the notebook:
   ```bash
   jupyter notebook RNN_for_Sentiment_Analysis_on_IMDB_Data.ipynb
   ```
3. Run all cells in order — the notebook handles preprocessing, training, and evaluation sequentially.

## Results

The model is trained for 10 epochs on an 80/20 train-test split. Final classification accuracy on the test set is printed at the end of the notebook.

*(Consider adding your actual accuracy score here once training completes, e.g. "Test Accuracy: XX.X%".)*

## Future Improvements

- Replace the vanilla RNN with **LSTM** or **GRU** layers to better capture long-range dependencies in text
- Use pretrained word embeddings (**GloVe**, **word2vec**) instead of TF-IDF for richer semantic representation
- Add **early stopping** and validation-based checkpointing to prevent overfitting
- Experiment with a **bidirectional RNN** to capture context from both directions
- Add a confusion matrix and precision/recall/F1 metrics for a fuller evaluation picture

## License

This project is open source and available under the [MIT License](LICENSE).

---

*Built by [Sumit Sana](https://github.com/sumitstat07)*
