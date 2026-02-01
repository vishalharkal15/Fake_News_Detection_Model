# Fake News Detection System

A production-quality NLP pipeline for detecting fake news using DistilBERT/BERT transformers.

## 🚀 Features

- **Advanced NLP Model**: Uses DistilBERT (or BERT) from HuggingFace Transformers
- **Production-Ready**: Modular, well-documented code
- **Complete Pipeline**: Data processing, training, evaluation, and inference
- **SHAP Explainability**: Understand which words influence predictions
- **Easy to Use**: Simple `predict_news()` function for quick inference

## 📁 Project Structure

```
fake_news_detection/
├── src/
│   ├── __init__.py
│   ├── data_processing.py      # Data loading & preprocessing
│   ├── text_preprocessing.py   # Text cleaning for transformers
│   ├── model_training.py       # DistilBERT training with Trainer API
│   ├── evaluation.py           # Metrics & visualization
│   ├── inference.py            # predict_news() function
│   └── explainability.py       # SHAP explanations
├── train.py                    # Main training script
├── quickstart.py               # Quick demo script
├── news_dataset.csv            # Sample dataset
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🛠️ Installation

1. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 📊 Dataset Format

Your dataset should be a CSV file with these columns:
- `label`: Either "REAL" or "FAKE"
- `text`: The news article content

Example:
```csv
label,text
REAL,"Scientists announce new breakthrough in renewable energy..."
FAKE,"SHOCKING: This one weird trick will make you rich overnight!"
```

## 🎯 Quick Start

Run the quick start script to see the system in action:
```bash
python quickstart.py
```

## 🏋️ Training

### Basic Training
```bash
python train.py --data_path news_dataset.csv
```

### Full Training with Custom Parameters
```bash
python train.py \
    --data_path news_dataset.csv \
    --output_dir ./saved_model \
    --model_name distilbert-base-uncased \
    --epochs 3 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --max_length 512
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_path` | news_dataset.csv | Path to CSV dataset |
| `--output_dir` | ./saved_model | Where to save the model |
| `--model_name` | distilbert-base-uncased | HuggingFace model name |
| `--epochs` | 3 | Number of training epochs |
| `--batch_size` | 16 | Training batch size |
| `--learning_rate` | 2e-5 | Learning rate |
| `--max_length` | 512 | Max sequence length |
| `--early_stopping_patience` | 2 | Early stopping patience |

## 🔮 Making Predictions

### Using the predict_news() Function

```python
from src.inference import predict_news, load_predictor

# Load the model once
load_predictor('./saved_model')

# Make predictions
result = predict_news("Breaking news: Scientists discover new species in Amazon...")

print(f"Prediction: {result['prediction']}")  # "REAL" or "FAKE"
print(f"Confidence: {result['confidence']:.2%}")
```

### Using the FakeNewsPredictor Class

```python
from src.inference import FakeNewsPredictor

# Initialize predictor
predictor = FakeNewsPredictor('./saved_model')

# Single prediction
result = predictor.predict("Your news article here...")
print(result.prediction)     # "REAL" or "FAKE"
print(result.confidence)     # 0.95
print(result.fake_probability)  # 0.05
print(result.real_probability)  # 0.95

# Batch prediction
texts = ["Article 1...", "Article 2...", "Article 3..."]
results = predictor.predict_batch(texts)
```

## 📈 Evaluation Metrics

The system reports:
- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual breakdown of predictions
- **ROC-AUC**: Area under the ROC curve

## 🔍 Explainability (SHAP)

Understand why the model makes specific predictions:

```python
from src.explainability import FakeNewsExplainer, explain_prediction

# Quick explanation
explanation = explain_prediction(
    "Your news article text...",
    model_path='./saved_model'
)

# Get important words
print(explanation['word_importance'])

# Full explainer with visualization
explainer = FakeNewsExplainer('./saved_model')
explainer.visualize("Your text here...")
explainer.plot_word_importance("Your text here...")
```

## 🧩 Module Usage

### Data Processing

```python
from src.data_processing import DataProcessor

processor = DataProcessor()
train_df, test_df, stats = processor.process_pipeline(
    filepath='news_dataset.csv',
    test_size=0.2
)
```

### Text Preprocessing

```python
from src.text_preprocessing import TextPreprocessor, clean_text

preprocessor = TextPreprocessor()
cleaned = preprocessor("Check out https://example.com for more info!")
# Output: "Check out for more info!"

# Batch processing
texts = ["Text 1...", "Text 2..."]
cleaned_texts = preprocessor.preprocess_batch(texts)
```

### Model Training

```python
from src.model_training import FakeNewsTrainer, ModelConfig

config = ModelConfig(
    model_name="distilbert-base-uncased",
    num_epochs=3,
    batch_size=16,
    learning_rate=2e-5
)

trainer = FakeNewsTrainer(config)
trainer.train(train_texts, train_labels, val_texts, val_labels)
trainer.save_model('./saved_model')
```

### Evaluation

```python
from src.evaluation import ModelEvaluator

evaluator = ModelEvaluator()
metrics = evaluator.generate_full_report(
    y_true=true_labels,
    y_pred=predictions,
    y_prob=probabilities,
    save_dir='./results'
)
```

## 💡 Tips for Better Results

1. **Use more data**: The sample dataset is small. For production, use datasets with thousands of samples.

2. **Balanced classes**: Ensure roughly equal numbers of FAKE and REAL samples.

3. **Quality data**: Remove duplicates and ensure labels are accurate.

4. **Fine-tuning**: Try different hyperparameters:
   - Learning rates: 1e-5, 2e-5, 3e-5
   - Batch sizes: 8, 16, 32
   - Epochs: 2-5 (more can lead to overfitting)

5. **Model choice**: 
   - `distilbert-base-uncased`: Faster, good for most cases
   - `bert-base-uncased`: Slightly better accuracy, slower

## 🔧 Troubleshooting

### Out of Memory (OOM)
- Reduce batch size: `--batch_size 8`
- Reduce max length: `--max_length 256`

### Slow Training
- Use GPU if available
- Use DistilBERT instead of BERT
- Reduce dataset size for testing

### Poor Results
- Check data quality and labeling
- Try different learning rates
- Ensure balanced class distribution
- Use more training data

## 📄 License

MIT License - feel free to use and modify for your projects.

## 🤝 Contributing

Contributions welcome! Please feel free to submit issues or pull requests.
