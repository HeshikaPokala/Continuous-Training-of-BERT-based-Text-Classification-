# Continuous-Training-of-BERT-based-Text-Classification-
This project implements a scalable BERT-based text classification pipeline, designed to ensure continuous learning and performance monitoring. The system includes automated retraining triggered by data drift, supports both real-time and batch processing, and integrates seamlessly with a tokenizer for efficient data handling.

# Overview
The project focuses on creating a robust text classification system using a pre-trained BERT model. It is designed to handle both individual text inputs and large batches of data from CSV files. Additionally, it monitors data drift using Wasserstein Distance and retrains the model to adapt to new data distributions.

# Features
BERT-based Text Classification: Leverages BERT's powerful language understanding for sentiment analysis.
Automated Retraining: Continuously monitors data drift and triggers retraining to maintain high accuracy.
Data Handling: Supports both single text inputs and CSV batch processing.
Drift Detection: Uses Wasserstein Distance to detect data drift and ensure model performance over time.
Confidence Scores: Assigns confidence scores to predictions for better interpretability.
Model Deployment: Offers REST-like interfaces for real-time predictions and batch processing.

# Technologies Used
Python
PyTorch
Transformers
Pandas
Scikit-learn
MLflow
TQDM (for progress tracking)
