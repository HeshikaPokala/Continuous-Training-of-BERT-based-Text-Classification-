# Continuous-Training-of-BERT-based-Text-Classification-
This project implements a scalable BERT-based text classification pipeline, designed to ensure continuous learning and performance monitoring. The system includes automated retraining triggered by data drift, supports both real-time and batch processing, and integrates seamlessly with a tokenizer for efficient data handling.

# Overview
The project focuses on creating a robust text classification system using a pre-trained BERT model. It is designed to handle both individual text inputs and large batches of data from CSV files. Additionally, it monitors data drift using Wasserstein Distance and retrains the model to adapt to new data distributions.

# Features
* Automated Retraining: The system detects data drift using Wasserstein Distance and automatically retrains the BERT model to maintain high accuracy.
* BERT-based Text Classification: Utilizes a pre-trained BERT model for accurate sentiment analysis across various text inputs.
* Real-time and Batch Processing: Supports both real-time predictions for single text inputs and batch processing from CSV files.
* Data Drift Detection: Continuously monitors for data drift, ensuring the model adapts to changing data distributions.
* Confidence Scoring: Assigns confidence scores to predictions, providing better insight into the model's certainty.
* REST-like Interface: Provides REST-like endpoints for seamless integration with other systems and applications.
* Efficient Tokenization: Integrates a tokenizer for handling large-scale data efficiently.
* Model Lifecycle Management: Uses MLflow for tracking, deploying, and managing the model lifecycle..


# Sentiment Analysis and Retraining Pipeline
This repository provides a comprehensive framework for sentiment analysis using BERT and includes mechanisms for detecting data drift and retraining the model as needed.

*  Sentiment Analysis
Classifies input text as Positive or Negative using a pre-trained BERT model.
* Supports:
Single text input.
Batch processing from a CSV file.
Saves predictions along with input reviews for further analysis.
*  Data Drift Detection
Compares embeddings of new data with historical embeddings.
Uses the Wasserstein Distance to detect significant drift in the data distribution.
Automatically retrains the model when the drift exceeds a defined threshold.
* Dataset Preparation
Handles raw review datasets, labeling them as "Positive" or "Negative" based on folder structure.
Prepares structured datasets for training and evaluation.

# Notebooks
## Retraining The Model
Implements the sentiment analysis pipeline.
Detects data drift and updates the model if needed.
* Outputs:
Predictions with confidence scores.
Retrained model and tokenizer (if required).
## Source Code
Prepares datasets for sentiment analysis by extracting and labeling reviews from a directory structure.
Provides a foundation for training/testing data processing.

# Sentiment Analysis Details
Sentiment is classified into two categories:
* Positive Review: Assigned a label of 1.
* Negative Review: Assigned a label of 0.

# Technologies Used
* Python
* PyTorch
* Transformers
* Pandas
* Scikit-learn
* MLflow
* TQDM (for progress tracking)
