# # from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
# # import pandas as pd
# # import torch
# # from transformers import BertTokenizer, BertForSequenceClassification
# # import os
# #
# # app = Flask(__name__)
# #
# # # Load the model and tokenizer from their respective directories
# # MODEL_PATH = "./bert_model"
# # TOKENIZER_PATH = "./bert_tokenizer"
# # model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
# # tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
# #
# # # Ensure model is in evaluation mode
# # model.eval()
# #
# # # Set CSV storage path
# # PREDICTIONS_CSV = "./static/predicted_reviews.csv"
# #
# # # Helper function: Generate embeddings for a given text
# # def generate_embedding(text):
# #     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
# #     with torch.no_grad():
# #         outputs = model(**inputs)
# #     return outputs.logits
# #
# # # Helper function: Compare drift between embeddings
# # def detect_drift(old_embeddings, new_embeddings, threshold=0.1):
# #     # Calculate mean squared error between embeddings
# #     drift_value = torch.nn.functional.mse_loss(old_embeddings, new_embeddings).item()
# #     return drift_value, drift_value > threshold
# #
# # # Route: Render the main web interface
# # @app.route("/")
# # def index():
# #     return render_template("index.html")
# #
# # # Route: Predict sentiment for user input text
# # @app.route("/predict_text", methods=["POST"])
# # def predict_text():
# #     input_text = request.form.get("input_text")
# #     if not input_text:
# #         return render_template("index.html", message="Please enter some text.")
# #
# #     # Generate embedding for the input text
# #     embedding = generate_embedding(input_text)
# #     sentiment = "Positive" if torch.argmax(embedding).item() == 1 else "Negative"
# #
# #     return render_template("index.html", message=f"Predicted Sentiment: {sentiment}")
# #
# # # Route: Upload CSV and check for drift
# # @app.route("/upload_csv", methods=["POST"])
# # def upload_csv():
# #     file = request.files.get("csv_file")
# #     if not file:
# #         return render_template("index.html", message="No file uploaded.")
# #
# #     # Load uploaded CSV
# #     df = pd.read_csv(file)
# #     if "text" not in df.columns:
# #         return render_template("index.html", message="CSV must contain a 'text' column.")
# #
# #     # Generate embeddings for all texts in the uploaded CSV
# #     new_embeddings = torch.vstack([generate_embedding(text) for text in df["text"]])
# #
# #     # Load old embeddings (assumed from the model or existing data)
# #     # For demonstration, use the first text embedding as "old" embedding
# #     old_embeddings = new_embeddings[0].unsqueeze(0)
# #
# #     # Detect drift between new and old embeddings
# #     drift_value, is_drifted = detect_drift(old_embeddings, new_embeddings)
# #
# #     if is_drifted:
# #         message = f"Drift detected! Drift Value: {drift_value:.4f}. Retraining recommended."
# #     else:
# #         message = f"No significant drift detected. Drift Value: {drift_value:.4f}."
# #
# #     # Save predictions and drift info to CSV
# #     df["predicted_sentiment"] = ["Positive" if torch.argmax(e).item() == 1 else "Negative" for e in new_embeddings]
# #     df.to_csv(PREDICTIONS_CSV, index=False)
# #
# #     return render_template("index.html", message=message)
# #
# # # Route: Download the predictions CSV
# # @app.route("/download_csv")
# # def download_csv():
# #     if os.path.exists(PREDICTIONS_CSV):
# #         return send_file(PREDICTIONS_CSV, as_attachment=True)
# #     return render_template("index.html", message="No predictions available to download.")
# #
# # if __name__ == "__main__":
# #     # Run the Flask app
# #     app.run(host="0.0.0.0", port=5000, debug=True)
# from flask import Flask, render_template, request, jsonify
# import pandas as pd
# import torch
# from tqdm import tqdm
# from transformers import BertTokenizer, BertForSequenceClassification
# from scipy.stats import wasserstein_distance
#
# app = Flask(__name__)
#
# # Load the pre-trained tokenizer and model
# tokenizer = BertTokenizer.from_pretrained("./bert_tokenizer")
# model = BertForSequenceClassification.from_pretrained("./bert_model")
# model.eval()
#
#
# # Function to parse embeddings from strings
# def parse_embedding(embedding_str):
#     return list(map(float, embedding_str.strip("[]").split()))
#
#
# # Function to generate embeddings
# def generate_embeddings(reviews):
#     embeddings = []
#     for review in tqdm(reviews, desc="Generating embeddings"):
#         inputs = tokenizer(review, return_tensors='pt', truncation=True, padding=True)
#         with torch.no_grad():
#             outputs = model(**inputs)
#             logits = outputs.logits
#             embedding = logits.numpy().flatten()  # Flatten logits as embedding
#         embeddings.append(embedding)
#     return embeddings
#
#
# # Function to compare embeddings and calculate drift
# def calculate_drift(current_embeddings, historical_embeddings):
#     historical_mean_embedding = torch.mean(torch.tensor(historical_embeddings), dim=0).numpy()
#     current_mean_embedding = torch.mean(torch.tensor(current_embeddings), dim=0).numpy()
#     drift_score = wasserstein_distance(historical_mean_embedding, current_mean_embedding)
#     return drift_score
#
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
#
# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400
#     if file and file.filename.endswith('.csv'):
#         df = pd.read_csv(file)
#
#         # Assume the column containing reviews is 'review'
#         review_column = 'review' if 'review' in df.columns else None
#         if not review_column:
#             return jsonify({"error": "CSV must contain a 'review' column."}), 400
#
#         # Generate embeddings for current reviews
#         current_embeddings = generate_embeddings(df[review_column].tolist())
#         df['embedding'] = current_embeddings
#
#         # Load historical embeddings (assumed to be precomputed and saved)
#         historical_embeddings_df = pd.read_csv("embeddings.csv")
#         historical_embeddings = historical_embeddings_df['embedding'].apply(parse_embedding).tolist()
#
#         # Calculate drift
#         drift_score = calculate_drift(current_embeddings, historical_embeddings)
#
#         # Check drift condition
#         threshold = 1.0
#         if drift_score > threshold:
#             # Here, you would implement your logic to assign confidence scores and retrain the model
#             return jsonify({"drift_score": drift_score, "action": "retrain model."}), 200
#         else:
#             return jsonify({"drift_score": drift_score, "action": "no retraining necessary."}), 200
#     return jsonify({"error": "File format not supported. Please upload a CSV file."}), 400
#
#
# if __name__ == '__main__':
#     app.run(debug=True)

# app.py

import os
import pandas as pd
import torch
from flask import Flask, request, render_template
from transformers import BertTokenizer, BertForSequenceClassification
from scipy.stats import wasserstein_distance
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained tokenizer and model
tokenizer = BertTokenizer.from_pretrained(r"./bert_tokenizer")
model = BertForSequenceClassification.from_pretrained(r"./bert_model")
model.eval()

# Custom dataset for retraining
class ReviewDataset(Dataset):
    def __init__(self, reviews, labels):
        self.reviews = reviews
        self.labels = labels

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        label = self.labels[idx]
        inputs = tokenizer(review, return_tensors='pt', truncation=True, padding=True)
        return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0), label

# Function to make predictions
def predict_sentiment(input_data):
    inputs = tokenizer(input_data, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

# Function to generate embeddings
def generate_embeddings(reviews):
    embeddings = []
    for review in tqdm(reviews, desc="Generating embeddings"):
        inputs = tokenizer(review, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            embedding = logits.numpy().flatten()  # Flatten logits as embedding
        embeddings.append(embedding)
    return embeddings

# Function to parse embeddings from strings
def parse_embedding(embedding_str):
    return list(map(float, embedding_str.strip("[]").split()))

# Function to retrain the model
def retrain_model(review_df):
    model.train()  # Set the model to training mode

    reviews = review_df['review'].tolist()
    labels = review_df['predicted_label'].tolist()
    dataset = ReviewDataset(reviews, labels)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(3):  # Number of epochs
        for input_ids, attention_mask, label in tqdm(dataloader, desc="Retraining model"):
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=label)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    # Save the retrained model
    model.save_pretrained(r"./bert_model")
    tokenizer.save_pretrained(r"./bert_tokenizer")
    print("Model retrained and saved.")

# Function to handle drift detection and retraining
def handle_drift_and_retrain(input_csv_path):
    review_df = pd.read_csv(input_csv_path)

    # Check if there are at least 900 reviews
    if len(review_df) >= 900:
        print("Generating embeddings and starting data drift calculation...")
        review_embeddings = generate_embeddings(review_df['review'].tolist())
        review_df['embedding'] = review_embeddings

        # Load historical embeddings
        historical_embeddings_csv_path = r"embeddings.csv"
        historical_embeddings_df = pd.read_csv(historical_embeddings_csv_path)

        # Parse stored embeddings from strings
        historical_embeddings = historical_embeddings_df['embedding'].apply(parse_embedding).tolist()
        current_embeddings = review_df['embedding'].tolist()

        # Calculate mean embeddings
        historical_mean_embedding = torch.mean(torch.tensor(historical_embeddings), dim=0).numpy()
        current_mean_embedding = torch.mean(torch.tensor(current_embeddings), dim=0).numpy()

        # Calculate Wasserstein Distance for drift detection
        drift_score = wasserstein_distance(historical_mean_embedding, current_mean_embedding)
        print(f"Drift Score: {drift_score}")

        # Optional: Threshold for drift detection
        threshold = 1
        if drift_score > threshold:
            print("Significant drift detected. Generating predictions for reviews.")

            # Initialize lists for predictions and confidence scores
            predicted_labels = []
            confidence_scores = []

            # Predict labels and confidence scores
            for review in tqdm(review_df['review'], desc="Predicting labels"):
                inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=1)
                    confidence, predicted_class = torch.max(probabilities, dim=1)
                    predicted_labels.append(predicted_class.item())
                    confidence_scores.append(confidence.item())

            # Add labels and confidence scores to DataFrame
            review_df['predicted_label'] = predicted_labels
            review_df['confidence_score'] = confidence_scores

            # Filter by confidence score >= 0.8
            review_df = review_df[review_df['confidence_score'] >= 0.8]

            # Save the updated CSV
            review_df.to_csv(input_csv_path, index=False)
            print(f"Updated review predictions saved to {input_csv_path}.")

            # Retrain the model with the updated DataFrame
            retrain_model(review_df)
        else:
            print("No significant drift detected. No predictions needed.")
    else:
        print(f"Number of reviews: {len(review_df)}. Waiting for at least 900 reviews.")

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling form submission
@app.route('/predict', methods=['POST'])
def predict():
    input_type = request.form['input_type']
    if input_type == 'text':
        input_text = request.form['review']
        predicted_class = predict_sentiment(input_text)
        sentiment = "Positive" if predicted_class == 1 else "Negative"
        review_records = [input_text]
        output_review_csv_path = r"./text_review.csv"
        review_df = pd.DataFrame(review_records, columns=['review'])
        review_df.to_csv(output_review_csv_path, mode='a', index=False, header=not pd.io.common.file_exists(output_review_csv_path))
        return f"Predicted sentiment: {sentiment} (Class: {predicted_class})<br>Review saved to {output_review_csv_path}."

    elif input_type == 'csv':
        input_csv_path = request.form['csv_file_path']
        handle_drift_and_retrain(input_csv_path)
        return f"Batch processing completed for {input_csv_path}."

if __name__ == "__main__":
    app.run(debug=True)
