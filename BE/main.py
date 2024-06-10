from fastapi import FastAPI, UploadFile, File
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from datasets import Dataset, load_metric
import tempfile
import fitz  # PyMuPDF
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

app = FastAPI()

origins = [
    "http://localhost:5000",  # Svelte development server
    "http://localhost:8080",  # FastAPI server (if needed)
    # Add other origins as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True)

# Load a pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Example dataset
# Example dataset with 20 text samples and their corresponding labels
data = {
    'text': [
        "This is the first document. It talks about machine learning.",
        "This document is about deep learning and neural networks.",
        "Natural language processing is a fascinating field.",
        "This text discusses the applications of AI in healthcare.",
        "The document covers the basics of reinforcement learning.",
        "This is an introduction to computer vision.",
        "The text explains the concept of transfer learning.",
        "This document is about the history of artificial intelligence.",
        "The text discusses the ethical implications of AI.",
        "This document covers the advancements in robotics.",
        "The text is about the use of AI in finance.",
        "This document discusses the future of autonomous vehicles.",
        "The text covers the basics of supervised learning.",
        "This document is about unsupervised learning techniques.",
        "The text discusses the role of AI in education.",
        "This document covers the applications of AI in agriculture.",
        "The text is about the impact of AI on employment.",
        "This document discusses the use of AI in cybersecurity.",
        "The text covers the basics of natural language understanding.",
        "This document is about the challenges of AI in healthcare."
    ],
    'label': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19
    ]
}
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True)

# Load a pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create a Dataset object
dataset = Dataset.from_dict(data)
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Split the dataset into training and validation sets
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Load a pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=20)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

# Define a compute_metrics function
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return load_metric("accuracy").compute(predictions=preds, references=p.label_ids)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Load the trained model and tokenizer
model = BertForSequenceClassification.from_pretrained('./model')
tokenizer = BertTokenizer.from_pretrained('./model')

# Define the category names
categories = [
    "Machine Learning", "Deep Learning", "Natural Language Processing", "AI in Healthcare",
    "Reinforcement Learning", "Computer Vision", "Transfer Learning", "History of AI",
    "Ethical Implications of AI", "Robotics", "AI in Finance", "Autonomous Vehicles",
    "Supervised Learning", "Unsupervised Learning", "AI in Education", "AI in Agriculture",
    "AI and Employment", "AI in Cybersecurity", "Natural Language Understanding", "Challenges of AI in Healthcare"
]

def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

@app.post("/classify/")
async def classify_pdf(file: UploadFile = File(...)):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    
    # Extract text from the uploaded PDF
    text = extract_text_from_pdf(tmp_path)
    
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Perform classification
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    return {"predicted_category": categories[predicted_class]}

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)