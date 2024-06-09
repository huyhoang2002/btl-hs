import fitz  # PyMuPDF
import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Step 1: Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

pdf_path = 'doc.pdf'
pdf_text = extract_text_from_pdf(pdf_path)
print("Extracted Text:\n", pdf_text)

# Step 2: Preprocess Text
def preprocess_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

cleaned_text = preprocess_text(pdf_text)
print("Cleaned Text:\n", cleaned_text)

# Training data
documents = [
    "This is an invoice for the purchase of office supplies.",
    "John Doe's resume includes his work experience and education.",
    "This research paper discusses the effects of climate change.",
    "Dear Sir or Madam, I am writing to inform you...",
    "The annual report for the company includes financial statements.",
    "This contract outlines the terms and conditions of the agreement.",
    "The user manual provides instructions for operating the device.",
    "This presentation covers the company's quarterly performance.",
    "The receipt shows the items purchased and the total amount paid.",
    "This memo is to inform all employees about the new policy."
]

labels = [
    "Invoice",
    "Resume",
    "Research Paper",
    "Letter",
    "Report",
    "Contract",
    "Manual",
    "Presentation",
    "Receipt",
    "Memo"
]

# Preprocess and tokenize training documents
cleaned_documents = [preprocess_text(doc) for doc in documents]

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(cleaned_documents)
sequences = tokenizer.texts_to_sequences(cleaned_documents)
word_index = tokenizer.word_index

# Pad sequences
max_sequence_length = 100  # Adjust based on your needs
X_train = pad_sequences(sequences, maxlen=max_sequence_length)

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(labels)
y_train = to_categorical(y_train, num_classes=len(label_encoder.classes_))

# Step 3: Build and Train the RNN Model
# Define the RNN model
model = Sequential()
model.add(Embedding(input_dim=len(word_index) + 1, output_dim=128, input_length=max_sequence_length))
model.add(SimpleRNN(128))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))  # Number of document types

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=1)

# Step 4: Analyze the Text
# Tokenize and pad the new data
new_sequences = tokenizer.texts_to_sequences([cleaned_text])
new_data = pad_sequences(new_sequences, maxlen=max_sequence_length)

# Predict on new data
predictions = model.predict(new_data)
predicted_class = np.argmax(predictions, axis=1)
print("Predicted Class:\n", predicted_class)

# Map the predicted class to document type
document_types = label_encoder.classes_
predicted_document_type = document_types[predicted_class[0]]
print("Predicted Document Type:\n", predicted_document_type)