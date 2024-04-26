import requests
from bs4 import BeautifulSoup
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re

def read_txt(file_path):
    with open(file_path, "r") as file:
        text = file.read()
    return text


train_directory = 'q_and_a.txt'
text_data = read_txt(train_directory)
text = re.sub(r'\n+', '\n', text_data).strip()

# Tokenize the text
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2")

# Split the text into question-answer pairs
qa_pairs = text.split('\n')

# Separate questions and answers
questions = [qa_pairs[i] for i in range(0, len(qa_pairs), 2)]
answers = [qa_pairs[i] for i in range(1, len(qa_pairs), 2)]

print(f"Number of questions: {len(questions)}")
print(f"Number of answers: {len(answers)}")

# Tokenize questions and answers
input_encodings = tokenizer(questions, return_tensors='pt', padding=True, truncation=True, max_length=1024)
label_encodings = tokenizer(answers, return_tensors='pt', padding=True, truncation=True, max_length=1024)

# Determine the maximum sequence length
max_length = max(input_encodings['input_ids'].shape[1], label_encodings['input_ids'].shape[1])

# Tokenize questions and answers with padding to max_length
input_encodings = tokenizer(questions, return_tensors='pt', padding='max_length', max_length=max_length, truncation=True)
label_encodings = tokenizer(answers, return_tensors='pt', padding='max_length', max_length=max_length, truncation=True)

print(f"Shape of input tensor: {input_encodings['input_ids'].shape}")
print(f"Shape of label tensor: {label_encodings['input_ids'].shape}")

# Ensure the batch sizes match
assert input_encodings['input_ids'].shape[0] == label_encodings['input_ids'].shape[0], "Input and label batch sizes do not match"

input_ids = input_encodings['input_ids']
label_ids = label_encodings['input_ids']

# Split data into training and validation sets
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, label_ids, test_size=0.2)

# Set the model epochs
num_epochs = 60

# Set the model learning rate
learning_rate = 1e-5

# Initialize the optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Create a learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_inputs) * num_epochs)

# Train the model
model.train()

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(train_inputs, labels=train_labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    # Update the learning rate
    scheduler.step()

# Evaluate the model
model.eval()
val_outputs = model(val_inputs)
val_predictions = torch.argmax(val_outputs.logits, dim=-1)

# Calculate accuracy
accuracy = accuracy_score(val_labels.flatten().numpy(), val_predictions.flatten().numpy())
print(f"Accuracy: {accuracy}")

# Save the fine-tuned model
model.save_pretrained("model2.py")