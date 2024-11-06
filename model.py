import torch
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
import time
import datetime

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# 1. Data Preparation

# Load the dataset
newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# Create a DataFrame
df = pd.DataFrame({'text': newsgroups_data.data, 'target': newsgroups_data.target})
df['target_names'] = df['target'].apply(lambda x: newsgroups_data.target_names[x])

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['target_names'])

# Split data into train and test sets
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    df['text'], df['label'],
    random_state=42,
    test_size=0.3,
    stratify=df['label']
)

val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels,
    random_state=42,
    test_size=0.5,
    stratify=temp_labels
)

# 2. Tokenization and Formatting

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def tokenize_texts(texts, labels, max_len=256):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,                      # Input text
            add_special_tokens=True,   # Add '[CLS]' and '[SEP]'
            max_length=max_len,        # Pad & truncate all sentences
            padding='max_length',
            truncation=True,
            return_attention_mask=True,   # Construct attention masks
            return_tensors='pt',          # Return pytorch tensors
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels.values)

    return input_ids, attention_masks, labels

# Tokenize datasets
train_inputs, train_masks, train_labels = tokenize_texts(train_texts, train_labels)
val_inputs, val_masks, val_labels = tokenize_texts(val_texts, val_labels)
test_inputs, test_masks, test_labels = tokenize_texts(test_texts, test_labels)

# 3. Create DataLoaders

batch_size = 16

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

# 4. Model Definition

# Load BERT pre-trained model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(label_encoder.classes_),
    output_attentions=False,
    output_hidden_states=False,
)
model.to(device)

# 5. Optimizer and Scheduler

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 3
total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# 6. Training Loop

def train_model():
    loss_values = []

    for epoch_i in range(epochs):
        total_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            outputs = model(
                b_input_ids,
                attention_mask=b_input_mask,
                labels=b_labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        loss_values.append(avg_train_loss)
    
    return model, loss_values

# 7. Prediction Function

def predict(texts):
    model.eval()

    inputs, masks, _ = tokenize_texts(texts, np.zeros(len(texts)))

    with torch.no_grad():
        outputs = model(inputs.to(device), attention_mask=masks.to(device))

    logits = outputs.logits
    predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)

    return predictions

# 8. Save Model
def save_model():
    model.save_pretrained("./model")

# 9. Load Model
def load_model():
    model = BertForSequenceClassification.from_pretrained("./model")
    model.to(device)
    return model
