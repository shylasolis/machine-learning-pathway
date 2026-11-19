"""
LSTM-based Text Classification with PyTorch

This script demonstrates a Long Short-Term Memory (LSTM) neural network for binary
sentiment classification. Unlike Bag of Words (which discards word order), LSTM networks
capture sequential dependencies and context in text.

Architecture:
  - Embedding Layer: Converts word IDs to dense vectors (learns word relationships)
  - LSTM Layer: Processes sequences and learns temporal patterns
  - Fully Connected (Dense) Layer: Makes binary classification decision
  - Sigmoid Activation: Outputs probability between 0 and 1

The model learns to predict sentiment (positive=1, negative=0) from the sequence
of words and their context.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ===== STEP 1: Prepare Data =====
print("=" * 70)
print("STEP 1: Prepare Data")
print("=" * 70)

documents = [
    "I love programming in Python",
    "Python is a great language",
    "I enjoy machine learning",
    "I dislike bugs in my code",
    "Debugging can be frustrating"
]
labels = [1, 1, 1, 0, 0]  # 1 for positive sentiment, 0 for negative sentiment

print(f"  Total documents: {len(documents)}")
print(f"  Positive samples: {sum(labels)}")
print(f"  Negative samples: {len(labels) - sum(labels)}")
print()

# Tokenize documents - convert words to integer IDs
print("=" * 70)
print("STEP 2: Tokenize Documents (Words → Integer IDs)")
print("=" * 70)
print("  Tokenizer learns which words exist and assigns each a unique ID")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(documents)
vocab_size = len(tokenizer.word_index) + 1  # +1 for padding
sequences = tokenizer.texts_to_sequences(documents)

print(f"  Vocabulary size: {vocab_size}")
print(f"  Sample tokenization:")
print(f"    Text: '{documents[0]}'")
print(f"    IDs:  {sequences[0]}")
print(f"  Word index sample:")
word_dict = list(tokenizer.word_index.items())[:5]
for word, idx in word_dict:
    print(f"    '{word}' → ID {idx}")
print()

# Pad sequences - make all sequences the same length
print("=" * 70)
print("STEP 3: Pad Sequences (Make Equal Length)")
print("=" * 70)
print("  Neural networks need fixed-size inputs")
print("  Padding adds zeros to shorter sequences")

max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

print(f"  Max sequence length: {max_length}")
print(f"  Padded shape: {padded_sequences.shape}")
print(f"  Example (Document 1 after padding):")
print(f"    Original IDs: {sequences[0]}")
print(f"    Padded IDs:   {padded_sequences[0]}")
print()

# ===== STEP 4: Convert to PyTorch Tensors =====
print("=" * 70)
print("STEP 4: Convert to PyTorch Tensors")
print("=" * 70)
print("  PyTorch works with tensors (multi-dimensional arrays)")

X = torch.LongTensor(padded_sequences)  # Input sequences
y = torch.FloatTensor(labels).unsqueeze(1)  # Labels (reshape to column)

print(f"  Input tensor shape: {X.shape} (samples, sequence_length)")
print(f"  Label tensor shape: {y.shape} (samples, 1)")
print()

# Create a simple DataLoader (optional, but good practice)
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

print(f"  DataLoader created with batch size: 2")
print()

# ===== STEP 5: Define LSTM Model =====
print("=" * 70)
print("STEP 5: Define LSTM Model Architecture")
print("=" * 70)

class LSTMClassifier(nn.Module):
    """LSTM-based sentiment classifier for binary classification."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        """
        Initialize the model layers.
        
        Args:
            vocab_size: Number of unique words in vocabulary
            embedding_dim: Dimension of word embeddings (50 = 50-dimensional vectors)
            hidden_dim: Hidden size of LSTM (64 = LSTM maintains 64-dimensional state)
        """
        super(LSTMClassifier, self).__init__()
        
        # Layer 1: Embedding - converts word IDs to dense vectors
        # (learns to represent similar words with similar vectors)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Layer 2: LSTM - captures sequential patterns
        # Input: 50 (embedding dimension), Output: 64 (hidden dimension)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        # Layer 3: Fully Connected - makes final classification
        # Input: 64 (last LSTM hidden state), Output: 1 (binary classification)
        self.fc = nn.Linear(hidden_dim, 1)
        
        # Layer 4: Sigmoid - outputs probability between 0 and 1
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
        
        Returns:
            Probability predictions of shape (batch_size, 1)
        """
        # Embedding: (batch_size, seq_len) → (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)
        
        # LSTM: processes sequence and returns (output, (hidden, cell))
        # We only use the last hidden state (captures summary of entire sequence)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use last hidden state: (batch_size, hidden_dim)
        last_hidden = hidden.squeeze(0)
        
        # Fully connected layer: (batch_size, hidden_dim) → (batch_size, 1)
        logits = self.fc(last_hidden)
        
        # Sigmoid: (batch_size, 1) → (batch_size, 1) with values in [0, 1]
        output = self.sigmoid(logits)
        
        return output


# Instantiate model
embedding_dim = 50   # Word vector size
hidden_dim = 64      # LSTM hidden state size
model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim)

print(f"  Model Architecture:")
print(f"    Embedding: vocab_size={vocab_size} → embedding_dim={embedding_dim}")
print(f"    LSTM: embedding_dim={embedding_dim} → hidden_dim={hidden_dim}")
print(f"    Dense: hidden_dim={hidden_dim} → output=1 (binary classification)")
print(f"    Activation: Sigmoid (output between 0 and 1)")
print()

# ===== STEP 6: Set Up Training =====
print("=" * 70)
print("STEP 6: Set Up Training (Loss Function & Optimizer)")
print("=" * 70)

# Loss function for binary classification
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss

# Optimizer - adjusts model weights to minimize loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"  Loss Function: Binary Cross-Entropy (BCE)")
print(f"  Optimizer: Adam (adaptive learning rate)")
print(f"  Learning Rate: 0.001")
print()

# ===== STEP 7: Train Model =====
print("=" * 70)
print("STEP 7: Train the Model")
print("=" * 70)

epochs = 10
print(f"  Training for {epochs} epochs...\n")

for epoch in range(epochs):
    total_loss = 0
    batch_count = 0
    
    for batch_X, batch_y in dataloader:
        # Forward pass: compute predictions
        predictions = model(batch_X)
        
        # Compute loss
        loss = criterion(predictions, batch_y)
        
        # Backward pass: compute gradients
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()        # Compute gradients
        
        # Update weights
        optimizer.step()       # Adjust weights
        
        total_loss += loss.item()
        batch_count += 1
    
    avg_loss = total_loss / batch_count
    print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

print()

# ===== STEP 8: Make Predictions =====
print("=" * 70)
print("STEP 8: Make Predictions on Training Data")
print("=" * 70)

model.eval()  # Set model to evaluation mode (disables dropout, etc.)
with torch.no_grad():  # Don't compute gradients for inference
    predictions = model(X)

print("  Predictions (probability of positive sentiment):\n")
for i, (doc, true_label, pred) in enumerate(zip(documents, labels, predictions)):
    predicted_class = 1 if pred.item() > 0.5 else 0
    correct = "✓" if predicted_class == true_label else "✗"
    print(f"  {correct} Doc {i+1}: '{doc}'")
    print(f"      True: {true_label}, Predicted: {pred.item():.3f} → Class {predicted_class}\n")

print("\nModel training complete!")
