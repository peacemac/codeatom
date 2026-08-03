import torch
import torch.nn as nn

# 1. Setup Parameters
batch_size = 4
seq_length = 10
d_model = 128
vocab_size = 100000

# 2. Define the Cutoffs for the Clusters
# We must sort the vocabulary by frequency first. 
# Here, indices 0-5000 are in the "head" cluster (most frequent).
# Indices 5001-20000 are cluster 1, 20001-50000 are cluster 2, and the rest are cluster 3.
cutoffs = [5000, 20000, 50000]

# 3. Initialize Adaptive Softmax Layer
adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(
    in_features=d_model,
    n_classes=vocab_size,
    cutoffs=cutoffs,
    div_value=4.0 # Shrinks the embedding size of rare word clusters to save memory
)

# 4. Dummy Input Data
# Simulate the hidden states outputted by a Transformer encoder
hidden_states = torch.randn(batch_size, seq_length, d_model)
# Flatten the hidden states for the softmax layer: (batch_size * seq_length, d_model)
flat_hidden = hidden_states.view(-1, d_model)

# Simulate the target tokens (the actual correct words)
# Ensure targets are within our vocab size
target_tokens = torch.randint(0, vocab_size, (batch_size * seq_length,))

# 5. Training Pass (Calculates Loss)
print("--- Training Pass ---")
# When you provide targets, it returns the heavily optimized loss automatically
out = adaptive_softmax(flat_hidden, target_tokens)
print(f"Computed Loss: {out.loss.item():.4f}")

# 6. Inference Pass (Predicting words)
print("\n--- Inference Pass ---")
# During inference, we don't have targets. We use the predict() method.
with torch.no_grad():
    predictions = adaptive_softmax.predict(flat_hidden)
    
print(f"Predictions Shape: {predictions.shape}") # Should be (40,)
print(f"First 5 Predicted Indices: {predictions[:5].tolist()}")