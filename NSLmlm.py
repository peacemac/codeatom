import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Define the Custom Negative Sampling Layer
class NegativeSamplingLoss(nn.Module):
    def __init__(self, vocab_size, d_model, num_negative_samples=5):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_negative_samples = num_negative_samples
        # This layer acts as our dictionary's output weights
        self.output_embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, hidden_states, target_tokens):
        batch_size, seq_length, d_model = hidden_states.shape
        
        # --- A. Positive Loss (The True Words) ---
        # Get embeddings for the actual correct words
        true_word_embs = self.output_embeddings(target_tokens) # Shape: (B, S, D)
        
        # Calculate dot product between hidden state and true word embedding
        true_logits = torch.sum(hidden_states * true_word_embs, dim=-1)
        # We want the model to predict 1 (True) for these
        true_loss = F.binary_cross_entropy_with_logits(true_logits, torch.ones_like(true_logits))
        
        # --- B. Negative Loss (The Fake Words) ---
        # Randomly sample 'k' incorrect words
        negative_tokens = torch.randint(0, self.vocab_size, 
                                        (batch_size, seq_length, self.num_negative_samples), 
                                        device=hidden_states.device)
        
        # Get embeddings for the fake words
        negative_word_embs = self.output_embeddings(negative_tokens) # Shape: (B, S, K, D)
        
        # Expand hidden states to compute dot product with all K negative samples at once
        hidden_expanded = hidden_states.unsqueeze(2) # Shape: (B, S, 1, D)
        negative_logits = torch.sum(hidden_expanded * negative_word_embs, dim=-1) # Shape: (B, S, K)
        
        # We want the model to predict 0 (False) for these
        negative_loss = F.binary_cross_entropy_with_logits(negative_logits, torch.zeros_like(negative_logits))
        
        # Combine losses
        return true_loss + negative_loss

# 2. Setup Parameters & Dummy Data
vocab_size = 100000
d_model = 128
batch_size = 4
seq_length = 10
num_neg_samples = 5 # For every true word, evaluate 5 wrong words

# Simulate the hidden states from the Transformer
hidden_states = torch.randn(batch_size, seq_length, d_model)
# Simulate the correct target words
target_tokens = torch.randint(0, vocab_size, (batch_size, seq_length))

# 3. Execute Training Pass
loss_fn = NegativeSamplingLoss(vocab_size, d_model, num_neg_samples)

print("--- Negative Sampling Training Pass ---")
loss = loss_fn(hidden_states, target_tokens)
print(f"Total Combined Loss: {loss.item():.4f}")