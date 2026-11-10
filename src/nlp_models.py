"""
NLP Deep Learning Model Architectures for Text Identification.

DL Algorithms implemented:
1. Token Embedding: Projects sparse vocabulary indices into continuous d-dimensional vector space.
2. Bidirectional LSTM (BiLSTM): Recurrent Neural Network with gating mechanisms:
   - Forget Gate: Controls what information to discard from cell state.
   - Input Gate: Decides which new values to update in cell state.
   - Cell State Update: Long-term memory highway resolving vanishing gradients.
   - Output Gate: Computes hidden state output based on filtered cell state.
   - Bidirectionality: Processes sequence in both forward and backward directions to capture full context.
3. Sentence Representation: Concatenates final forward and backward recurrent hidden states [hn[-2], hn[-1]].
4. Classification Head: Projects final hidden representation to class logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMTextClassifier(nn.Module):
    """
    Bidirectional Long Short-Term Memory (BiLSTM) Deep Learning Text Classifier.
    """

    def __init__(
        self,
        vocab_size,
        embed_dim=64,
        hidden_dim=64,
        num_classes=4,
        num_layers=2,
        dropout_rate=0.3,
        pad_idx=0
    ):
        super(BiLSTMTextClassifier, self).__init__()

        # Word Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # Multi-layer Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )

        # Dense projection head
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # *2 for forward + backward states
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, text_indices, seq_lengths=None):
        """
        Args:
            text_indices: Tensor of token IDs [batch_size, max_seq_len]
        Returns:
            Logits of shape [batch_size, num_classes]
        """
        # [batch_size, seq_len] -> [batch_size, seq_len, embed_dim]
        embedded = self.embedding(text_indices)
        embedded = self.dropout(embedded)

        # LSTM output: lstm_out, (hn, cn)
        # hn shape: [num_layers * num_directions, batch_size, hidden_dim]
        lstm_out, (hn, cn) = self.lstm(embedded)

        # Concatenate final forward (hn[-2]) and backward (hn[-1]) hidden states
        final_hidden = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)

        # Fully connected layers with dropout
        x = self.dropout(final_hidden)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits
