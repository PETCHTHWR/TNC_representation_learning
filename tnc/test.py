import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        assert d_model % nhead == 0, "embed_dim must be divisible by num_heads"
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        encodings = self.encoder(x)
        return encodings

# Example usage:
d_model = 100  # Set the desired d_model
nhead = 5  # Set the desired num_heads (should be a factor of d_model)
dim_feedforward = 2048
dropout = 0.1
num_layers = 6

transformer_encoder = TransformerEncoder(d_model, nhead, num_layers, dim_feedforward, dropout)

# Input tensor with shape (batch_size, sequence_length, d_model)
input_tensor = torch.randn(40, 3, d_model)

# Output of the Transformer Encoder
output_tensor = transformer_encoder(input_tensor)

print(output_tensor.shape)  # Output: torch.Size([40, 3, 100])