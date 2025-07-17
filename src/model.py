import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, input_size=4096, hidden_size=1024, num_layers=2):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1, bidirectional=False)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)

        return (hidden, cell)


class Decoder(nn.Module):
    def __init__(self, input_size=4096, hidden_size=1024, output_size=4096, num_layers=2):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1, bidirectional=False)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        output, (hidden, cell) = self.lstm(x, hidden)
        prediction = self.fc(output)
        # prediction = output # ablation study

        return prediction, (hidden, cell)


class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, window_size: int = 1, **kwargs) -> None:
        super(LSTMAutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.window_size = window_size

        if "num_layers" in kwargs:
            num_layers = kwargs.pop("num_layers")
        else:
            num_layers = 1

        self.encoder = Encoder(
            input_size=input_dim,
            hidden_size=latent_dim,
            num_layers=num_layers,
        )

        self.reconstruct_decoder = Decoder(
            input_size=input_dim,
            output_size=input_dim,
            hidden_size=latent_dim,
            num_layers=num_layers,
        )

    def forward(self, src: torch.Tensor, **kwargs):
        batch_size, sequence_length, var_length = src.size()

        encoder_hidden = self.encoder(src)

        inv_idx = torch.arange(sequence_length - 1, -1, -1).long()
        reconstruct_output = []
        temp_input = torch.zeros((batch_size, 1, var_length), dtype=torch.float).to(src.device)
        hidden = encoder_hidden
        for t in range(sequence_length):
            temp_input, hidden = self.reconstruct_decoder(temp_input, hidden)
            reconstruct_output.append(temp_input)
        reconstruct_output = torch.cat(reconstruct_output, dim=1)[:, inv_idx, :]

        return [reconstruct_output, src]

    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        input = args[1]

        loss = F.mse_loss(recons, input)
        return loss