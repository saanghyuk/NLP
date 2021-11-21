import torch
import torch.nn as nn


class SequenceClassifier(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        n_layers=4,
        dropout_p=.2,
    ):
        # input : 28짜리 row가 time stamp별로 28개
        self.input_size = input_size
        #size of hidden vector
        self.hidden_size = hidden_size
        
        # output size is used in the NN in the last layer 
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super().__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=True, 
        )
        
        
        self.layers = nn.Sequential(
            nn.ReLU(),
            # hidden size : normal & reverse direction
            nn.BatchNorm1d(hidden_size * 2),
            nn.Linear(hidden_size * 2, output_size),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        # |x| = (batch_size, h, w)
        
        # in this point
        # (batch size, Sequence length(Time Stamp), Input size )
        z, _ = self.rnn(x)
        # |z| = (batch_size, h, hidden_size * 2)
        z = z[:, -1]
        # |z| = (batch_size, hidden_size * 2)
        y = self.layers(z)
        # |y| = (batch_size, output_size)

        return y
