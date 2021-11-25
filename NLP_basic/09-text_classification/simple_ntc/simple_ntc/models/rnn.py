import torch.nn as nn


class RNNClassifier(nn.Module):

    def __init__(
        self,
        input_size, # the number of vaca
        word_vec_size, # size of the word vector(vector size after embeded)
        hidden_size, # in RNN, vector size of the hidden state/cell state
        n_classes, # the number of classes
        n_layers=4, # how many layers in the RNN
        dropout_p=.3, # Dropout in the LSTM layer
    ):
        self.input_size = input_size  # vocabulary_size
        self.word_vec_size = word_vec_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super().__init__()

        self.emb = nn.Embedding(input_size, word_vec_size)
        self.rnn = nn.LSTM(
            input_size=word_vec_size, # embeded vector
            hidden_size=hidden_size, # hidden vector of the LSTM
            num_layers=n_layers, # layers in the LSTM
            dropout=dropout_p,
            batch_first=True, # I will put batch first, just habitually(output tensor size)
            bidirectional=True,
        )
        # dimension reduction
        self.generator = nn.Linear(hidden_size * 2, n_classes)
        # We use LogSoftmax + NLLLoss instead of Softmax + CrossEntropy
        # 
        self.activation = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # |x| = (batch_size, length)
        x = self.emb(x)
        # |x| = (batch_size, length, word_vec_size)
        x, _ = self.rnn(x) 
        # |x| = (batch_size, length, hidden_size * 2)
        y = self.activation(self.generator(x[:, -1]))
        # | (x[:, -1]) | : (batch_size, hidden_size * 2) 
        
        # |y| = (batch_size, n_classes)

        return y
