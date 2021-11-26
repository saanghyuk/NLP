import torch
import torch.nn as nn


class CNNClassifier(nn.Module):

    def __init__(
        self,
        input_size,
        word_vec_size,
        n_classes,
        use_batch_norm=False,
        dropout_p=.5,
        
        # these two below are specific parameter to CNN layer
        window_sizes=[3, 4, 5],
        n_filters=[100, 100, 100],
    ):
        self.input_size = input_size  # input size : vocabulary size
        self.word_vec_size = word_vec_size
        self.n_classes = n_classes
        self.use_batch_norm = use_batch_norm
        self.dropout_p = dropout_p
        # window_size means that how many words a pattern covers.
        self.window_sizes = window_sizes
        # n_filters means that how many patterns to cover.
        self.n_filters = n_filters

        super().__init__()
        # input size : vocabulary size, word_vec_size : embedded layer size
        self.emb = nn.Embedding(input_size, word_vec_size)
        # Use nn.ModuleList to register each sub-modules.
        # if we just use list, insead of ModuleList, CNN layer will not be added as layer
        # parameters in these CNN layers will not be updated with back propa
        self.feature_extractors = nn.ModuleList()
        # window_sizes : [3, 4, 5]
        # n_filters : [100, 100, 100]
        for window_size, n_filter in zip(window_sizes, n_filters):
            self.feature_extractors.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=1, # We only use one embedding layer.
                        out_channels=n_filter,
                        kernel_size=(window_size, word_vec_size),
                        # output size = (sentence size - window_size + 1, 1)
                    ),
                    nn.ReLU(),
                    # (maybe) Dropout layer here will be nullify some outputs of the ReLu. 
                    nn.BatchNorm2d(n_filter) if use_batch_norm else nn.Dropout(dropout_p),
                )
            )

        # An input of generator layer is max values from each filter.
        # input size of this layer : (total number of filters, 1)
        self.generator = nn.Linear(sum(n_filters), n_classes)
        # We use LogSoftmax + NLLLoss instead of Softmax + CrossEntropy
        self.activation = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # |x| = (batch_size, length)
        x = self.emb(x)
        # |x| = (batch_size, length, word_vec_size)
        
        # What if the sentence length is shorter than the window_size? -> occur error 
        # in those cases, we will add padding to lengthen the sentence size
        min_length = max(self.window_sizes)
        
        # |x| = (batch_size, length, word_vector_size)
        if min_length > x.size(1):
            # Because some input does not long enough for maximum length of window size,
            # we add zero tensor for padding.
            pad = x.new(x.size(0), min_length - x.size(1), self.word_vec_size).zero_()
            # |pad| = (batch_size, min_length - length, word_vec_size)
            x = torch.cat([x, pad], dim=1)
            # |x| = (batch_size, min_length, word_vec_size)

        # In ordinary case of vision task, you may have 3 channels on tensor,
        # but in this case, you would have just 1 channel,
        # which is added by 'unsqueeze' method in below:
        x = x.unsqueeze(1) # just to put thiese x tensor to CNN layer
        # |x| = (batch_size, 1, length, word_vec_size)

        cnn_outs = []
        for block in self.feature_extractors:
            cnn_out = block(x)
            # |cnn_out| = (batch_size, n_filter, length - window_size + 1, 1)

            # In case of max pooling, we does not know the pooling size,
            # because it depends on the length of the sentence.
            # Therefore, we use instant function using 'nn.functional' package.
            # This is the beauty of PyTorch. :)
            
            # max pool from each filter
            cnn_out = nn.functional.max_pool1d(
                input=cnn_out.squeeze(-1),
                kernel_size=cnn_out.size(-2) # length - window_size + 1
            ).squeeze(-1)
            # before squeeze : |cnn_out| = (batch_size, n_filter, 1) 
            # |cnn_out| = (batch_size, n_filter)
            cnn_outs += [cnn_out]
        # Merge output tensors from each convolution layer.
        cnn_outs = torch.cat(cnn_outs, dim=-1)
        # |cnn_outs| = (batch_size, sum(n_filters))
        y = self.activation(self.generator(cnn_outs))
        # |y| = (batch_size, n_classes)

        return y
