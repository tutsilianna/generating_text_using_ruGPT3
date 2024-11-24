import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pre_processing_function import one_hot_encode

class CharRNN(nn.Module):

    def __init__(self, tokens, n_steps=100, n_hidden=256, n_layers=2,
                               drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        # creating character dictionaries
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        ## TODO: define the LSTM, self.lstm
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers, dropout=drop_prob, batch_first=True)
        # input is a one_hot vector over the unique 83 characters (len(chars))

        ## TODO: define a dropout layer, self.dropout
        self.dropout = nn.Dropout(drop_prob)

        ## TODO: define the final, fully-connected output layer, self.fc
        self.fc = nn.Linear(n_hidden, len(self.chars))

        # initialize the weights
        self.init_weights()


    def forward(self, x, hc):
        ''' Forward pass through the network.
            These inputs are x, and the hidden/cell state `hc`. '''

        ## TODO: Get x, and the new hidden state (h, c) from the lstm
        lstm_out, (h, c) = self.lstm(x, hc)

        # stack up LSTM outputs
        x = lstm_out.contiguous()

        ## TODO: pass x through a droupout layer
        x = self.dropout(x)

        x = x.view(-1, self.n_hidden)

        # Stack up LSTM outputs using view
        #x = x.view(x.size()[0]*x.size()[1], self.n_hidden)

        ## TODO: put x through the fully-connected layer
        x = self.fc(x)

        # return x and the hidden state (h, c)
        return x, (h, c)


    def predict(self, char, h=None, cuda=False, top_k=None):
        ''' Given a character, predict the next character.

            Returns the predicted character and the hidden state.
        '''
        if cuda:
            self.cuda()
        else:
            self.cpu()

        if h is None:
            h = self.init_hidden(1)

        x = np.array([[self.char2int[char]]])
        x = one_hot_encode(x, len(self.chars))
        inputs = torch.from_numpy(x)
        if cuda:
            inputs = inputs.cuda()

        h = tuple([each.data for each in h])
        out, h = self.forward(inputs, h)

        p = F.softmax(out, dim=1).data
        if cuda:
            p = p.cpu()

        if top_k is None:
            top_ch = np.arange(len(self.chars))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()

        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p/p.sum())

        return self.int2char[char], h

    def init_weights(self):
        ''' Initialize weights for fully connected layer '''
        initrange = 0.1

        # Set bias tensor to all zeros
        self.fc.bias.data.fill_(0)
        # FC weights as random uniform
        self.fc.weight.data.uniform_(-1, 1)

    def init_hidden(self, n_seqs):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x n_seqs x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        return (weight.new(self.n_layers, n_seqs, self.n_hidden).zero_(),
                weight.new(self.n_layers, n_seqs, self.n_hidden).zero_())
