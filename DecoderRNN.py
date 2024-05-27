import torch
import numpy as np
from torch import nn

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers, vocab_size):
        super(DecoderRNN, self).__init__()
        self.embed_size= embed_size
        self.hidden_size= hidden_size
        self.num_layers= num_layers
        self.vocab_size= vocab_size

        self.embeddings= nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm= nn.LSTM(self.embed_size,self.hidden_size, self.num_layers, batch_first= True)
        self.linear= nn.Linear(self.embed_size ,self.vocab_size)
        self.dropout= nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings= self.embeddings(captions)
        embeddings= torch.cat((features.unsqueeze(0), embeddings), dim=0)
        output, hidden= self.lstm(features)
        logits= self.linear(output)
        return logits


if __name__=="__main__":
    rnn= DecoderRNN(50, 256, 1, 2048)
    features= torch.randint(0, 256, (10, 256)).to(torch.float)
    captions= torch.randint(0, 2048, (10, 30))
    print(rnn(features, captions))

