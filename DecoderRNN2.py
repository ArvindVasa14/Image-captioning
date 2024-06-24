import torch
import numpy as np
from torch import nn
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token= 1

class DecoderRNN2(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, bidirectional):
        super(DecoderRNN2, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, num_layers=self.num_layers, batch_first=True,
                            bidirectional=self.bidirectional)
        self.linear = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.Linear(self.hidden_size * 2, self.vocab_size)
        )


    def forward(self, features, captions, teacher_forcing_ratio=0.5):
        batch_size= captions.size(0)
        print(features.size())
        # decoder_input = t1 = torch.empty(batch_size,1, 256, dtype=torch.long).fill_(SOS_token).to(device)
        caption_size = captions.size(1)
        batch_size = features.size(0)
        captions = self.embedding(captions)
        features = features.unsqueeze(0)
        # inputs = torch.cat((decoder_input, captions), dim=1).to(device)
        inputs= captions
        outputs = torch.zeros(batch_size, caption_size, self.vocab_size).to(device)

        hidden, cell = self.init_hidden(batch_size)

        for t in range(captions.size(1)):

            output, (hidden, cell) = self.lstm(inputs[:, t:t + 1, :], (features, cell))
            output = self.linear(output.squeeze(1))
            outputs[:, t, :] = output

            teacher_force = np.random.rand() < teacher_forcing_ratio
            top1 = output.argmax(1)
            next_input = captions[:, t, :].unsqueeze(1) if teacher_force else self.embedding(top1).unsqueeze(1)

            # Create a new tensor for inputs to avoid in-place modification
            if t + 1 < captions.size(1):
                inputs = torch.cat((inputs[:, :t + 1, :], next_input, inputs[:, t + 2:, :]), dim=1)

        return outputs

    def init_hidden(self, batch_size):
        direction_factor = 2 if self.bidirectional else 1
        hidden = torch.zeros(self.num_layers * direction_factor, batch_size, self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers * direction_factor, batch_size, self.hidden_size).to(device)
        return hidden.to(device), cell.to(device)

    def generate_caption(self, features, vocab, max_len=20):
        generated_captions = []
        batch_size= features.size(1)
        # image = torch.randint(0, 256, (3, 224, 224))
        with torch.no_grad():
            hidden = features.to(device).to(torch.float)
            cell= torch.zeros(features.size()).to(device).to(torch.float)
            inputs= torch.empty(batch_size, 1, dtype= torch.int).fill_(SOS_token).to(device)
            for i in range(max_len):
                embeddings= self.embedding(inputs)
                print(inputs.size(), hidden.size(), cell.size())
                hidden, state = self.lstm(inputs.to(torch.float), (hidden.to(torch.float), cell.to(torch.float)))
                outputs = self.linear(hidden.squeeze(1))
                predicted = outputs.argmax(1)
                generated_captions.append(predicted.item())
                input = self.embedding(predicted)
                if predicted == 2:
                    break

        return " ".join([vocab.get_itos()[idx] for idx in generated_captions])

if __name__=="__main__":
    dec = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=2048, num_layers=1, bidirectional=False).to(device)
    f = torch.randint(0, 256, (10, 512)).to(device).to(torch.float)
    c = torch.randint(0, 2048, (10, 30)).to(device)
    print(dec(f, c))
# class DecoderRNN(nn.Module):
#     def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
#         """
#         Args:
#             embed_size: final embedding size of the CNN encoder
#             hidden_size: hidden size of the LSTM
#             vocab_size: size of the vocabulary
#             num_layers: number of layers of the LSTM
#         """
#         super(DecoderRNN, self).__init__()
#
#         # Assigning hidden dimension
#         self.hidden_dim = hidden_size
#         # Map each word index to a dense word embedding tensor of embed_size
#         self.embed = nn.Embedding(vocab_size, embed_size)
#         # Creating LSTM layer
#         self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
#         # Initializing linear to apply at last of RNN layer for further prediction
#         self.linear = nn.Linear(hidden_size, vocab_size)
#         # Initializing values for hidden and cell state
#         self.hidden = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size))
#
#     def forward(self, features, captions):
#         """
#         Args:
#             features: features tensor. shape is (bs, embed_size)
#             captions: captions tensor. shape is (bs, cap_length)
#         Returns:
#             outputs: scores of the linear layer
#
#         """
#         # remove <end> token from captions and embed captions
#         cap_embedding = self.embed(
#             captions[:, :-1]
#         )  # (bs, cap_length) -> (bs, cap_length-1, embed_size)
#
#         # concatenate the images features to the first of caption embeddings.
#         # [bs, embed_size] => [bs, 1, embed_size] concat [bs, cap_length-1, embed_size]
#         # => [bs, cap_length, embed_size] add encoded image (features) as t=0
#         embeddings = torch.cat((features.unsqueeze(dim=1), cap_embedding), dim=1)
#
#         #  getting output i.e. score and hidden layer.
#         # first value: all the hidden states throughout the sequence. second value: the most recent hidden state
#         lstm_out, self.hidden = self.lstm(
#             embeddings
#         )  # (bs, cap_length, hidden_size), (1, bs, hidden_size)
#         outputs = self.linear(lstm_out)  # (bs, cap_length, vocab_size)
#
#         return outputs
#
#     def sample(self, inputs, states=None, max_len=20):
#         """
#         accepts pre-processed image tensor (inputs) and returns predicted
#         sentence (list of tensor ids of length max_len)
#         Args:
#             inputs: shape is (1, 1, embed_size)
#             states: initial hidden state of the LSTM
#             max_len: maximum length of the predicted sentence
#
#         Returns:
#             res: list of predicted words indices
#         """
#         res = []
#
#         # Now we feed the LSTM output and hidden states back into itself to get the caption
#         for i in range(max_len):
#             lstm_out, states = self.lstm(
#                 inputs, states
#             )  # lstm_out: (1, 1, hidden_size)
#             outputs = self.linear(lstm_out.squeeze(dim=1))  # outputs: (1, vocab_size)
#             _, predicted_idx = outputs.max(dim=1)  # predicted: (1, 1)
#             res.append(predicted_idx.item())
#             # if the predicted idx is the stop index, the loop stops
#             if predicted_idx == 1:
#                 break
#             inputs = self.embed(predicted_idx)  # inputs: (1, embed_size)
#             # prepare input for next iteration
#             inputs = inputs.unsqueeze(1)  # inputs: (1, 1, embed_size)
#
#         return res
