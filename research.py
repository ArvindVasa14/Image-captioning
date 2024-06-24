# import torch
# from torch import nn
# from DecoderRNN import DecoderRNN
# from EncoderCNN import EncoderCNN
import torch

# print(torch.cuda.is_available())

# t= torch.randint(1,100, (1,1000))
# print(t.unsqueeze())

# batch size, seq_len, embed_dim
# batch_size, 1, embed_dim (made for image)

# i am groot
# i am froot
#
# 1 2 3
# 1 2 4   2, 3
#
# 2, 3, 256
# 2, 256 > 2, 1, 256
#
# concat
# 2, 4, 256
# torch.Size([10, 256]) torch.Size([10, 30, 256])

# cnn= EncoderCNN(embed_size= 256)
#
# images= torch.randint(0,256, (32, 3, 224, 224)).to(torch.float)
#
# features= cnn(images)
#
# print(features.shape)

# rnn= DecoderRNN(256, 512, 1,  8915)
#
# features= torch.randint(0, 256, (32, 256)).to(torch.float)
# captions= torch.randint(0, 8915, (32, 30))
#
# logits= rnn(features, captions)
#
# print(logits.shape)

# final_out= torch.randint(0,8915, (32, 30, 8915)).to(torch.float)
# captions= torch.randint(0, 8915, (32, 30))
#
# final_out= final_out.permute(0, 2, 1)
#
# ce= nn.CrossEntropyLoss()
# loss= ce(final_out, captions)
#
# print(loss)

# decoder= DecoderRNN(256, 256, 1, 8915)
# gn=[]
# image= torch.randint(0, 256, (3, 224, 224))
# input= image.unsqueeze(0).to(torch.float)
# encoder= EncoderCNN(256)
# features= encoder(input)
# input= features
# print(input.shape)
# for i in range(20):
#     hidden, state= decoder.lstm(input)
#     outputs= decoder.fc(hidden.squeeze(1))
#     predicted= outputs.argmax(1)
#     gn.append(predicted.item())
#     input= decoder.embedding(predicted)
#     print(input.shape)
# print(input.shape)

# gn.append(input.argmax(1))

# print(gn)

# t1= torch.randint(1,100, (32, 256))
# t2= torch.randint(1,100, (32, 30, 256))
# print(t1.unsqueeze(1).shape, t2.shape)
# print(torch.cat((t1.unsqueeze(1), t2), dim=1))

# t= torch.randint(0, 2048, (64, 30))
# embed= nn.Embedding(2048, 256)
# embeddings= embed(t)
# lstm = nn.LSTM(256, 512, num_layers=1, batch_first=True, bidirectional=False)
# outputs, (hidden, cell)= lstm(embeddings)
# print(outputs.shape, hidden.shape, cell.shape)

# decoder= DecoderRNN(256, 512, 2, 2048, False)
# f= torch.randint(0,2000, (64, 256))
# c= torch.randint(0, 2048, (64, 30))
# fc= decoder(f, c)


t= torch.randint(0, 2048 , (2, 5, 3))
# print(t[:,1:2, :].shape)
d=[]
for i in t:
    d.append(i)

print(d)
print(torch.cat(d, dim=1))

# This code is contributed by Alok Khansali


