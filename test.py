import torch
emb = torch.nn.Embedding(3, 5)
print(emb(torch.LongTensor([1])))