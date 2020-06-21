# -*- coding: utf-8 -*-
"""RNN_LSTM_Model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KNjlxwkzYanovUBnQVJbM4oOJcBL07Ww
"""

import torch.nn as nn
import torch
from torch.autograd import Variable
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import time
from torch.utils.data import DataLoader

def collect_data(datapath, type, count_vector):
    lines = []
    with open(datapath,'r',errors='ignore') as f:
        lines = f.read().splitlines()

    labels, docs = [], []
    for line in lines:
        labels.append(line.split('<fff>')[0])
        docs.append(line.split('<fff>')[-1])

    if type == 'train':
        docs = count_vector.fit_transform(docs)
    else:
        docs = count_vector.transform(docs)


    docs = docs.toarray()
    # docs = docs.astype(np.float32)
    labels = np.array(labels, dtype=int)
    labels = torch.tensor(labels, dtype = torch.long)
    labels = Variable(labels)
    docs = Variable(torch.from_numpy(docs))
    return labels, docs

class LSTM(nn.Module):
    def __init__(self, vocab_size, e_dim, number_layers, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, e_dim)
        self.lstm = nn.LSTM(e_dim, hidden_size, number_layers, batch_first=True)
        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(hidden_size, output_size)
  
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        output = self.embeddings(x)
        output = self.dropout(output)
        lstm_output, (hidden, cell) = self.lstm(output)
        output = self.fc(hidden[-1])
        output = self.softmax(output)
        return output

count_vector = CountVectorizer(min_df = 6)
labels_train, docs_vector_train = collect_data('/content/train_data.txt','train', count_vector)
labels_test, docs_vector_test = collect_data('/content/test_data.txt', 'test', count_vector)

import sklearn
from sklearn.decomposition import TruncatedSVD

# md = TruncatedSVD(n_components=1000)
# md.fit(docs_vector_train)

# train_svd = md.transform(docs_vector_train)
# test_svd = md.transform(docs_vector_test)
# train_svd, test_svd = torch.from_numpy(train_svd), torch.from_numpy(test_svd)

batch_size = 64
vocal_size = docs_vector_train.shape[1]
train = torch.utils.data.TensorDataset(docs_vector_train, labels_train)
train_loader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=batch_size)
test = torch.utils.data.TensorDataset(docs_vector_test, labels_test)
test_loader = torch.utils.data.DataLoader(test, shuffle=True, batch_size=batch_size)

LSTMmodel = LSTM(vocal_size, 80, 1, 50, 20 )
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
LSTMmodel.to(device)

losss = nn.CrossEntropyLoss()
lr = 0.001
optimizer = torch.optim.Adam(LSTMmodel.parameters(), lr = lr)

epochs = 50
lenght = docs_vector_train.shape[0]
count = 0
torch.cuda.empty_cache()
for epoch in range(epochs):
  t1 = time.time()
  correct = 0
  for(docs, labels) in train_loader:
    train = Variable(docs).to(device)
    labels = Variable(labels).to(device)

    optimizer.zero_grad()

    output = LSTMmodel(train)

    loss = losss(output, labels)

    loss.backward()

    optimizer.step()

    count += 1
    # print(output)
    predict = torch.argmax(output.data, axis= 1)
    # print(predict)
    correct += (predict == labels).float().sum()
  t2 = time.time()
  print("Epoch: {}/{}, Accuracy: {:.6f}, Loss: {} Time: {}".format(epoch+1, epochs, correct/lenght, str(t2-t1), loss))

correct = 0
total = 0
for(docs, labels) in test_loader:
    test = Variable(docs).to(device)
    labels = Variable(labels).to(device)
    output = LSTMmodel(test)
    predict = torch.argmax(output.data, axis= 1)
    correct += (predict == labels).float().sum()
    total += len(labels)
accuracy = correct * 100. / total
print(accuracy)