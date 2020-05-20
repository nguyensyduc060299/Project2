import torch.nn as nn
import torch
from torch.autograd import Variable
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def collect_data(datapath, type, tf_idf):
    lines = []
    with open(datapath,'r') as f:
        lines = f.read().splitlines()

    labels, docs = [], []
    for line in lines:
        labels.append(line.split('<fff>')[0])
        docs.append(line.split('<fff>')[-1])

    if type == 'train':
        docs = tf_idf.fit_transform(docs)
    else:
        docs = tf_idf.transform(docs)


    docs = docs.toarray()
    docs = docs.astype(np.float32)
    labels = np.array(labels, dtype=int)
    labels = torch.tensor(labels, dtype = torch.long)
    labels = Variable(labels)
    docs = Variable(torch.from_numpy(docs))
    return labels, docs

class Neural_Network(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Neural_Network,self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.reLu1 = nn.ReLU6()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.elu3 = nn.ELU()
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        output = self.fc1(X)
        output = self.reLu1(output)
        output = self.fc2(output)
        output = self.tanh2(output)
        output = self.fc3(output)
        output = self.elu3(output)
        output = self.fc4(output)
        return output

def runNN():
    tf_idf = TfidfVectorizer(min_df= 6)
    labels_train, docs_vector_train = collect_data('E:\\20192\Project 2\\20news-bydate\\result\\train_data.txt','train', tf_idf)
    labels_test, docs_vector_test = collect_data('E:\\20192\Project 2\\20news-bydate\\result\\test_data.txt', 'test', tf_idf)
    input_dim = docs_vector_train.shape[1]
    output_dim = 20
    hidden_dim = 50
    model_ANN = Neural_Network(input_dim, hidden_dim, output_dim)
    loss_fc = nn.CrossEntropyLoss()
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model_ANN.parameters(), learning_rate)
    for epoch in range(10000):
        output = model_ANN(docs_vector_train)
        optimizer.zero_grad()
        loss = loss_fc(output, labels_train)
        loss.backward()
        optimizer.step()

        if (epoch%100) == 0:
            correct = 0
            total = 0
            output = model_ANN(docs_vector_test)
            print(output)
            predict = torch.max(output.data, 1)[1]
            print(predict)
            total += len(labels_test)
            correct += (predict == labels_test).sum()
            accuracy = 100 * correct / float(total)
            print(accuracy)
            print(loss.data)

#--------------------------------------

runNN()

