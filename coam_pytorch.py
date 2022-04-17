# Credit to: https://github.com/easyfan327/Pytorch-RETAIN
# -*- coding: utf-8 -*-
from unicodedata import bidirectional
import torch
import torch.nn as nn
import sys
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import pickle as pickle
import random
from sklearn.metrics import roc_auc_score

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    
    def __init__(self, diagnoses, prescriptions, labels):
        self.diagnoses = diagnoses
        self.prescriptions = prescriptions
        self.y = labels
    
    def __len__(self):
        
        """
        TODO: Return the number of samples (i.e. patients).
        """
        return len(self.diagnoses)
    
    def __getitem__(self, index):
        
        """
        TODO: Generates one sample of data.
        
        Note that you DO NOT need to covert them to tensor as we will do this later.
        """
        return (self.diagnoses[index], self.prescriptions[index]), self.y[index]


def yToTensor(y):
    y = torch.from_numpy(np.array(y)).long().to(device)

    return y

def xToTensor(x, embedding_dim):
    x = padMatrixWithoutTime(x, embedding_dim)
    x = torch.from_numpy(x).float().to(device)
    return x

class RetainNN(nn.Module):
    def __init__(self, params: dict):
        super(RetainNN, self).__init__()
        """
        num_embeddings(int): size of the dictionary of embeddings
        embedding_dim(int) the size of each embedding vector
        """
        #self.emb_layer = nn.Embedding(num_embeddings=params["num_embeddings"], embedding_dim=params["embedding_dim"])
        self.emb_layer_diagnoses = nn.Linear(in_features=params["num_embeddings_diagnoses"], out_features=params["embedding_dim"])
        self.emb_layer_prescriptions = nn.Linear(in_features=params["num_embeddings_prescriptions"], out_features=params["embedding_dim"])
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)
        
        self.dropout = nn.Dropout(params["dropout_p"])

        self.diagnoses_rnn = nn.GRU(params["embedding_dim"], params["embedding_dim"], bidirectional=True)
        self.prescriptions_rnn = nn.GRU(params["embedding_dim"], params["embedding_dim"], bidirectional=True)

        self.diagnoses_attention = nn.Linear(in_features=2*params["embedding_dim"], out_features=params["embedding_dim"] )
        self.treatment_attention = nn.Linear(in_features=2*params["embedding_dim"], out_features=params["embedding_dim"] )


        self.concatenation = nn.Linear(in_features=params["embedding_dim"], out_features=params["embedding_dim"])
        self.tanh = nn.Tanh()


        self.prediction = nn.Linear(in_features=params["embedding_dim"], out_features=params["num_embeddings_diagnoses"]+params["num_embeddings_prescriptions"])
        self.sigmoid = nn.Sigmoid()
        # self.variable_level_rnn = nn.GRU(params["var_rnn_hidden_size"], params["var_rnn_output_size"])
        # self.visit_level_rnn = nn.GRU(params["visit_rnn_hidden_size"], params["visit_rnn_output_size"])
        # self.variable_level_attention = nn.Linear(params["var_rnn_output_size"], params["var_attn_output_size"])
        # self.visit_level_attention = nn.Linear(params["visit_rnn_output_size"], params["visit_attn_output_size"])
        # self.output_dropout = nn.Dropout(params["output_dropout_p"])
        # self.output_layer = nn.Linear(params["embedding_output_size"], params["num_class"])

        self.var_hidden_size = params["embedding_dim"]
        self.visit_hidden_size = params["embedding_dim"]

        # self.n_samples = params["batch_size"]
        # self.reverse_rnn_feeding = params["reverse_rnn_feeding"]


    def forward(self, diagnoses, prescriptions, d_rnn_hidden, p_rnn_hidden):
        """
        :param input:
        :param var_rnn_hidden:
        :param visit_rnn_hidden:
        :return:
        """
        # emb_layer: input(*): LongTensor of arbitrary shape containing the indices to extract
        # emb_layer: output(*,H): where * is the input shape and H = embedding_dim
        print(diagnoses.shape, prescriptions.shape)
        d = self.emb_layer_diagnoses(diagnoses)
        p = self.emb_layer_prescriptions(prescriptions)
        d = self.relu(d)
        p = self.relu(p)

        d = self.dropout(d)
        p = self.dropout(p)

        # GRU:
        # input of shape (seq_len, batch, input_size)
        # seq_len: visit_seq_len
        # batch: batch_size
        # input_size: embedding dimension
        #
        # h_0 of shape (num_layers*num_directions, batch, hidden_size)
        # num_layers(1)*num_directions(1)
        # batch: batch_size
        # hidden_size:
        # if self.reverse_rnn_feeding:
        #     visit_rnn_output, visit_rnn_hidden = self.visit_level_rnn(torch.flip(v, [0]), visit_rnn_hidden)
        #     alpha = self.visit_level_attention(torch.flip(visit_rnn_output, [0]))
        # else:
        #     visit_rnn_output, visit_rnn_hidden = self.visit_level_rnn(v, visit_rnn_hidden)
        #     alpha = self.visit_level_attention(visit_rnn_output)
        # visit_attn_w = F.softmax(alpha, dim=0)

        

        # if self.reverse_rnn_feeding:
        #     var_rnn_output, var_rnn_hidden = self.variable_level_rnn(torch.flip(v, [0]), var_rnn_hidden)
        #     beta = self.variable_level_attention(torch.flip(var_rnn_output, [0]))
        # else:
        #     var_rnn_output, var_rnn_hidden = self.variable_level_rnn(v, var_rnn_hidden)
        #     beta = self.variable_level_attention(var_rnn_output)
        # var_attn_w = torch.tanh(beta)
        d_rnn_output, d_rnn_hidden = self.diagnoses_rnn(d, d_rnn_hidden)
        p_rnn_output, p_rnn_hidden = self.prescriptions_rnn(p, p_rnn_hidden)
        print(d_rnn_output.shape, p_rnn_output.shape)
        
        # d = torch.permute(d_rnn_output, (1, 2, 0))
        # p = torch.permute(p_rnn_output, (1, 2, 0))
        # print(p.shape, d.shape)


        # Combining operation is unclear
        # com = torch.cat((d, p), 2).permute((2, 0, 1))
        com = d_rnn_output + p_rnn_output # torch.sum((d_rnn_output, p_rnn_output), dim=0)
        print(com.shape)


        alpha = self.diagnoses_attention(com)
        beta = self.treatment_attention(com)
        print(alpha.shape, beta.shape)

        alpha_t = self.softmax(alpha)
        beta_t = self.softmax(beta)
        print(alpha_t.shape, beta_t.shape)

        h_t = beta_t*d_rnn_hidden
        g_t = alpha_t*p_rnn_hidden

        print(h_t.shape, g_t.shape)

        p = self.concatenation(h_t + g_t)
        p = self.tanh(p)

        output = self.prediction(p)
        output = self.softmax(output)



        # # print("beta attn:")
        # # print(var_attn_w.shape)
        # # '*' = hadamard product (element-wise product)
        # attn_w = visit_attn_w * var_attn_w
        # c = torch.sum(attn_w * v, dim=0)
        # # print("context:")
        # # print(c.shape)

        # c = self.output_dropout(c)
        # #print("context:")
        # #print(c.shape)
        # output = self.output_layer(c)
        # #print("output:")
        # #print(output.shape)
        # output = F.softmax(output, dim=1)
        # # print("output:")
        # # print(output.shape)

        return output

    def init_hidden(self, current_batch_size):
        return torch.zeros(2, current_batch_size, self.var_hidden_size).to(device), torch.zeros(2, current_batch_size, self.visit_hidden_size).to(device)


def init_params(params: dict):
    # embedding matrix
    params["num_embeddings_diagnoses"] = 942
    params["num_embeddings_prescriptions"] = 3271
    params["embedding_dim"] = 128
    # embedding dropout
    params["dropout_p"] = 0.5
    # Alpha
    params["visit_rnn_hidden_size"] = 128
    params["visit_rnn_output_size"] = 128
    params["visit_attn_output_size"] = 1
    # Beta
    params["var_rnn_hidden_size"] = 128
    params["var_rnn_output_size"] = 128
    params["var_attn_output_size"] = 128

    params["embedding_output_size"] = 128
    params["num_class"] = 2
    params["output_dropout_p"] = 0.8

    params["batch_size"] = 100
    params["n_epoches"] = 100

    params["test_ratio"] = 0.2
    params["validation_ratio"] = 0.1
    
    DATA_PATH = "data/processed_data/"
    params["diagnoses_file"] = DATA_PATH+"3digitICD9.seqs.pkl"
    params["label_file"] = DATA_PATH+"morts.pkl"
    params["prescriptions_file"] = DATA_PATH+"prescriptions.pkl"

    params["reverse_rnn_feeding"] = True

    # TODO: Customized Loss
    # TODO: REF: https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
    params["customized_loss"] = True

def padMatrixWithoutTime(data, num_embeddings):
    lengths = np.array([len(d) for d in data]).astype('int32')
    n_samples = len(data)
    maxlen = np.max(lengths)

    x = np.zeros((maxlen, n_samples, num_embeddings))
    for idx, d in enumerate(data):
        for xvec, subd in zip(x[:, idx, :], d):
            xvec[subd] = 1.

    return x


def init_data(params: dict):
    diagnoses = np.array(pickle.load(open(params["diagnoses_file"], 'rb')))
    labels = np.array(pickle.load(open(params["label_file"], 'rb')))
    prescriptions = np.array(pickle.load(open(params["prescriptions_file"], 'rb')))
    # xToTensor(diagnoses, params["diagnoses_embedding_dim"]) ################# Convert to tensor here?!?!?!??!

    dataset = CustomDataset(diagnoses, prescriptions, labels)
    data_size = len(dataset)
    ind = np.random.permutation(data_size)

    test_size = int(params["test_ratio"] * data_size)
    validation_size = int(params["validation_ratio"] * data_size)

    test_indices = ind[:test_size]
    valid_indices = ind[test_size:test_size + validation_size]
    train_indices = ind[test_size + validation_size:]

    train_set_x, train_set_y = dataset[train_indices]
    test_set_x, test_set_y = dataset[test_indices]
    valid_set_x, valid_set_y = dataset[valid_indices]

    def len_argsort(data):
        return sorted(range(len(data)), key=lambda x: len(data[x]))

    train_sorted_index = len_argsort(diagnoses[train_indices])
    train_set_x = [(train_set_x[0][i], train_set_x[1][i]) for i in train_sorted_index]
    train_set_y = [train_set_y[i] for i in train_sorted_index]

    valid_sorted_index = len_argsort(diagnoses[valid_indices])
    valid_set_x = [(valid_set_x[0][i], valid_set_x[1][i]) for i in valid_sorted_index]
    valid_set_y = [valid_set_y[i] for i in valid_sorted_index]

    test_sorted_index = len_argsort(diagnoses[test_indices])
    test_set_x = [(test_set_x[0][i], test_set_x[1][i]) for i in test_sorted_index]
    test_set_y = [test_set_y[i] for i in test_sorted_index]

    return train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(device)
    parameters = dict()
    init_params(parameters)

    train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y = init_data(parameters)

    model = RetainNN(params=parameters).to(device)
    tot = 0
    for p in model.parameters():
        if p.requires_grad:
            tot += p.numel()
    print('numParameters:', tot)
    exit(1)
    # for name, parm in model.named_parameters():
    #   print(name, parm)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-6, weight_decay=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    n_batches = int(np.ceil(float(len(train_set_y)) / float(parameters["batch_size"])))
    best_valid_auc = 0
    best_test_auc = 0
    best_epoch = 0
    print()
    print()
    for epoch in range(parameters["n_epoches"]):
        model.train()
        loss_vector = torch.zeros(n_batches, dtype=torch.float)
        for index in random.sample(range(n_batches), n_batches):
            lIndex = index*parameters["batch_size"]
            rIndex = (index+1)*parameters["batch_size"]

            x = train_set_x[lIndex:rIndex]
            xDiagnoses = [i[0] for i in x]
            xPrescriptions = [i[1] for i in x]
            y = train_set_y[lIndex:rIndex]

            xDiagnoses = xToTensor(xDiagnoses, parameters['num_embeddings_diagnoses'])
            xPrescriptions = xToTensor(xPrescriptions, parameters['num_embeddings_prescriptions'])
            y = yToTensor(y)
            
            print('xDiagnoss.shape:', xDiagnoses.shape)
            diagnoses_rnn_hidden_init, prescriptions_rnn_hidden_init = model.init_hidden(xDiagnoses.shape[1])

            pred = model(xDiagnoses, xPrescriptions, diagnoses_rnn_hidden_init, prescriptions_rnn_hidden_init)
            pred = pred.squeeze(1)
            # print("pred:")
            # print(pred.shape)
            # print(pred.data)
            # print("ybtensor:")
            # print(ybtensor.shape)

            print(pred.shape, y.shape)
            loss = loss_fn(pred, y)
            loss.backward()
            loss_vector[index] = loss
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        xDiagnoses, xPrescriptions = xToTensor(valid_set_x)
        y = yToTensor(valid_set_y)
        var_rnn_hidden_init, visit_rnn_hidden_init = model.init_hidden(x.shape[1])
        y_hat, var_rnn_hidden_init, visit_rnn_hidden_init = model(x, var_rnn_hidden_init, visit_rnn_hidden_init)
        y_true = y_true.unsqueeze(1)
        y_true_oh = torch.zeros(y_hat.shape).to(device).scatter_(1, y_true, 1)
        auc = roc_auc_score(y_true=y_true_oh.detach().cpu().numpy(), y_score=y_hat.detach().cpu().numpy())
            
        xDiagnoses, xPrescriptions = xToTensor(test_set_x)
        y = yToTensor(test_set_y)
        var_rnn_hidden_init, visit_rnn_hidden_init = model.init_hidden(x.shape[1])
        y_hat, var_rnn_hidden_init, visit_rnn_hidden_init = model(x, var_rnn_hidden_init, visit_rnn_hidden_init)
        y_true = y_true.unsqueeze(1)
        y_true_oh = torch.zeros(y_hat.shape).to(device).scatter_(1, y_true, 1)
        test_auc = roc_auc_score(y_true=y_true_oh.detach().cpu().numpy(), y_score=y_hat.detach().cpu().numpy())

        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_epoch = epoch

        print("{},{},{},{}".format(epoch, torch.mean(loss_vector), auc, test_auc))

    # print("best auc = {} at epoch {}".format(best_test_auc, best_epoch))
    """
    model.eval()
    x, x_length = padMatrixWithoutTime(seqs=test_set_x, options=parameters)
    x = torch.from_numpy(x).float().to(device)
    y_true = torch.from_numpy(np.array(test_set_y)).long().to(device)
    var_rnn_hidden_init, visit_rnn_hidden_init = model.init_hidden(x.shape[1])
    y_hat, var_rnn_hidden_init, visit_rnn_hidden_init = model(x, var_rnn_hidden_init, visit_rnn_hidden_init)
    y_true = y_true.unsqueeze(1)
    y_true_oh = torch.zeros(y_hat.shape).to(device).scatter_(1, y_true, 1)
    auc = roc_auc_score(y_true=y_true_oh.detach().cpu().numpy(), y_score=y_hat.detach().cpu().numpy())
    print("test auc:{}".format(auc))
    """
