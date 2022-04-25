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
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

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

def separateData(data):
    xDiagnoses = [i[0] for i in data]
    xPrescriptions = [i[1] for i in data]
    return xDiagnoses, xPrescriptions

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
        self.emb_layer_diagnoses = nn.Linear(in_features=params["num_diagnoses_codes"], out_features=256)
        self.emb_layer_prescriptions = nn.Linear(in_features=params["num_prescription_codes"], out_features=256)
        
        self.dropout = nn.Dropout(params["dropout_p"])

        self.diagnoses_hidden_size = 128
        self.prescriptions_hidden_size = 128

        self.diagnoses_rnn = nn.GRU(input_size=256, hidden_size=self.diagnoses_hidden_size, bidirectional=True)
        self.prescriptions_rnn = nn.GRU(input_size=256,  hidden_size=self.prescriptions_hidden_size, bidirectional=True)

        self.diagnoses_attention = nn.Linear(in_features=2*params["embedding_dim"], out_features=2*params["embedding_dim"] )
        self.treatment_attention = nn.Linear(in_features=2*params["embedding_dim"], out_features=2*params["embedding_dim"] )


        self.concatenation = nn.Linear(in_features=256, out_features=128)


        self.prediction = nn.Linear(in_features=128, out_features=params["num_class"])
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
        # print(diagnoses.shape, prescriptions.shape)
        d = self.emb_layer_diagnoses(diagnoses)
        p = self.emb_layer_prescriptions(prescriptions)
        # d = F.relu(d)
        # p = F.relu(p)

        d = self.dropout(d)
        p = self.dropout(p)

        d_rnn_output, d_rnn_hidden = self.diagnoses_rnn(d, d_rnn_hidden)
        p_rnn_output, p_rnn_hidden = self.prescriptions_rnn(p, p_rnn_hidden)
        # print(d_rnn_output.shape, p_rnn_output.shape)

        # Combining operation is unclear
        # com = torch.cat((d, p), 2).permute((2, 0, 1))
        com = d_rnn_output + p_rnn_output # torch.sum((d_rnn_output, p_rnn_output), dim=0)
        # print(com.shape)


        alpha = self.diagnoses_attention(com)
        beta = self.treatment_attention(com)
        # print(alpha.shape, beta.shape)

        alpha_t = F.softmax(alpha, dim=2)
        beta_t = F.softmax(beta, dim=2)

        # print(alpha_t.shape, beta_t.shape)

        # print(beta_t.shape, d_rnn_hidden.shape)
        h_t = torch.sum(beta_t*d_rnn_output, dim=0)
        g_t = torch.sum(alpha_t*p_rnn_output, dim=0)

        # print(h_t.shape, g_t.shape)

        p = self.concatenation(h_t + g_t)
        p = torch.tanh(p)

        # print(p.shape)
        output = self.prediction(p)
        # print(output)
        output = F.softmax(output, dim=1)
        # print(output)

        return output

    def init_hidden(self, input_shape):
        current_padding_len = 2 #input_shape[0]
        current_batch_size = input_shape[1]
        return torch.zeros(current_padding_len, current_batch_size, self.diagnoses_hidden_size).to(device), torch.zeros(current_padding_len, current_batch_size, self.prescriptions_hidden_size).to(device)


def init_params(params: dict):
    # embedding matrix
    params["num_diagnoses_codes"] = 942
    params["num_prescription_codes"] = 3271
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
    params["n_epoches"] = 50

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
    diagnoses = np.array(pickle.load(open(params["diagnoses_file"], 'rb')), dtype='object')
    labels = np.array(pickle.load(open(params["label_file"], 'rb')), dtype='object')
    prescriptions = np.array(pickle.load(open(params["prescriptions_file"], 'rb')), dtype='object')

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

def evalModel(model, set_x, set_y):
    xDiagnoses, xPrescriptions = separateData(set_x)
    xDiagnoses = xToTensor(xDiagnoses, parameters['num_diagnoses_codes'])
    xPrescriptions = xToTensor(xPrescriptions, parameters['num_prescription_codes'])
    y_true = yToTensor(set_y)
    diagnoses_rnn_hidden_init, prescriptions_rnn_hidden_init = model.init_hidden(xDiagnoses.shape)
    pred = model(xDiagnoses, xPrescriptions, diagnoses_rnn_hidden_init, prescriptions_rnn_hidden_init)
    y_true = y_true.unsqueeze(1)
    y_true_oh = torch.zeros(pred.shape).to(device).scatter_(1, y_true, 1)

    y_true = y_true_oh.detach().cpu().numpy()
    y_pred = pred.detach().cpu().numpy()
    # print(y_pred)
    # print(y_true)
    auc = roc_auc_score(y_true=y_true, y_score=y_pred)
    
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    y_pred = y_pred.astype(np.int32)
    y_true = y_true.astype(np.int32)
    p = precision_score(y_true, y_pred, average='macro')
    r = recall_score(y_true, y_pred, average='macro')
    f = f1_score(y_true, y_pred, average='macro')

    return p, r, f

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
    # for name, parm in model.named_parameters():
    #   print(name, parm)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
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

            xDiagnoses, xPrescriptions = separateData(train_set_x[lIndex:rIndex])
            y = train_set_y[lIndex:rIndex]

            xDiagnoses = xToTensor(xDiagnoses, parameters['num_diagnoses_codes'])
            xPrescriptions = xToTensor(xPrescriptions, parameters['num_prescription_codes'])
            y = yToTensor(y)
            
            # print('xDiagnoss.shape:', xDiagnoses.shape)
            diagnoses_rnn_hidden_init, prescriptions_rnn_hidden_init = model.init_hidden(xDiagnoses.shape)

            pred = model(xDiagnoses, xPrescriptions, diagnoses_rnn_hidden_init, prescriptions_rnn_hidden_init)
            pred = pred.squeeze(1)

            # print(pred.shape, y.shape)
            loss = loss_fn(pred, y)
            loss.backward()
            loss_vector[index] = loss
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        # evalModel(model, valid_set_x)

        # p, r, f = evalModel(model, train_set_x, train_set_y)
        p, r, f = evalModel(model, test_set_x, test_set_y)

        # if test_auc > best_test_auc:
        #     best_test_auc = test_auc
        #     best_epoch = epoch

        print("{},{:.4f} \t {:.4f},{:.4f},{:.4f}".format(epoch, torch.mean(loss_vector),p,r,f))

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
