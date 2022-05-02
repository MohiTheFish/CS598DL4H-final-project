import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from util import xToTensor, yToTensor

def init_data_coam(params: dict):
    diagnoses = np.array(pickle.load(open(params["diagnoses_file"], 'rb')), dtype='object')
    prescriptions = np.array(pickle.load(open(params["prescriptions_file"], 'rb')), dtype='object')
    labels = np.array(pickle.load(open(params["labels_file"], 'rb')), dtype='object')

    dataset = CustomDataset(diagnoses, prescriptions, labels)
    data_size = len(dataset)
    # A list of indices, randomly ordered
    ind = np.random.permutation(data_size)

    test_size = int(params["test_ratio"] * data_size)
    validation_size = int(params["validation_ratio"] * data_size)

    # Split the indices into training/valid/test split
    test_indices = ind[:test_size]
    valid_indices = ind[test_size:test_size + validation_size]
    train_indices = ind[test_size + validation_size:]

    # Retrieve data from those indices
    train_set_x, train_set_y = dataset[train_indices]
    test_set_x, test_set_y = dataset[test_indices]
    valid_set_x, valid_set_y = dataset[valid_indices]

    # Order the data by number of visit
    def len_argsort(data):
        return sorted(range(len(data)), key=lambda x: len(data[x]))

    # We can use the number of visits from diagnoses for prescriptions because both shapes are identical
    def order_set(set_x, set_y, indices):
        sorted_index = len_argsort(diagnoses[indices])
        set_x = [(set_x[0][i], set_x[1][i]) for i in sorted_index]
        set_y = [set_y[i] for i in sorted_index]
        return set_x, set_y

    train_set_x, train_set_y = order_set(train_set_x, train_set_y, train_indices)
    test_set_x, test_set_y = order_set(test_set_x, test_set_y, test_indices)
    valid_set_x, valid_set_y = order_set(valid_set_x, valid_set_y, valid_indices)

    return train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y


class CustomDataset(Dataset):
    def __init__(self, diagnoses, prescriptions, labels):
        self.diagnoses = diagnoses
        self.prescriptions = prescriptions
        self.y = labels
    
    def __len__(self):
        return len(self.diagnoses)
    
    def __getitem__(self, index):
        return (self.diagnoses[index], self.prescriptions[index]), self.y[index]

def _separateData(data):
    xDiagnoses = [i[0] for i in data]
    xPrescriptions = [i[1] for i in data]
    return xDiagnoses, xPrescriptions

def _toTensor(xDiagnoses, xPrescriptions, set_y, params):
    xDiagnoses = xToTensor(xDiagnoses, params['num_diagnoses_codes'], params)
    xPrescriptions = xToTensor(xPrescriptions, params['num_prescription_codes'], params)
    y = yToTensor(set_y, params)
    return xDiagnoses, xPrescriptions, y

def load_data_coam(set_x, set_y, params):
    xDiagnoses, xPrescriptions = _separateData(set_x)
    return _toTensor(xDiagnoses, xPrescriptions, set_y, params)


def init_CoamNN(self, params):
    """
    num_embeddings(int): size of the dictionary of embeddings
    embedding_dim(int) the size of each embedding vector
    """
    self.diagnoses_hidden_size = 128
    self.prescriptions_hidden_size = 128
    self.num_output_classes = 2
    self.embedding_size = 256

    self.emb_layer_diagnoses = nn.Linear(in_features=params["num_diagnoses_codes"], out_features=self.embedding_size)
    self.emb_layer_prescriptions = nn.Linear(in_features=params["num_prescription_codes"], out_features=self.embedding_size)
    
    self.dropout = nn.Dropout(params["dropout_p"])

    self.diagnoses_rnn = nn.GRU(input_size=self.embedding_size, hidden_size=self.diagnoses_hidden_size, bidirectional=True)
    self.prescriptions_rnn = nn.GRU(input_size=self.embedding_size,  hidden_size=self.prescriptions_hidden_size, bidirectional=True)

    self.diagnoses_attention = nn.Linear(in_features=self.embedding_size, out_features=self.embedding_size )
    self.treatment_attention = nn.Linear(in_features=self.embedding_size, out_features=self.embedding_size )

    self.concatenation = nn.Linear(in_features=self.embedding_size, out_features=self.embedding_size//2)

    self.prediction = nn.Linear(in_features=self.embedding_size//2, out_features=self.num_output_classes)


class CoamNN(nn.Module):
    def __init__(self, params: dict):
        super(CoamNN, self).__init__()
        init_CoamNN(self, params)


    def forward(self, diagnoses, prescriptions):
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
        d = F.relu(d)
        p = F.relu(p)

        d = self.dropout(d)
        p = self.dropout(p)

        h, _ = self.diagnoses_rnn(d)
        g, _ = self.prescriptions_rnn(p)
        # print(d_rnn_output.shape, p_rnn_output.shape)

        # Combining operation is unclear
        # com = torch.cat((d, p), 2).permute((2, 0, 1))
        com = h + g
        # print(com.shape)

        alpha = self.diagnoses_attention(com)
        beta = self.treatment_attention(com)
        # print(alpha.shape, beta.shape)

        alpha_t = F.softmax(alpha, dim=2)
        beta_t = F.softmax(beta, dim=2)

        # print(alpha_t.shape, beta_t.shape)

        h_t = torch.sum(beta_t*h, dim=0)
        g_t = torch.sum(alpha_t*g, dim=0)

        # print(h_t.shape, g_t.shape)

        p = self.concatenation(h_t + g_t)
        p = torch.tanh(p)

        # print(p.shape)
        output = self.prediction(p)
        # print(output.shape)
        output = F.softmax(output, dim=1)

        return output

    def default_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=.0001)
    def default_loss_fn(self):
        return torch.nn.CrossEntropyLoss()

class CoamAlphaNN(nn.Module):
    def __init__(self, params: dict):
        super(CoamAlphaNN, self).__init__()
        init_CoamNN(self, params)


    def forward(self, diagnoses, prescriptions):
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
        d = F.relu(d)
        p = F.relu(p)

        d = self.dropout(d)
        p = self.dropout(p)

        h, _ = self.diagnoses_rnn(d)
        g, _ = self.prescriptions_rnn(p)
        # print(d_rnn_output.shape, p_rnn_output.shape)

        # Combining operation is unclear
        # com = torch.cat((d, p), 2).permute((2, 0, 1))
        # com = h + g
        # print(com.shape)


        alpha = self.diagnoses_attention(h)
        beta = self.treatment_attention(g)
        # print(alpha.shape, beta.shape)

        alpha_t = F.softmax(alpha, dim=2)
        beta_t = F.softmax(beta, dim=2)

        # print(alpha_t.shape, beta_t.shape)

        h_t = torch.sum(beta_t*h, dim=0)
        g_t = torch.sum(alpha_t*g, dim=0)

        # print(h_t.shape, g_t.shape)

        p = self.concatenation(h_t + g_t)
        p = torch.tanh(p)

        # print(p.shape)
        output = self.prediction(p)
        # print(output.shape)
        output = F.softmax(output, dim=1)

        return output

    def default_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=.0001)
    def default_loss_fn(self):
        return torch.nn.CrossEntropyLoss()


class CoamBetaNN(nn.Module):
    def __init__(self, params: dict):
        super(CoamBetaNN, self).__init__()
        
        
        self.diagnoses_hidden_size = 128
        self.prescriptions_hidden_size = 128
        self.num_output_classes = 2
        self.embedding_size = 256

        self.emb_layer_diagnoses = nn.Linear(in_features=params["num_diagnoses_codes"], out_features=self.embedding_size)
        self.emb_layer_prescriptions = nn.Linear(in_features=params["num_prescription_codes"], out_features=self.embedding_size)
        
        self.dropout = nn.Dropout(params["dropout_p"])

        self.diagnoses_rnn = nn.GRU(input_size=self.embedding_size, hidden_size=self.diagnoses_hidden_size, bidirectional=True)
        self.prescriptions_rnn = nn.GRU(input_size=self.embedding_size,  hidden_size=self.prescriptions_hidden_size, bidirectional=True)

        self.attention = nn.Linear(in_features=self.embedding_size, out_features=self.embedding_size )

        self.concatenation = nn.Linear(in_features=self.embedding_size, out_features=self.embedding_size//2)

        self.prediction = nn.Linear(in_features=self.embedding_size//2, out_features=self.num_output_classes)


    def forward(self, diagnoses, prescriptions):
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
        d = F.relu(d)
        p = F.relu(p)

        d = self.dropout(d)
        p = self.dropout(p)

        h, _ = self.diagnoses_rnn(d)
        g, _ = self.prescriptions_rnn(p)
        # print(d_rnn_output.shape, p_rnn_output.shape)

        # Combining operation is unclear
        # com = torch.cat((d, p), 2).permute((2, 0, 1))
        com = h + g
        # print(com.shape)


        alpha = self.attention(com)
        beta = self.attention(com)
        # print(alpha.shape, beta.shape)

        alpha_t = F.softmax(alpha, dim=2)
        beta_t = F.softmax(beta, dim=2)

        # print(alpha_t.shape, beta_t.shape)

        h_t = torch.sum(beta_t*h, dim=0)
        g_t = torch.sum(alpha_t*g, dim=0)

        # print(h_t.shape, g_t.shape)

        p = self.concatenation(h_t + g_t)
        p = torch.tanh(p)

        # print(p.shape)
        output = self.prediction(p)
        # print(output.shape)
        output = F.softmax(output, dim=1)

        return output

    def default_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=.0001)
    def default_loss_fn(self):
        return torch.nn.CrossEntropyLoss()