import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from util import xToTensor, yToTensor

def init_data_retain(params: dict):
    sequences = np.array(pickle.load(open(params["diagnoses_file"], 'rb')), dtype='object')
    labels = np.array(pickle.load(open(params["labels_file"], 'rb')), dtype='object')

    data_size = len(labels)
    ind = np.random.permutation(data_size)

    test_size = int(params["test_ratio"] * data_size)
    validation_size = int(params["validation_ratio"] * data_size)

    test_indices = ind[:test_size]
    valid_indices = ind[test_size:test_size + validation_size]
    train_indices = ind[test_size + validation_size:]

    train_set_x = sequences[train_indices]
    train_set_y = labels[train_indices]
    test_set_x = sequences[test_indices]
    test_set_y = labels[test_indices]
    valid_set_x = sequences[valid_indices]
    valid_set_y = labels[valid_indices]

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    train_sorted_index = len_argsort(train_set_x)
    train_set_x = [train_set_x[i] for i in train_sorted_index]
    train_set_y = [train_set_y[i] for i in train_sorted_index]

    valid_sorted_index = len_argsort(valid_set_x)
    valid_set_x = [valid_set_x[i] for i in valid_sorted_index]
    valid_set_y = [valid_set_y[i] for i in valid_sorted_index]

    test_sorted_index = len_argsort(test_set_x)
    test_set_x = [test_set_x[i] for i in test_sorted_index]
    test_set_y = [test_set_y[i] for i in test_sorted_index]

    return train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y

def _toTensor(set_x, set_y, params):
    x = xToTensor(set_x, params['num_diagnoses_codes'], params)
    y = yToTensor(set_y, params)
    return x, y

def load_data_retain(set_x, set_y, params):
    return _toTensor(set_x, set_y, params)

class RetainNN(nn.Module):
    def __init__(self, params: dict):
        super(RetainNN, self).__init__()
        """
        num_embeddings(int): size of the dictionary of embeddings
        embedding_dim(int) the size of each embedding vector
        """
        self.variable_rnn_hidden_size = 128
        self.visit_rnn_hidden_size = 128
        self.num_output_classes = 2
        self.embedding_size = 128
        #self.emb_layer = nn.Embedding(num_embeddings=params["num_embeddings"], embedding_dim=params["embedding_dim"])
        self.emb_layer = nn.Linear(in_features=params["num_diagnoses_codes"], out_features=self.embedding_size)
        self.dropout = nn.Dropout(params["dropout_p"])
        self.variable_level_rnn = nn.GRU(self.embedding_size, hidden_size=self.variable_rnn_hidden_size)
        self.visit_level_rnn = nn.GRU(input_size=self.embedding_size, hidden_size=self.visit_rnn_hidden_size)
        self.variable_level_attention = nn.Linear(self.embedding_size, self.embedding_size)
        self.visit_level_attention = nn.Linear(self.embedding_size, self.embedding_size)
        self.output_dropout = nn.Dropout(params["output_dropout_p"])
        self.output_layer = nn.Linear(self.embedding_size, self.num_output_classes)

        self.n_samples = params["batch_size"]
        self.reverse_rnn_feeding = params["reverse_rnn_feeding"]


    def forward(self, input):
        """
        :param input:
        :param var_rnn_hidden:
        :param visit_rnn_hidden:
        :return:
        """
        # emb_layer: input(*): LongTensor of arbitrary shape containing the indices to extract
        # emb_layer: output(*,H): where * is the input shape and H = embedding_dim
        # print("size of input:")
        # print(input.shape)
        v = self.emb_layer(input)
        # print("size of v:")
        # print(v.shape)
        v = self.dropout(v)

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
        if self.reverse_rnn_feeding:
            visit_rnn_output, _ = self.visit_level_rnn(torch.flip(v, [0]))
            alpha = self.visit_level_attention(torch.flip(visit_rnn_output, [0]))
        else:
            visit_rnn_output, _ = self.visit_level_rnn(v)
            alpha = self.visit_level_attention(visit_rnn_output)
        visit_attn_w = F.softmax(alpha, dim=0)

        if self.reverse_rnn_feeding:
            var_rnn_output, _ = self.variable_level_rnn(torch.flip(v, [0]))
            beta = self.variable_level_attention(torch.flip(var_rnn_output, [0]))
        else:
            var_rnn_output, _ = self.variable_level_rnn(v)
            beta = self.variable_level_attention(var_rnn_output)
        var_attn_w = torch.tanh(beta)

        # print("beta attn:")
        # print(var_attn_w.shape)
        # '*' = hadamard product (element-wise product)
        attn_w = visit_attn_w * var_attn_w
        c = torch.sum(attn_w * v, dim=0)
        # print("context:")
        # print(c.shape)

        c = self.output_dropout(c)
        # print("context:")
        # print(c.shape)
        output = self.output_layer(c)
        # print("output:")
        # print(output.shape)
        output = F.softmax(output, dim=1)
        # print("output:")
        # print(output.shape)

        return output

    def default_optimizer(self):
        return torch.optim.Adadelta(self.parameters(), lr=1.0, rho=0.9, eps=1e-6, weight_decay=0.001)
    def default_loss_fn(self):
        return torch.nn.CrossEntropyLoss()