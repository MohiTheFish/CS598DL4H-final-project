# Credit to: https://github.com/easyfan327/Pytorch-RETAIN
# We are using someone else's implementation as the baseline for ours.

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle as pickle
import random
import argparse
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from torch.utils.data import Dataset


def parse_arguments(parser):
    """Read user arguments"""
    parser.add_argument(
        "--data_dir", type=str, default='data/processed_data/', help="Directory for processed MIMIC-III data. This gets prefixed to all of the other file names."
    )
    parser.add_argument(
        "--diagnoses_file", type=str, default='3digitICD9.seqs.pkl', help="processed DIAGNOSES data file"
    )
    parser.add_argument(
        "--prescriptions_file", type=str, default='prescriptions.pkl', help="processed PRESCRIPTIONS data file"
    )
    parser.add_argument(
        "--labels_file", type=str, default='morts.pkl', help="PRESCRIPTIONS data file"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to run for"
    )
    parser.add_argument(
        "--save_model", type=str, help="Output path to save the trained model parameters to.", metavar='OUTPUT_PATH'
    )
    parser.add_argument(
        "--load_model", type=str, 
        metavar='MODEL_PATH',
        help="You can load an existing saved model by providing the path to the model parameters. Only the evaluation will occur if this is provided"
    )
    args = parser.parse_args()

    if args.save_model and args.load_model:
        raise AttributeError('Load and Save model should not both be provided')

    return args

class CustomDataset(Dataset):
    def __init__(self, diagnoses, prescriptions, labels):
        self.diagnoses = diagnoses
        self.prescriptions = prescriptions
        self.y = labels
    
    def __len__(self):
        return len(self.diagnoses)
    
    def __getitem__(self, index):
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

def toTensor(xDiagnoses, xPrescriptions, set_y, params):
    xDiagnoses = xToTensor(xDiagnoses, params['num_diagnoses_codes'])
    xPrescriptions = xToTensor(xPrescriptions, params['num_prescription_codes'])
    y = yToTensor(set_y)
    return xDiagnoses, xPrescriptions, y

def loadData(set_x, set_y, params):
    xDiagnoses, xPrescriptions = separateData(set_x)
    return toTensor(xDiagnoses, xPrescriptions, set_y, params)

class CoamNN(nn.Module):
    def __init__(self, params: dict):
        super(CoamNN, self).__init__()
        """
        num_embeddings(int): size of the dictionary of embeddings
        embedding_dim(int) the size of each embedding vector
        """
        self.emb_layer_diagnoses = nn.Linear(in_features=params["num_diagnoses_codes"], out_features=params["embedding_size"])
        self.emb_layer_prescriptions = nn.Linear(in_features=params["num_prescription_codes"], out_features=params["embedding_size"])
        
        self.dropout = nn.Dropout(params["dropout_p"])

        self.diagnoses_hidden_size = 128
        self.prescriptions_hidden_size = 128

        self.diagnoses_rnn = nn.GRU(input_size=params["embedding_size"], hidden_size=self.diagnoses_hidden_size, bidirectional=True)
        self.prescriptions_rnn = nn.GRU(input_size=params["embedding_size"],  hidden_size=self.prescriptions_hidden_size, bidirectional=True)

        self.diagnoses_attention = nn.Linear(in_features=params["embedding_size"], out_features=params["embedding_size"] )
        self.treatment_attention = nn.Linear(in_features=params["embedding_size"], out_features=params["embedding_size"] )

        self.concatenation = nn.Linear(in_features=params["embedding_size"], out_features=params["embedding_size"]//2)

        self.prediction = nn.Linear(in_features=params["embedding_size"]//2, out_features=params["num_class"])


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

        d_rnn_output, _ = self.diagnoses_rnn(d)
        p_rnn_output, _ = self.prescriptions_rnn(p)
        # print(d_rnn_output.shape, p_rnn_output.shape)

        # Combining operation is unclear
        # com = torch.cat((d, p), 2).permute((2, 0, 1))
        com = d_rnn_output + p_rnn_output
        # print(com.shape)


        alpha = self.diagnoses_attention(com)
        beta = self.treatment_attention(com)
        # print(alpha.shape, beta.shape)

        alpha_t = F.softmax(alpha, dim=2)
        beta_t = F.softmax(beta, dim=2)

        # print(alpha_t.shape, beta_t.shape)

        h_t = torch.sum(beta_t*d_rnn_output, dim=0)
        g_t = torch.sum(alpha_t*p_rnn_output, dim=0)

        # print(h_t.shape, g_t.shape)

        p = self.concatenation(h_t + g_t)
        p = torch.tanh(p)

        # print(p.shape)
        output = self.prediction(p)
        # print(output.shape)
        output = F.softmax(output, dim=1)

        return output

def init_params(args):
    params = {}
    # embedding matrix
    params["num_diagnoses_codes"] = 942
    params["num_prescription_codes"] = 3271
    params["embedding_size"] = 256
    # embedding dropout
    params["dropout_p"] = 0.5
    params["num_class"] = 2

    params["batch_size"] = 100
    params["n_epochs"] = args.epochs

    params["test_ratio"] = 0.2
    params["validation_ratio"] = 0.1

    # Add all the command line args to this params dict
    for arg in vars(args):
        params[arg] = getattr(args, arg)
    
    # Override all of the files with the prefix 
    if not args.data_dir.endswith("/"):
        args.data_dir += "/"
    params["diagnoses_file"] = args.data_dir+args.diagnoses_file
    params["labels_file"] = args.data_dir+args.labels_file
    params["prescriptions_file"] = args.data_dir+args.prescriptions_file

    # This shouldn't ever be directly used
    del params["data_dir"]
    return params

def padMatrixWithoutTime(data, max_codes):
    lengths = np.array([len(d) for d in data]).astype('int32')
    n_samples = len(data)
    maxlen = np.max(lengths)

    x = np.zeros((maxlen, n_samples, max_codes))
    for idx, d in enumerate(data):
        for xvec, subd in zip(x[:, idx, :], d):
            xvec[subd] = 1.

    return x


def init_data(params: dict):
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

    # Using the number of visits from diagnoses for diagnoses and prescriptions because the shapes are identical
    def order_set(set_x, set_y, indices):
        sorted_index = len_argsort(diagnoses[indices])
        set_x = [(set_x[0][i], set_x[1][i]) for i in sorted_index]
        set_y = [set_y[i] for i in sorted_index]
        return set_x, set_y

    train_set_x, train_set_y = order_set(train_set_x, train_set_y, train_indices)
    test_set_x, test_set_y = order_set(test_set_x, test_set_y, test_indices)
    valid_set_x, valid_set_y = order_set(valid_set_x, valid_set_y, valid_indices)

    return train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y

# Runs the data through the model and then backpropagates. Returns the loss vector for all of the batches.
def trainModel(model, train_set_x, train_set_y, optimizer, loss_fn, params):
    loss_vector = torch.zeros(params["n_batches"], dtype=torch.float)
    for index in random.sample(range(params["n_batches"]), params["n_batches"]):
        lIndex = index*params["batch_size"]
        rIndex = (index+1)*params["batch_size"]

        set_x = train_set_x[lIndex:rIndex]
        set_y = train_set_y[lIndex:rIndex]
        xDiagnoses, xPrescriptions, y_true = loadData(set_x, set_y, params)
        
        # print('xDiagnoss.shape:', xDiagnoses.shape)
        pred = model(xDiagnoses, xPrescriptions)
        y_pred = pred.squeeze(1)

        # print(pred.shape, y.shape)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        loss_vector[index] = loss
        optimizer.step()
        optimizer.zero_grad()
    return loss_vector

# Evaluates the current model against the test_set data.
def evalModel(model, set_x, set_y):
    xDiagnoses, xPrescriptions, y_true = loadData(set_x, set_y, params)

    pred = model(xDiagnoses, xPrescriptions)
    y_pred = pred.detach().cpu().numpy()

    y_true = y_true.unsqueeze(1)
    y_true = torch.zeros(pred.shape).to(device).scatter_(1, y_true, 1)
    y_true = y_true.detach().cpu().numpy()
    # print(y_pred)
    # print(y_true)
    auc = roc_auc_score(y_true=y_true, y_score=y_pred)
    
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    y_pred = np.floor(y_pred + 0.5)
    y_pred = y_pred.astype(np.int32)
    y_true = y_true.astype(np.int32)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    return auc, p, r, f

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    PARSER = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
    params = init_params(parse_arguments(PARSER))
    train_set_x, train_set_y, _, _, test_set_x, test_set_y = init_data(params)

    model = CoamNN(params=params).to(device)
    tot = 0
    for p in model.parameters():
        if p.requires_grad:
            tot += p.numel()
    print('numParameters:', tot)

    optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
    loss_fn = torch.nn.CrossEntropyLoss()

    if params["load_model"]: # If loading a model
        print('Evaluating Model')
        model.load_state_dict(torch.load(params["load_model"]))
        model.eval()
        test_auc, p, r, f = evalModel(model, test_set_x, test_set_y)
        print("{:.4f}, \t {:.4f},{:.4f},{:.4f}".format(test_auc, p,r,f))
    else: # If training a model
        print('Begin training')
        params["n_batches"] = int(np.ceil(float(len(train_set_y)) / float(params["batch_size"])))
        best_test_auc = 0
        best_epoch = 0
        for epoch in range(params["n_epochs"]):
            model.train()
            loss_vector = trainModel(model, train_set_x, train_set_y, optimizer=optimizer, loss_fn=loss_fn, params=params)

            model.eval()
            # evalModel(model, valid_set_x)
            test_auc, p, r, f = evalModel(model, test_set_x, test_set_y)

            if test_auc > best_test_auc:
                best_test_auc = test_auc
                best_epoch = epoch

            print("{},{:.4f},{:.4f} \t {:.4f},{:.4f},{:.4f}".format(epoch, torch.mean(loss_vector), test_auc, p,r,f))

        if params["save_model"]:
            torch.save(model.state_dict(), params["save_model"])
        print("best auc = {} at epoch {}".format(best_test_auc, best_epoch))