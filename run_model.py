# -*- coding: utf-8 -*-
# Credit to: https://github.com/easyfan327/Pytorch-RETAIN
# We are using someone else's implementation as the base structure for ours.
# Implements the model proposed in:
    # An Interpretable Disease Onset Predictive Model
    # Using Crossover Attention Mechanism
    # From Electronic Health Records

import torch
import numpy as np
import random
import argparse
import time
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from torch.utils.data import Dataset

from model_coam import CoamBetaNN, CoamNN, CoamAlphaNN, load_data_coam, init_data_coam
from model_retain import RetainNN, load_data_retain, init_data_retain

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
        "--epochs", type=int, dest="n_epochs", default=10, help="Number of epochs to run for"
    )
    parser.add_argument(
        "--batch_size", type=int, default=100, help="How many data points to consider in a single batch"
    )
    parser.add_argument(
        "--model", type=str, default="coam", help="Which model to run", choices=['coam', 'retain', 'coam_alpha', 'coam_beta']
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

def init_params(args):
    params = {}
    # Some Global Constant parameters
    params["num_diagnoses_codes"] = 942
    params["num_prescription_codes"] = 3271

    params["dropout_p"] = 0.5

    params["test_ratio"] = 0.2
    params["validation_ratio"] = 0.1
    params["reverse_rnn_feeding"] = True
    params["output_dropout_p"] = 0.8
    params["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Add all the command line args to this params dict
    for arg in vars(args):
        params[arg] = getattr(args, arg)
    
    # Override all of the files with the prefix specified by data_dir
    if not args.data_dir.endswith("/"):
        args.data_dir += "/"
    params["diagnoses_file"] = args.data_dir+args.diagnoses_file
    params["labels_file"] = args.data_dir+args.labels_file
    params["prescriptions_file"] = args.data_dir+args.prescriptions_file

    # This shouldn't ever be directly used
    del params["data_dir"]
    return params

def init_data(params):
    if params['model'] == 'retain':
        return init_data_retain(params)
    else:
        return init_data_coam(params)

def model_data(model, set_x, set_y, params):
    if params['model'] == 'retain':
        x, y_true = load_data_retain(set_x, set_y, params)
        return model(x), y_true
    else:
        xDiagnoses, xPrescriptions, y_true = load_data_coam(set_x, set_y, params)
        return model(xDiagnoses, xPrescriptions), y_true

# Runs the data through the model and then backpropagates. Returns the loss vector for all of the batches.
def trainModel(model, train_set_x, train_set_y, optimizer, loss_fn, params):
    loss_vector = torch.zeros(params["n_batches"], dtype=torch.float)
    for index in random.sample(range(params["n_batches"]), params["n_batches"]):
        lIndex = index*params["batch_size"]
        rIndex = (index+1)*params["batch_size"]

        set_x = train_set_x[lIndex:rIndex]
        set_y = train_set_y[lIndex:rIndex]
        
        pred, y_true = model_data(model, set_x, set_y, params)
        y_pred = pred.squeeze(1)

        loss = loss_fn(y_pred, y_true)
        loss.backward()
        loss_vector[index] = loss
        optimizer.step()
        optimizer.zero_grad()
    return loss_vector

# Evaluates the current model against the test_set data.
def evalModel(model, set_x, set_y, params):
    pred, y_true = model_data(model, set_x, set_y, params)
    y_pred = pred.detach().cpu().numpy()

    y_true = y_true.unsqueeze(1)
    y_true = torch.zeros(pred.shape).to(params["device"]).scatter_(1, y_true, 1)
    y_true = y_true.detach().cpu().numpy()
    auc = roc_auc_score(y_true=y_true, y_score=y_pred)
    
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    y_pred = np.floor(y_pred + 0.5)
    y_pred = y_pred.astype(np.int32)
    y_true = y_true.astype(np.int32)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    return auc, p, r, f

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
    params = init_params(parse_arguments(PARSER))
    train_set_x, train_set_y, _, _, test_set_x, test_set_y = init_data(params)

    model = None
    if params["model"] == 'retain':
        model = RetainNN(params=params).to(params["device"])
    elif params["model"] == 'coam':
        model = CoamNN(params=params).to(params["device"])
    elif params["model"] == 'coam_alpha':
        model = CoamAlphaNN(params=params).to(params["device"])
    elif params["model"] == 'coam_beta':
        model = CoamBetaNN(params=params).to(params["device"])
    tot = 0
    for p in model.parameters():
        if p.requires_grad:
            tot += p.numel()
    print('numParameters:', tot)

    optimizer = model.default_optimizer()
    loss_fn = model.default_loss_fn()

    if params["load_model"]: # If loading a model
        print('Evaluating Model')
        model.load_state_dict(torch.load(params["load_model"]))
        model.eval()
        test_auc, p, r, f = evalModel(model, test_set_x, test_set_y, params)
        print("{:.4f}, \t {:.4f},{:.4f},{:.4f}".format(test_auc, p,r,f))
    else: # If training a model
        print(f'Begin training {params["model"]}')
        params["n_batches"] = int(np.ceil(float(len(train_set_y)) / float(params["batch_size"])))
        best_test_auc = 0
        best_epoch = 0
        timeTotal = 0
        numTimes = 0
        prevTime = time.perf_counter()
        currTime = -1 # Initialized before printing
        for epoch in range(params["n_epochs"]):
            model.train()
            loss_vector = trainModel(model, train_set_x, train_set_y, optimizer=optimizer, loss_fn=loss_fn, params=params)

            model.eval()
            test_auc, p, r, f = evalModel(model, test_set_x, test_set_y, params)

            if test_auc > best_test_auc:
                best_test_auc = test_auc
                best_epoch = epoch
            
            currTime = time.perf_counter()
            elapsed = currTime - prevTime
            print("{}, {:.4f} \t {:.4f},{:.4f} \t {:.4f},{:.4f},{:.4f}".format(epoch, elapsed, torch.mean(loss_vector), test_auc, p,r,f))
            timeTotal += elapsed
            prevTime = currTime
        
        avgEpochTime = timeTotal / params["n_epochs"]

        if params["save_model"]:
            torch.save(model.state_dict(), params["save_model"])
        print("best auc = {} at epoch {}. Average Time Per Epoch = {}".format(best_test_auc, best_epoch, avgEpochTime))