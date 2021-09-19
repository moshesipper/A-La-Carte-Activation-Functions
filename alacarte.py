# Deep, A La Carte
# copyright 2021 moshe sipper  
# www.moshesipper.com 

from string import ascii_lowercase
from random import choices, choice, seed
from sys import stdin, exit
# from os import makedirs
# from os.path import exists
from pandas import read_csv
from statistics import median
from pathlib import Path
import numpy as np
from copy import deepcopy
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import fetch_openml, make_classification
from pmlb import fetch_data, classification_dataset_names
import torch
import torch.nn as nn
from torch.autograd import Variable
from mlxtend.evaluate import permutation_test
import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)#optuna.logging.WARNING)

from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
EASYDS = { 'cancer': load_breast_cancer, 'iris': load_iris, 'wine': load_wine, 'digits': load_digits }

# activation functions
import activation_layers as al # defines math functions as AF layers
ALAYERS = [nn.ELU, nn.Hardshrink, nn.Hardtanh, nn.LeakyReLU, nn.LogSigmoid, nn.PReLU, nn.ReLU, nn.ReLU6, nn.RReLU, nn.SELU, nn.CELU, nn.GELU, nn.Sigmoid, nn.Softplus, nn.Softshrink, nn.Softsign, nn.Tanh, nn.Tanhshrink, nn.Softmin, nn.Softmax, nn.LogSoftmax, al.Abs, al.Acos, al.Angle, al.Asin, al.Atan, al.Ceil, al.Cos, al.Cosh, al.Digamma, al.Erf, al.Erfc, al.Exp, al.Floor, al.Frac, al.Log, al.Log10, al.Neg, al.Round, al.Sin, al.Sinh, al.Tan, al.Trunc, al.GumbelSoftmax, al.GeneralizedSwish, al.SigmoidDerivative, al.CLogLogM, al.Mish]
ALAYER_NAMES = [l.__name__ for l in ALAYERS]


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_EPOCHS = 300 # neural network training epochs
N_EPOCHS_STOP = 10 # stop traning if no improvment on test set during last N_EPOCHS_STOP epochs
N_TRIALS = 1000 # number of optuna trials, also number of random networks
N_LAYERS = (5, 10)

def rndstr(n): 
    return ''.join(choices(ascii_lowercase, k=n))

def fprint(fname, s):
    if stdin.isatty(): # running interactively 
        print(s) 
    with open(Path(fname), 'a') as f: 
        f.write(s)

def save_params(fname, dsname, version, n_samples, n_features, n_classes, n_replicates):
    activations = ', '.join([a.__name__ for a in ALAYERS])
    n_layers = ', '.join([str(n) for n in N_LAYERS])
    fprint(fname, f' DEVICE: {DEVICE}\n dsname: {dsname}\n version: {version}\n n_samples: {n_samples}\n n_features: {n_features}\n n_classes: {n_classes}\n n_replicates: {n_replicates}\n N_EPOCHS: {N_EPOCHS}\n N_LAYERS: ({n_layers})\n N_EPOCHS_STOP: {N_EPOCHS_STOP}\n N_TRIALS: {N_TRIALS}\n ALAYERS: ({activations}) \n')

def get_args():  
    parser = ArgumentParser()
    parser.add_argument('-resdir', dest='resdir', type=str, action='store', help='directory where results are placed')
    parser.add_argument('-dsname', dest='dsname', type=str, action='store', help='dataset name')
    # parser.add_argument('-nlayers', dest='n_layers', type=int, action='store', help='number of neural-network layers')
    parser.add_argument('-nrep', dest='n_replicates', type=int, action='store', help='number of replicate runs')
    args = parser.parse_args()
    if None in [getattr(args, arg) for arg in vars(args)]:
        parser.print_help()
        exit()
    resdir, dsname, n_replicates = args.resdir+'/', args.dsname, args.n_replicates
    return resdir, dsname, n_replicates

def get_dataset(dsname):
    version, openml = -1, False
    if dsname ==  'clftest':
        X, y = make_classification(n_samples=10, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2)
    elif dsname in EASYDS.keys():
        X, y = EASYDS[dsname](return_X_y=True)
    elif dsname in classification_dataset_names: # PMLB datasets
        X, y = fetch_data(dsname, return_X_y=True, local_cache_dir='../datasets/pmlbclf')
    else:
        try: # dataset from openml? 
            data = fetch_openml(data_id=int(dsname), cache=True, data_home='../datasets/scikit_learn_data')
            X, y = data['data'], data['target']
            dsname = data['details']['name']
            version = data['details']['version']
            openml = True
        except:
            try: # a csv file in datasets folder?
                data = read_csv('../datasets/' + dsname + '.csv', sep=',')
                array = data.values
                X, y = array[:,0:-1], array[:,-1] # target is last col
                # X, y = array[:,1:], array[:,0] # target is 1st col
            except Exception as e: # give up
                print('oops, looks like there is no such dataset: ' + dsname)
                exit(e)
              
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    
    y = LabelEncoder().fit_transform(y) # Encode target labels with value between 0 and n_classes-1
    
    return X.to_numpy(), y, n_samples, n_features, n_classes, dsname, version, openml

class Model(nn.Module):
    def __init__(self, input_dim, output_dim, activations):
        super(Model, self).__init__()
        
        n_layers = len(activations)
        assert n_layers in N_LAYERS
                
        kwargs = [ {'dim': 1} if layer in [nn.Softmin, nn.Softmax, nn.LogSoftmax] else {} for layer in activations ] 
        
        layers = [nn.Linear(input_dim, 64), activations[0](**kwargs[0])]
        for i in range(1, n_layers-1):
            layers += [nn.Linear(64, 64), activations[i](**kwargs[i])]
        layers += [nn.Linear(64, output_dim), activations[n_layers-1](**kwargs[n_layers-1])]
        
        self.main = nn.Sequential(*layers) 

    def forward(self, x):
        return self.main(x)

def network_train(activations, X_train, y_train, n_classes):
    model = Model(input_dim=X_train.shape[1], output_dim=n_classes, activations=activations)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    train_score = np.zeros((N_EPOCHS,))
    epochs_small_improve = 0
    for epoch in range(N_EPOCHS):
        model.train() # set to training mode, https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        
        optimizer.zero_grad() # before backward pass zero all gradients for the variables to be updated
        loss.backward() # compute backward pass (gradient of loss with respect to all tensors with requires_grad=True)
        optimizer.step() # calling the step function on an Optimizer makes an update to its parameters
        
        model.eval() # set to evaluation mode, https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        with torch.no_grad():
            correct = (torch.argmax(model(X_train), dim=1) == y_train).type(torch.FloatTensor)
            train_score[epoch] = correct.mean()
        
        if (epoch>0) and (train_score[epoch] - train_score[epoch-1])<0.001:
            epochs_small_improve += 1
        else:
            epochs_small_improve = 0
        
        if epochs_small_improve >= N_EPOCHS_STOP:
            break;
            
    return model, float(train_score[epoch]) # tensor to float

def network_predict(model, X, y_true):
    model.eval() # set to evaluation mode, https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    with torch.no_grad():
        correct = (torch.argmax(model(X), dim=1) == y_true).type(torch.FloatTensor)
        score = correct.mean()
    return float(score) # tensor to float

def standard_network(X_train, y_train, X_test, y_test, n_classes, n_layers): 
# return test score of standard network (ReLU,..., ReLU, Softmax)
    activations = []
    for i in range(n_layers-1):
        activations.append(nn.ReLU)
    activations.append(nn.Softmax)
    model, train_score = network_train(activations, X_train, y_train, n_classes)
    test_score = network_predict(model, X_test, y_test)
    net_str = ', '.join([t.__name__ for t in activations])
    return test_score, net_str

def random_network(X_train, y_train, X_test, y_test, n_classes, n_layers, n_rnd_nets): 
# return best test score and best network of n_rnd_nets random networks
    best_test_score, best_net = 0, None
    for i in range(n_rnd_nets):
        rndnet = [choice(ALAYERS) for i in range(n_layers)]
        model, train_score = network_train(rndnet, X_train, y_train, n_classes)
        test_score = network_predict(model, X_test, y_test)
        if test_score > best_test_score:
            best_test_score = test_score
            best_net = deepcopy(rndnet)
 
    best_net_str = ', '.join([t.__name__ for t in best_net])
    return best_test_score, best_net_str
    
class Objective(object): # used by Optuna
    def __init__(self, X_train, y_train, n_classes, n_layers):
        self.X_train = X_train
        self.y_train = y_train
        self.n_classes = n_classes
        self.n_layers = n_layers

    def create_activations(self, trial):
        activations = []
        for i in range(self.n_layers):
            name = trial.suggest_categorical(f'activation{i}', ALAYER_NAMES)
            activations.append(ALAYERS[ALAYER_NAMES.index(name)])
        return activations

    def __call__(self, trial):
        activations = self.create_activations(trial)    
        model, train_score = network_train(activations, self.X_train, self.y_train, self.n_classes)
        trial.set_user_attr(key='best_model', value=model)
        return train_score
# end class Objective

def optuna_callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key='best_model', value=trial.user_attrs['best_model'])

def optuna_network(X_train, y_train, X_test, y_test, n_classes, n_layers, sampler, n_trials): 
# return best test score and best network found by Optuna
    objective = Objective(X_train, y_train, n_classes, n_layers)
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(objective, n_trials=n_trials, callbacks=[optuna_callback])
    s = [study.best_trial.params[f'activation{i}'] for i in range(n_layers)]
    best_net_str = ', '.join(s)
    best_model = study.user_attrs['best_model']
    best_test_score = network_predict(best_model, X_test, y_test)
    return best_test_score, best_net_str
        
# main 
def main():
    seed() # initialize internal state of random number generator
    resdir, dsname_id, n_replicates = get_args()
    X, y, n_samples, n_features, n_classes, dsname, version, openml = get_dataset(dsname_id)
    print_ds = f'{dsname} ({dsname_id})' if openml else f'{dsname}' # openml datasets given as ints, get_dataset converts to string
    # if not exists(resdir): 
        # makedirs(resdir)
    fname = resdir + dsname_id + '_' + rndstr(6) + '.txt'
    save_params(fname, print_ds, version, n_samples, n_features, n_classes, n_replicates)

    # X = torch.from_numpy(X).to(DEVICE)
    # y = torch.from_numpy(y.astype(np.int))

    for n_layers in N_LAYERS:
        fprint(fname, f'\n\nExecuting {n_replicates} replicates with {n_layers}-layer networks\n')
        scores = { 'standard': [], 'random': [], 'tpe': [], 'cmaes': [] }
        for rep in range(n_replicates):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  
            
            sc = StandardScaler() 
            X_train = sc.fit_transform(X_train) # scaled data has mean 0 and variance 1 (only over training set)
            X_test = sc.transform(X_test) # use same scaler as one fitted to training data

            X_train = torch.from_numpy(X_train).to(DEVICE)    
            X_test = torch.from_numpy(X_test).to(DEVICE)    
            y_train = torch.from_numpy(y_train.astype(np.int))
            y_test = torch.from_numpy(y_test.astype(np.int))
            X_train = Variable(X_train, requires_grad=True).float().to(DEVICE)
            X_test  = Variable(X_test, requires_grad=True).float().to(DEVICE)
            y_train = Variable(y_train).long().to(DEVICE)
            y_test  = Variable(y_test).long().to(DEVICE)
            
            # standard network
            test_score, net_str = standard_network(X_train, y_train, X_test, y_test, n_classes, n_layers)
            scores['standard'].append(test_score)
            fprint(fname, f'\n replicate {rep}, standard network: test_score {test_score}, net ({net_str})') 
            
            # random network
            best_test_score, best_net_str = random_network(X_train, y_train, X_test, y_test, n_classes, n_layers, N_TRIALS)
            scores['random'].append(best_test_score)
            fprint(fname, f'\n replicate {rep}, best random network: best_test_score {best_test_score}, best net ({best_net_str})') 
            
            # network found via optuna using TPESampler
            sampler = optuna.samplers.TPESampler()
            best_test_score, best_net_str = optuna_network(X_train, y_train, X_test, y_test, n_classes, n_layers, sampler, N_TRIALS)
            scores['tpe'].append(best_test_score)
            fprint(fname, f'\n replicate {rep}, best optuna TPESampler network: best_test_score {best_test_score}, best net ({best_net_str})') 
    
            # network found via optuna using CmaEsSampler
            sampler = optuna.samplers.CmaEsSampler()
            best_test_score, best_net_str = optuna_network(X_train, y_train, X_test, y_test, n_classes, n_layers, sampler, N_TRIALS)
            scores['cmaes'].append(best_test_score)
            fprint(fname, f'\n replicate {rep}, best optuna CmaEsSampler network: best_test_score {best_test_score}, best net ({best_net_str})') 
    
         
        # done all replicates, compute final stats
        fprint(fname, f'\n\n summary of {n_replicates} replicates with {n_layers}-layer networks:\n')
        n_rounds=10000 # number of permutation-test rounds
        th1, th2 = 0.001, 0.05 # p-value thresholds
        medians = []
        for net, net_scores in scores.items():
            medians.append((net, median(net_scores), net_scores))
        n_scores = len(medians)
        
        # permutation testing of each method vs. standard network
        assert(medians[0][0] == 'standard')
        med = round(medians[0][1],3)
        lrows = len(N_LAYERS)
        if n_layers==N_LAYERS[0]: 
            latex = f'\multirow{{{lrows}}}*{{{dsname_id}}} & \multirow{{{lrows}}}*{{{n_samples}}} & \multirow{{{lrows}}}*{{{n_features}}}  & \multirow{{{lrows}}}*{{{n_classes}}} &'
        else:
            latex = '& & & &'   # 'latex' holds a row in the LaTeX table of the paper
        latex += f' {n_layers} & {med} & '
        s = f' standard {med} vs., '
        for i in range(1, n_scores):
            net = medians[i][0]
            med = round(medians[i][1],3)
            pval = permutation_test(medians[i][2], medians[0][2],  method='approximate', num_rounds=n_rounds,\
                                         func=lambda x, y: np.abs(np.median(x) - np.median(y)))
            pv = '!!' if pval<th1 else '!' if pval<th2 else '\b'
            pval = round(pval,4)
            s += f'{net}: {med} (p {pval} {pv}), ' 
            latex += f'{med} {pv} & '
        fprint(fname, s[:-2] + '\n')  
        
        # rank methods, do permutation testing of rank i vs lower rank i+1
        medians = sorted(medians, key = lambda x: x[1], reverse=True)    
        s = ' sorted, '
        for i in range(n_scores):
            net = medians[i][0]
            med = round(medians[i][1],3)
            if i < n_scores-1:
                pval = permutation_test(medians[i][2], medians[i+1][2],  method='approximate', num_rounds=n_rounds,\
                                             func=lambda x, y: np.abs(np.median(x) - np.median(y)))
                pv = '!!' if pval<th1 else '!' if pval<th2 else '\b'
                pval = round(pval,4)
                s += f'{net}: {med} (p {pval} {pv}), ' 
            else:
                s += f'{net}: {med}'
                
            if i==0: #first place
                latex += f'{net} {pv} '
        fprint(fname, s + '\n')
        fprint(fname, f'{latex} \\\\ \n')

##############        
if __name__== "__main__":
  main()
