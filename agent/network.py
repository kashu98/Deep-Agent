from collections import OrderedDict
import h5py as h5
import yaml
from prettytable import PrettyTable
from activation import *
from layers import *
from optimizer import *
from visualizer import *

class Sequential:
    '''# Sequential Network Model
    ## Methods
    add: add a new layer to the neural network
    compile: add optimizer and loss function
    train: train the network
    predict: process the input without learning (without updating the weights)
    evaluate: get the accuracy of the network (test of the network)
    load_weights: load the optimized weight from HDF5 file
    save_weights: save the optimized weight to HDF5 file
    load_model: load the model from yaml format
    save_model: save the model to yaml format
    '''
    def __init__(self):
        # List all the layers to validate the input
        self.list_layer = []
        self.list_layer.append(type(Affine()))
        self.list_layer.append(type(Convolution()))
        self.list_layer.append(type(Pooling()))
        self.list_layer.append(type(Padding()))
        self.list_layer.append(type(Dropout()))
        self.list_layer.append(type(Classifier()))
        #self.list_layer.append(type(Maxout()))
        #self.list_layer.append(type(BatchNormalization()))
        #self.list_layer.append(type(Skip()))

        # List all the optimizer to validate the input
        self.list_opt = []
        self.list_opt.append(type(SGD()))
        self.list_opt.append(type(Momentum()))
        #self.list_opt.append(type(Nesterov_Momentum()))
        self.list_opt.append(type(AdaGrad()))
        self.list_opt.append(type(RMSProp()))
        #self.list_opt.append(type(Adam()))
        #self.list_opt.append(type(Adamdelta()))
        #self.list_opt.append(type(AdaMax()))
        #self.list_opt.append(type(Nadam()))

        self.layers = OrderedDict()
        self.params = {}
        self.grads = {}
        self.opt = None

    def add(self, layer):
        if not isinstance(layer, tuple(self.list_layer)):
            raise TypeError('The layer' +str(len(self.layers)) +' must be a layer class definened in layers.py. Not found: ' +str(layer))
        else:
            ly = layer
            self.layers.update({str(len(self.layers)):ly})

    def compile(self, optimizer):
        if not isinstance(optimizer, tuple(self.list_opt)):
            raise TypeError('The optimizer ' +str(optimizer) + ' is not defined.')
        else:
            self.opt = optimizer
    
    def train(self, X, T):
        self.layers[str(len(self.layers)-1)].set_label(T)
        
        # forward
        for i in self.layers.values():
            X = i.forward(X)

        # get all the parameters
        for i in self.layers:
            if self.layers[i].has_params() == True:
                self.params[i] = self.layers[i].get_params()
        
        # backward
        dY = None
        for i in OrderedDict(reversed(list(self.layers.items()))):
            dY = self.layers[str(i)].backward(dY)

        # get all the gradient
        for i in self.layers:
            if self.layers[i].has_params() == True:
                self.grads[i] = self.layers[i].get_grads()

        self.opt.optimize(self.params, self.grads)
    
    def predict(self, X):
        for i in self.layers.values():
            X = i.predict(X)
        return X

    def evaluate(self, X, T, disp=True):
        X = self.predict(X)
        if not X.size == T.size:
            T = np.argmax(T, axis=1)
        eval_list = (X == T)
        if disp:
            print("=============== Test Accuracy ===============")
            print('accuracy: ' + str(np.sum(eval_list)/eval_list.size*100) + '%')
        return np.sum(eval_list)/eval_list.size

    def save_weights(self, filename="weights.hdf5"):
        with h5.File(filename, 'w') as file:
            for i in self.params:
                grp = file.create_group(i)
                grp.create_dataset('weight', data=self.params[i]['weight'])
                grp.create_dataset('bias', data=self.params[i]['bias'])

    def load_weights(self, filename="weights.hdf5"):        
        with h5.File(filename, 'r') as file:
            for i in file:
                self.layers[i].load_params({'weight':np.array(file[i]['weight']), 'bias':np.array(file[i]['bias'])})

    def save_model(self, filename="model.yml", save_weights=True, weights_file="weights.hdf5"):
        with open(filename, 'w') as file:
            file.write(yaml.dump(self.layers))
        
        if save_weights:
            self.save_weights(weights_file)
        
    def load_model(self, filename="model.yml", load_weights=True, weights_file="weights.hdf5"):
        with open(filename, 'r') as file:
            self.layers = yaml.load(file)
        
        if load_weights:
            self.load_weights(weights_file)
        
    def show_model(self):
        table = PrettyTable(['Layer', 'Output Shape'])
        table.align['Layer'] = 'l'
        table.padding_width = 2
        for i in self.layers.values():
            table.add_row([str(i), i.get_out_shape()])
        print(table)
