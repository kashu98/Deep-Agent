import pickle
import numpy as np
import os.path

class MNIST:
    def __init__(self):
        self.dataset = {}
        self.path = os.path.dirname(os.path.abspath(__file__))

    def _load_img(self, filename):
        with open(filename, 'rb') as file:
            data = np.frombuffer(file.read(), np.uint8, offset=16)
        return data.reshape(-1, 784)
    
    def _load_label(self, filename):
        with open(filename, 'rb') as file:
            labels = np.frombuffer(file.read(), np.uint8, offset=8)    
        return labels
    
    def _creat_pickle(self):
        self.dataset['train_img'] =  self._load_img(self.path + '/train-images.idx3-ubyte')
        self.dataset['train_label'] = self._load_label(self.path + '/train-labels.idx1-ubyte')    
        self.dataset['test_img'] = self._load_img(self.path + '/t10k-images.idx3-ubyte')
        self.dataset['test_label'] = self._load_label(self.path + '/t10k-labels.idx1-ubyte')
        with open(self.path + '/mnist.pkl', 'wb') as file:
            pickle.dump(self.dataset, file, -1)

    def _one_hot_label(self, X):
        T = np.zeros((X.size, 10))
        for idx, row in enumerate(T):
            row[X[idx]] = 1
        return T

    def load_data(self, normalize=True, flatten=False, one_hot_label=False, option='train', **kwargs):
        '''
        ## Arguments
        normalize : if true, normalize the input pixel
        one_hot_label : if true, creat one hot label
        flatten : if true, load the image as a line
        option: select option
            train: return train data only\n
            test: return test data only\n
            both: return both train and test data
        '''
        if not os.path.exists(self.path + '/mnist.pkl'):
            self._creat_pickle()

        with open(self.path + '/mnist.pkl', 'rb') as file:
            dataset = pickle.load(file)
    
        if normalize:
            for i in ('train_img', 'test_img'):
                dataset[i] = dataset[i].astype(np.float32)
                dataset[i] /= 255.0
                dataset[i] += 0.01
            
        if one_hot_label:
            dataset['train_label'] = self._one_hot_label(dataset['train_label'])
            dataset['test_label'] = self._one_hot_label(dataset['test_label'])
    
        if not flatten:
            for i in ('train_img', 'test_img'):
                dataset[i] = dataset[i].reshape(-1, 1, 28, 28)

        if option == 'train':
            return (dataset['train_img'], dataset['train_label'])
        elif option == 'test':
            return (dataset['test_img'], dataset['test_label'])
        elif option == 'both':
            return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label']) 
