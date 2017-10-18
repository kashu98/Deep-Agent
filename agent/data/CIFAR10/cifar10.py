import pickle
import numpy as np
import os.path

class CIFAR10:
    def __init__(self):
        self.dataset = {}
        self.path = os.path.dirname(os.path.abspath(__file__))
    
    def _load_data(self, filename):
        with open(filename, 'rb') as file:
            dataset = pickle.load(file, encoding='bytes')
        return (dataset[b'data'], dataset[b'labels'])

    def _creat_pickle(self):
        img, label = self._load_data(self.path + '/data_batch_1')
        for i in range(4):
            np.r_[img, self._load_data(self.path + '/data_batch_' + str(i+2))[0]]
            np.r_[label, self._load_data(self.path + '/data_batch_' + str(i+2))[1]]
        self.dataset['train_img'] = img
        self.dataset['train_label'] = label
        self.dataset['test_img'] = self._load_data(self.path + '/test_batch')[0]
        self.dataset['test_label'] = self._load_data(self.path + '/test_batch')[1]
        with open(self.path + '/cifar10.pkl', 'wb') as file:
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
        if not os.path.exists(self.path + '/cifar10.pkl'):
            self._creat_pickle()

        with open(self.path + '/cifar10.pkl', 'rb') as file:
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
                dataset[i] = dataset[i].reshape(-1, 3, 32, 32)

        if option == 'train':
            return (dataset['train_img'], dataset['train_label'])
        elif option == 'test':
            return (dataset['test_img'], dataset['test_label'])
        elif option == 'both':
            return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])

if __name__ == '__main__':
    cifar = CIFAR10()
    data = cifar.load_data(True, False, False, 'both')
    print(data)
