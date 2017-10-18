import numpy as np
from MNIST.mnist import MNIST
from CIFAR10.cifar10 import CIFAR10

def load_mnist(normalize=True, flatten=False, one_hot_label=False):
    mnist = MNIST()
    return mnist.load_data(normalize, flatten, one_hot_label, 'both')

def load_cifar10(normalize=True, flatten=False, one_hot_label=False):
    cifar = CIFAR10()
    return cifar.load_data(normalize, flatten, one_hot_label, 'both')

def create_minibatch(dataset, batch_number=50):
    '''
    # Minibatch Generator
    generates minibatch from a dataset
    ## Arguments
    dataset: tuple of (data, label)
    batch_number: batch number
    '''
    data, label = dataset
    index = np.random.choice(data.shape[0],batch_number)
    data_batch = data[index]
    label_batch = label[index]
    return (data_batch, label_batch)
