import numpy as np
import sys

class Activation:
    def __init__(self):
        self.X = None
        self.Y = None
        self.sign = None
    def forward(self, X):
        self.X = X

class Identity:
    '''Identity function
    '''
    def __init__(self):
        pass
    
    def forward(self, X):
        return X

    def backward(self, dY):
        return dY

class ReLU(Activation):
    '''Rectified Linear Unit 
    '''
    def __init__(self, α=10e-7):
        super().__init__()
        self.α = α

    def forward(self, X):
        self.sign = (X <= 0)
        X[self.sign] = X[self.sign] * self.α
        return X

    def backward(self, dY):
        dY[self.sign] = dY[self.sign] * self.α
        return dY

class LReLU(ReLU):
    '''Leaky Rectified Linear Unit 
    '''
    def __init__(self):
        super().__init__(α=0.01)

    def forward(self, X):
        return super().forward(X)

    def backward(self, dY):
        return super().backward(dY)

class PReLU(ReLU):
    '''Parameteric Rectified Linear Unit
    '''
    def __init__(self, α=0.01):
        super().__init__()
        self.α = α

    def forward(self, X):
        return super().forward(X)

    def backward(self, dY):
        return super().backward(dY)

class ELU(Activation):
    '''Exponential Linear Unit
    '''
    def __init__(self, α=0.5, λ=1.0):
        super().__init__()
        self.α = α
        self.λ = λ

    def forward(self, X):
        X = self.λ * X
        self.sign = (X <= 0)
        X[self.sign] = self.α * (np.exp(X[self.sign]) - 1.0)
        self.Y = X
        return X

    def backward(self, dY):
        dY = self.λ * dY
        dY[self.sign] = dY[self.sign] * (self.Y + self.α)
        return dY

class SELU(ELU):
    '''Scaled Exponential Linear Unit (Klambauer et al., 2017)
    '''
    def __init__(self):
        α = 1.67326
        λ = 1.0507
        super().__init__(α, λ) 
        
    def forward(self, X):
        return super().forward(X)

    def backward(self, dY):
        return super().backward(dY)

class Sigmoid(Activation):
    '''Logistic Function
    '''
    def __init__(self):
        super().__init__()

    def forward(self, X):
        self.Y = 1/(1 + np.exp(-X))
        return self.Y

    def backward(self, dY):
        dX = dY * self.Y * (1.0 - self.Y)
        return dX

class SoftPlus(Sigmoid):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        self.X = X
        return np.log(1.0 + np.exp(X))

    def backward(self, dY):
        dX = dY * super().forward(self.X)
        return dX

class Tanh(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        self.Y = 2.0/(1.0 + np.exp(-2 * X) - 1.0)
        return self.Y

    def backward(self, dY):
        dX = dY * (1.0 - (self.Y)**2)
        return dX

class ArcTan(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        self.X = X
        return np.arctan(X)

    def backward(self, dY):
        dX = dY/(1.0 + (self.X)**2)
        return dX

class SoftSign(Activation):
    def __init__(self):
        super().__init__()

    def forard(self, X):
        self.sign = (X < 0)
        aX = X.copy()
        aX[self.sign] = -1.0 * aX[self.sign]
        self.X = aX
        return X/(1.0 + aX)

    def backward(self, dY):
        return dY/(1.0 + self.X)**2 

