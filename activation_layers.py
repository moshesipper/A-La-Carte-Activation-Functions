# activation-function layers
# define math functions as activation layers
# also define new activation functions
# copyright 2021 moshe sipper  
# www.moshesipper.com 

import torch
import torch.nn as nn
import torch.nn.functional as tf

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# math functions

class Abs(nn.Module):
    def __init__(self):
        super(Abs, self).__init__()
    def forward(self, input):
        return torch.abs(input)

class Acos(nn.Module):
    def __init__(self):
        super(Acos, self).__init__()
    def forward(self, input):
        return torch.acos(input)

class Angle(nn.Module):
    def __init__(self):
        super(Angle, self).__init__()
    def forward(self, input):
        return torch.angle(input)

class Asin(nn.Module):
    def __init__(self):
        super(Asin, self).__init__()
    def forward(self, input):
        return torch.asin(input)

class Atan(nn.Module):
    def __init__(self):
        super(Atan, self).__init__()
    def forward(self, input):
        return torch.atan(input)

class Ceil(nn.Module):
    def __init__(self):
        super(Ceil, self).__init__()
    def forward(self, input):
        return torch.ceil(input)

class Cos(nn.Module):
    def __init__(self):
        super(Cos, self).__init__()
    def forward(self, input):
        return torch.cos(input)

class Cosh(nn.Module):
    def __init__(self):
        super(Cosh, self).__init__()
    def forward(self, input):
        return torch.cosh(input)

class Digamma(nn.Module):
    def __init__(self):
        super(Digamma, self).__init__()
    def forward(self, input):
        return torch.digamma(input)

class Erf(nn.Module):
    def __init__(self):
        super(Erf, self).__init__()
    def forward(self, input):
        return torch.erf(input)

class Erfc(nn.Module):
    def __init__(self):
        super(Erfc, self).__init__()
    def forward(self, input):
        return torch.erfc(input)

class Exp(nn.Module):
    def __init__(self):
        super(Exp, self).__init__()
    def forward(self, input):
        return torch.exp(input)

class Floor(nn.Module):
    def __init__(self):
        super(Floor, self).__init__()
    def forward(self, input):
        return torch.floor(input)

class Frac(nn.Module):
    def __init__(self):
        super(Frac, self).__init__()
    def forward(self, input):
        return torch.frac(input)

class Log(nn.Module):
    def __init__(self):
        super(Log, self).__init__()
    def forward(self, input):
        return torch.log(input)

class Log10(nn.Module):
    def __init__(self):
        super(Log10, self).__init__()
    def forward(self, input):
        return torch.log10(input)

class Neg(nn.Module):
    def __init__(self):
        super(Neg, self).__init__()
    def forward(self, input):
        return torch.neg(input)

class Round(nn.Module):
    def __init__(self):
        super(Round, self).__init__()
    def forward(self, input):
        return torch.round(input)

class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()
    def forward(self, input):
        return torch.sin(input)

class Sinh(nn.Module):
    def __init__(self):
        super(Sinh, self).__init__()
    def forward(self, input):
        return torch.sinh(input)

class Tan(nn.Module):
    def __init__(self):
        super(Tan, self).__init__()
    def forward(self, input):
        return torch.tan(input)

class Trunc(nn.Module):
    def __init__(self):
        super(Trunc, self).__init__()
    def forward(self, input):
        return torch.trunc(input)

class GumbelSoftmax(nn.Module):
    def __init__(self):
        super(GumbelSoftmax, self).__init__()
    def forward(self, input):
        return tf.gumbel_softmax(input)


# new activation functions

# From "New activation functions for single layer feedforward neural network", Kocak and Siray
class GeneralizedSwish(nn.Module):
    def __init__(self):
        super(GeneralizedSwish, self).__init__()
    def forward(self, input):
        # zeros = torch.zeros(input.size()).to(DEVICE)
        # return torch.addcmul(zeros, input, torch.sigmoid(torch.exp(-input)), value=1)
        return input*torch.sigmoid(torch.exp(-input))

class SigmoidDerivative(nn.Module):
    def __init__(self):
        super(SigmoidDerivative, self).__init__()
    def forward(self, input):
        # zeros = torch.zeros(input.size()).to(DEVICE)
        # square = torch.addcmul(zeros, torch.sigmoid(input), torch.sigmoid(input), value=1).to(DEVICE)
        # return torch.addcmul(zeros, torch.exp(-input), square, value=1)
        x = torch.sigmoid(input).to(DEVICE)
        return torch.exp(-input)*x*x

# From "Comparison of new activation functions in neural network for forecasting financial time series", Gecynalda, Gomes, Ludermir, and Lima
class CLogLogM(nn.Module):
    def __init__(self):
        super(CLogLogM, self).__init__()
    def forward(self, input):
        # zeros = torch.zeros(input.size()).to(DEVICE)
        # ones = torch.ones(input.size()).to(DEVICE)
        # twos = -2*ones
        # pointseven = -0.7*ones
        # b1 = torch.addcmul(zeros, pointseven, torch.exp(input), value=1).to(DEVICE)
        # b2 = torch.exp(b1).to(DEVICE)
        # b = torch.addcmul(zeros, twos, b2, value=1).to(DEVICE)
        # return torch.add(ones, b)
    
        return 1 - 2*torch.exp(-0.7*torch.exp(input))

# From "Mish: A Self Regularized Non-Monotonic Neural Activation Function", Misra
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    def forward(self, input):
        # zeros = torch.zeros(input.size()).to(DEVICE)
        # return torch.addcmul(zeros, input, torch.tanh(tf.softplus(input)), value=1)
        return input*torch.tanh(tf.softplus(input))
