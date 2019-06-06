import math
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
from .BBBdistributions import Normal, Normalout, distribution_selector
from torch.nn.modules.utils import _pair
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FlattenLayer(nn.Module):

    def __init__(self, num_features):
        super(FlattenLayer, self).__init__()
        self.num_features = num_features

    def forward(self, x):
        return x.view(-1, self.num_features)


class BBBModule(nn.Module):
    _Var = lambda x: Variable(torch.from_numpy(x).type(torch.FloatTensor))

    def __init__(self):
        super(BBBModule, self).__init__()
        # Instantiate a large Gaussian block to sample from, much faster than generating random sample every time
        self._gaussian_block = np.random.randn(10000)

    def _random(self, shape):
        Z_noise = np.random.choice(self._gaussian_block, size=shape)
        Z_noise = np.expand_dims(Z_noise, axis=1).reshape(*shape)
        return BBBModule._Var(Z_noise)

    @staticmethod
    def kl_divergence(mu1, std1, mu2=0, std2=0.05):
        if mu2==0:
            return (torch.log(torch.div(std2,std1)) + (torch.pow(std1, 2) + torch.pow(mu1, 2))/(2*torch.pow(std2, 2)) - .5).sum()
        return (torch.log(torch.div(std2,std1)) + (torch.pow(std1, 2) + torch.pow((mu1-mu2), 2))/(2*torch.pow(std2, 2)) - .5).sum()

    def kl_divergence_fixed_gaussian(self, mu, std):
        return (self.pw_logstd - torch.log(std) + (torch.pow(std, 2) + torch.pow(mu, 2))/(2*torch.pow(self.pw_std, 2)) - .5).sum()

    def kl_divergence_fixed_mixture_gaussian(self, mu, std):
        return .5 * (self.pw_logstd - torch.log(std) -1 + self.pw_trinvstd * std + mu**2 * self.pw_trinvstd).sum()


class _ConvNd(BBBModule):
    """
    Describes a Bayesian convolutional layer with
    a distribution over each of the weights and biases
    in the layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, output_padding, groups,
                 p_logvar_init=[-0, -6], p_pi=[0.5, 0.5], q_logvar_init=-3,
                 kl_calc=False, bias=False):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.groups = groups
        self.kl_calc = kl_calc
        self.bias = bias

        # initialize log variance of p and q
        self.p_logvar_init = p_logvar_init
        self.q_logvar_init = q_logvar_init

        # ...and output...
        self.qw_mu = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        self.qw_log_alpha = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))

        self.qw_mu_b, self.qw_alpha_b = None, None
        if self.bias:
            self.qw_mu_b = Parameter(torch.zeros(out_channels))
            self.qw_alpha_b = Parameter(torch.zeros(out_channels))
        # prior model
        # (does not have any trainable parameters so we use fixed normal or fixed mixture normal distributions)
        # self.pw = distribution_selector(mu=0.0, logvar=p_logvar_init, pi=p_pi)
        # self.pb = distribution_selector(mu=0.0, logvar=p_logvar_init, pi=p_pi)
        if len(p_logvar_init) == 1:
            self.pw_std = Variable(torch.tensor(math.sqrt(math.exp(p_logvar_init))))
            self.pw_logstd = torch.log(self.pw_std)
            self.pw_trinvstd = 0
            self.kl = kl_divergence_fixed_gaussian
        else:
            self.pw_std = Variable(torch.tensor(np.diag([math.exp(logvar) for logvar in p_logvar_init])))
            self.pw_logstd = torch.log(torch.det(self.pw_std))
            self.pw_trinvstd = torch.trace(torch.inverse(self.pw_std))
            self.kl = self.kl_divergence_fixed_mixture_gaussian


        # initialize all parameters
        self.reset_parameters()

    @property
    def qw_var(self):
        return torch.exp(self.qw_log_alpha) * self.qw_mu.pow(2)

    def reset_parameters(self):
        # initialise (learnable) approximate posterior parameters
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.qw_mu.data.uniform_(-stdv, stdv)
        self.qw_log_alpha.data.fill_(self.q_logvar_init)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class BBBConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, **kwargs):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super(BBBConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, _pair(0), groups, **kwargs)

    def forward(self, input):
        return self.convprobforward(input)

    def convprobforward(self, input):
        """
        Convolutional probabilistic forwarding method.
        :param input: data tensor
        :return: output, KL-divergence
        """
        # local reparameterization trick for convolutional layer
        qw_mean = F.conv2d(input=input, weight=self.qw_mu, bias=self.qw_mu_b, stride=self.stride, padding=self.padding,
                           dilation=self.dilation, groups=self.groups).to(DEVICE)
        qw_std = torch.sqrt(1e-8 + F.conv2d(input=input.pow(2), weight=self.qw_var, bias=self.qw_alpha_b,
                                            stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)).to(DEVICE)

        # sample from output
        output = qw_mean + qw_std * (self._random(qw_std.size())).to(DEVICE)
        output.to(DEVICE)


        # KL divergence
        kl = 0
        if self.training and self.kl_calc:
            #kl = self.complexity_cost(output, qw_mean, qw_std, 0, self.pw_std)
            kl = self.kl(qw_mean, qw_std)

        return output, kl


class BBBLinearFactorial(BBBModule):
    """
    Describes a Linear fully connected Bayesian layer with
    a distribution over each of the weights and biases
    in the layer.
    """
    def __init__(self, in_features, out_features,
        p_logvar_init=[-0, -6], p_pi=[0.5, 0.5], q_logvar_init=-3,
        kl_calc=False, bias=False):
        # p_logvar_init, p_pi can be either
        # (list/tuples): prior model is a mixture of Gaussians components=len(p_pi)=len(p_logvar_init)
        # float: Gussian distribution
        # q_logvar_init: float, the approximate posterior is currently always a factorized gaussian
        super(BBBLinearFactorial, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.p_logvar_init = p_logvar_init
        self.q_logvar_init = q_logvar_init
        self.bias = bias

        self.kl_calc = kl_calc

        # Approximate posterior parameters...
        self.qw_mu = Parameter(torch.Tensor(out_features, in_features))
        self.qw_rho = Parameter(torch.Tensor(out_features, in_features))

        self.qw_mu_b, self.qw_rho_b = None, None
        if self.bias:
            self.qw_mu_b = Parameter(torch.zeros(out_features))
            self.qw_rho_b = Parameter(torch.zeros(out_features))

        # Approximate posterior
        # self.qw = NormalRho(mu=self.qw_mean, rho=self.qw_rho)

        # prior model
        # self.pw = distribution_selector(mu=0.0, logvar=p_logvar_init, pi=p_pi)
        if len(p_logvar_init) == 1:
            self.pw_std = Variable(torch.tensor(math.sqrt(math.exp(p_logvar_init))))
            self.pw_logstd = torch.log(self.pw_std)
            self.pw_invstd = 0
            self.kl = kl_divergence_fixed_gaussian
        else:
            self.pw_std = Variable(torch.tensor(np.diag([math.sqrt(math.exp(logvar)) for logvar in p_logvar_init])))
            self.pw_logstd = torch.log(torch.det(self.pw_std))
            self.pw_trinvstd = torch.trace(torch.inverse(self.pw_std))
            self.kl = self.kl_divergence_fixed_mixture_gaussian

        # initialize all paramaters
        self.reset_parameters()

    @property
    def qw_std(self):
        return torch.log1p(torch.exp(self.qw_rho))

    @property
    def qw_std_b(self):
        if self.bias:
            return torch.log1p(torch.exp(self.qw_rho_b))
        return None
     
    def reset_parameters(self):
        # initialize (trainable) approximate posterior parameters
        stdv = 1. / math.sqrt(self.in_features)
        self.qw_mu.data.uniform_(-stdv, stdv)
        self.qw_rho.data.fill_(self.q_logvar_init)

    def forward(self, input):
        return self.fcprobforward(input)

    def fcprobforward(self, input):
        """
        Probabilistic forwarding method.
        :param input: data tensor
        :return: output, kl-divergence
        """
        sample = self.qw_mu + self.qw_std * (self._random(self.qw_rho.size())).to(DEVICE)
        sample_bias = None
        if self.bias: sample_bias = self.qw_mu_b + self.qw_std_b * (self._random(self.qw_rho_b.size())).to(DEVICE)
        output = F.linear(input=input, weight=sample, bias=sample_bias).to(DEVICE)
        #qw_mean = F.linear(input=input, weight=self.qw_mu).to(DEVICE)
        #qw_std = torch.sqrt(1e-8 + F.linear(input=input.pow(2), weight=torch.log1p(torch.exp(self.qw_rho)).pow(2))).to(DEVICE)
        #output = qw_mean + qw_std * (self._random(qw_mean.size())).to(DEVICE)
        #output.to(DEVICE)

        # KL divergence
        kl = 0
        if self.training and self.kl_calc:
            #kl = self.complexity_cost(output, self.qw_mu, self.qw_std, 0, self.pw_std)
            kl = self.kl(self.qw_mu, self.qw_std)

        return output, kl

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GaussianVariationalInference(nn.Module):
    def __init__(self, loss=nn.CrossEntropyLoss()):
        super(GaussianVariationalInference, self).__init__()
        self.loss = loss

    def forward(self, logits, y, kl, beta):
        logpy = self.loss(logits, y)

        ll = logpy + beta * kl  # ELBO

        return ll, logpy
