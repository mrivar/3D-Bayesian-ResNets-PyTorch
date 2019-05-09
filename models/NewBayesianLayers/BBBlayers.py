import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from .BBBdistributions import *
from torch.nn.modules.utils import _pair, _triple

cuda = torch.cuda.is_available()

class FlattenLayer(nn.Module):

    def __init__(self, num_features):
        super(FlattenLayer, self).__init__()
        self.num_features = num_features

    def forward(self, x):
        return x.view(-1, self.num_features)


class GaussianVariationalInference(nn.Module):
    def __init__(self, loss=nn.CrossEntropyLoss()):
        super(GaussianVariationalInference, self).__init__()
        self.loss = loss

    def forward(self, logits, y, kl, beta):
        logpy = -self.loss(logits, y)

        ll = logpy - beta * kl  # ELBO
        loss = -ll

        return loss


class BBBLinearFactorial(nn.Module):
    """
    Describes a Linear fully connected Bayesian layer with
    a distribution over each of the weights and biases
    in the layer.
    """
    def __init__(self, in_features, out_features, p_logvar_init=-3, p_pi=1.0, q_logvar_init=math.log(5**2)):
        # p_logvar_init, p_pi can be either
        # (list/tuples): prior model is a mixture of Gaussians components=len(p_pi)=len(p_logvar_init)
        # float: Gussian distribution
        # q_logvar_init: float, the approximate posterior is currently always a factorized gaussian
        super(BBBLinearFactorial, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.p_logvar_init = p_logvar_init
        self.q_logvar_init = q_logvar_init
        # Weight parameters
        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = Parameter(torch.Tensor(out_features, in_features))
        self.weight = GaussianWithRho(self.weight_mu, self.weight_rho)
        # Bias parameters
        self.bias_mu = Parameter(torch.Tensor(out_features))
        self.bias_rho = Parameter(torch.Tensor(out_features))
        self.bias = GaussianWithRho(self.bias_mu, self.bias_rho)
        # Prior distributions
        self.weight_prior = FixedGaussian(mu=0.0, logvar=p_logvar_init)
        #self.bias_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.log_prior = 0
        self.log_variational_posterior = 0
        # initialize all paramaters
        self.reset_parameters()

    def reset_parameters(self):
        # initialize (trainable) approximate posterior parameters
        stdv = 10. / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_rho.data.uniform_(-stdv, stdv).add_(self.q_logvar_init)
        self.bias_mu.data.uniform_(-stdv, stdv)
        self.bias_rho.data.uniform_(-stdv, stdv).add_(self.q_logvar_init)

    @property
    def weight_std(self):
        return torch.log1p(torch.exp(self.weight_rho))

    def forward(self, input, sample=False, calculate_log_probs=False):
        """
        Probabilistic forwarding method.
        :param input: data tensor
        :return: output, kl-divergence
        """
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.logpdf(weight)# + self.bias_prior.logpdf(bias)
            self.log_variational_posterior = self.weight.logpdf(weight) + self.bias.logpdf(bias)
            kl = self.log_variational_posterior - self.log_prior
        else:
            self.log_prior, self.log_variational_posterior, kl = 0, 0, 0

        return F.linear(input, weight, bias), kl

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class _ConvNd(nn.Module):
    """
    Describes a Bayesian convolutional layer with
    a distribution over each of the weights and biases
    in the layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, output_padding, groups,
                 p_logvar_init=-3, p_pi=1.0, q_logvar_init=math.log(5**2),
                 bias=None):
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
        self.bias = bias
        # initialize log variance of p and q
        self.p_logvar_init = p_logvar_init
        self.q_logvar_init = q_logvar_init
        # Weight parameters
        self.weight_mu = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        self.weight_alpha = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        self.weight = GaussianWithLogvar(self.weight_mu, torch.log(self.weight_var))
        # Prior distributions
        self.weight_prior = FixedGaussian(mu=0.0, logvar=p_logvar_init)
        self.log_prior = 0
        self.log_variational_posterior = 0
        # initialize all paramaters
        self.reset_parameters()

    @property
    def weight_var(self):
        return self.weight_alpha*self.weight_mu.pow(2)

    def reset_parameters(self):
        # initialize (trainable) approximate posterior parameters
        stdv = 10. / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_alpha.data.uniform_(-stdv, stdv)

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
                 padding=0, dilation=1, groups=1):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super(BBBConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, _pair(0), groups)

    def forward(self, input):
        # local reparameterization trick for convolutional layer
        qw_mean = F.conv2d(input=input, weight=self.weight_mu, stride=self.stride, padding=self.padding,
                           dilation=self.dilation, groups=self.groups)
        qw_std = torch.sqrt(1e-8 + F.conv2d(input=input.pow(2), weight=self.weight_var,
                                            stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups))

        # sample from output
        if cuda:
            qw_mean.cuda()
            qw_std.cuda()
            output = qw_mean + qw_std * (torch.randn(qw_mean.size())).cuda()
        else:
            output = qw_mean + qw_std * (torch.randn(qw_mean.size()))

        if self.training or calculate_log_probs:
            weight = self.weight.sample()
            self.log_prior = self.weight_prior.logpdf(weight)
            self.log_variational_posterior = self.weight.logpdf(weight)
            kl = self.log_variational_posterior - self.log_prior
        else:
            self.log_prior, self.log_variational_posterior = 0, 0
        return output, kl


class BBBConv3d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1):

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        super(BBBConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, _triple(0), groups)

    def forward(self, input):
        # local reparameterization trick for convolutional layer
        qw_mean = F.conv3d(input=input, weight=self.weight_mu, stride=self.stride, padding=self.padding,
                           dilation=self.dilation, groups=self.groups)
        qw_std = torch.sqrt(1e-8 + F.conv3d(input=input.pow(2), weight=self.weight_var,
                                            stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups))

        # sample from output
        if cuda:
            qw_mean.cuda()
            qw_std.cuda()
            output = qw_mean + qw_std * (torch.randn(qw_mean.size())).cuda()
        else:
            output = qw_mean + qw_std * (torch.randn(qw_mean.size()))

        if self.training or calculate_log_probs:
            weight = self.weight.sample()
            self.log_prior = self.weight_prior.logpdf(weight)
            self.log_variational_posterior = self.weight.logpdf(weight)
            kl = self.log_variational_posterior - self.log_prior
        else:
            self.log_prior, self.log_variational_posterior = 0, 0
        return output, kl
