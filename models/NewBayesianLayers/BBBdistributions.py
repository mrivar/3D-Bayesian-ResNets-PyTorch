import torch
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Distribution(object):
    """
    Base class for torch-based probability distributions.
    """
    def pdf(self, x):
        raise NotImplementedError

    def logpdf(self, x):
        raise NotImplementedError

    def cdf(self, x):
        raise NotImplementedError

    def logcdf(self, x):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class GaussianWithRho(Distribution):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.shape = mu.size()
        self.normal = torch.distributions.Normal(0,1)
    
    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))

    def sample(self):
        epsilon = self.normal.sample(self.shape).to(DEVICE)
        return self.mu + self.sigma * epsilon
    
    def logpdf(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()

    def pdf(self, input):
        return torch.exp(self.logpdf(input))

    def entropy(self):
        return 0.5 * math.log(2. * math.pi * math.e) + 0.5 * torch.log(self.sigma ** 2)


class GaussianWithLogvar(Distribution):
    def __init__(self, mu, logvar):
        super().__init__()
        self.mu = mu
        self.logvar = logvar
        self.shape = mu.size()
        self.normal = torch.distributions.Normal(0,1)

    def sample(self):
        epsilon = self.normal.sample(self.shape).to(DEVICE)
        return self.mu + torch.exp(0.5 * self.logvar) * epsilon
    
    def logpdf(self, input):
        c = - float(0.5 * math.log(2 * math.pi))
        return c - 0.5 * self.logvar - (input - self.mu).pow(2) / (2 * torch.exp(self.logvar))

    def pdf(self, input):
        return torch.exp(self.logpdf(input))

    def entropy(self):
        return 0.5 * math.log(2. * math.pi * math.e) + 0.5 * self.logvar


class GaussianWithVar(Distribution):
    def __init__(self, mu, var):
        super().__init__()
        self.mu = mu
        self.var = var
        self.shape = mu.size()
        self.normal = torch.distributions.Normal(0,1)

    def sample(self):
        epsilon = self.normal.sample(self.shape).to(DEVICE)
        return self.mu + torch.exp(0.5 * torch.log(self.var)) * epsilon
    
    def logpdf(self, input):
        c = - float(0.5 * math.log(2 * math.pi))
        return c - 0.5 * torch.log(self.var) - (input - self.mu).pow(2) / (2 * self.var)

    def pdf(self, input):
        return torch.exp(self.logpdf(input))

    def entropy(self):
        return 0.5 * math.log(2. * math.pi * math.e) + 0.5 * torch.log(self.var)


class FixedGaussian(Distribution):
    def __init__(self, mu, sigma):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
    
    def logpdf(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()


class FixedMixtureGaussian(Distribution):
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0,sigma1)
        self.gaussian2 = torch.distributions.Normal(0,sigma2)
    
    def logpdf(self, input):
        prob1 = torch.exp(self.gaussian1.logpdf(input))
        prob2 = torch.exp(self.gaussian2.logpdf(input))
        return (torch.log(self.pi * prob1 + (1-self.pi) * prob2)).sum()


