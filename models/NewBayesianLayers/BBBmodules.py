from torch import nn

class BBBSequential(nn.Sequential):

    def forward(self, input):
        kl = 0
        for module in self._modules.values():
            try:
                input, _kl = module(input)
                kl += _kl
            except:
                input = module(input)
        return input, kl
