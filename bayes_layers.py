# -*- coding: utf-8 -*-


"""
File containing classes used to implement a Bayesian Neural Network Layer.
    GaussianDistribution - Trainable Distribution used for the posterior of a Bayesian layer.
    ScaleMixtureGaussian - Distribution used for the prior of a Bayesian layer.
    BayesianLinearLayer - Bayesian Fully Connected Layer.
"""


# Built-in/Generic Imports
import math

# Library Imports
import torch
from torch import nn
from torch.nn import functional as F


__author__    = ["Jacob Carse", "Andres Alvarez Olmo"]
__copyright__ = "Copyright 2022, Calibration"
__credits__   = ["Jacob Carse", "Andres Alvarez Olmo"]
__license__   = "MIT"
__version__   = "1.0.0"
__maintainer  = ["Jacob Carse", "Andres Alvarez Olmo"]
__email__     = ["j.carse@dundee.ac.uk", "alvarezolmoandres@gmail.com"]
__status__    = "Development"


class GaussianDistribution:
    """
    Trainable Distribution used as parameter for weight and bias in Bayesian layers.
        init - Initialise for the distribution.
        sigma - Calculates the sigma.
        sample_distribution - Samples a parameter from the distribution.
        log_posterior - Calculate the log posterior from a sampled parameter.
    """

    def __init__(self, mu: torch.Tensor, rho: torch.Tensor, device: torch.device) -> None:
        """
        Initialiser for the Distribution that saves parameters and defines a Normal distribution for the parameter.
        :param mu: The mean of the distribution.
        :param rho: The variance of the distribution.
        :param device: The PyTorch device the distribution will be stored on.
        """

        self.mu = mu
        self.rho = rho
        self.device = device
        self.normal = torch.distributions.Normal(0, 1, validate_args=True)

    def sigma(self) -> torch.Tensor:
        """
        Calculates the sigma of the distribution.
        :return: Torch Tensor of the sigma of the distribution.
        """

        return torch.log1p(torch.exp(self.rho))

    def sample_distribution(self) -> torch.Tensor:
        """
        Samples the normal distribution using the mean, variance and sigma.
        :return: Torch Tensor of the sampled parameters.
        """

        e = self.normal.sample(self.rho.size()).to(self.device)
        return self.mu + self.sigma() * e

    def log_posterior(self, input: torch.Tensor) -> float:
        """
        Calculates the log posterior for the sampled parameters.
        :param input: Torch Tensor with the sampled parameters.
        :return: Value with the log posterior.
        """

        return (-math.log(math.sqrt(2 * math.pi)) - torch.log(self.sigma()) -
                ((input - self.mu) ** 2) / (2 * self.sigma() ** 2)).sum()


class ScaleMixtureGaussian:
    """
    ScaleMixture model used as a prior for the weight and bias in a Bayesian Layer.
        init - Initialiser for the Scale Mixture Gaussian Distribution.
        log_prob - Calculates the log prior probability for the sampled parameter.
    """

    def __init__(self, pi: torch.Tensor, sigma_1: torch.Tensor, sigma_2: torch.Tensor, device: torch.device) -> None:
        """
        Initialiser for the Scale Mixture Gaussian Distribution a distribution used as the prior of a Bayesian layer.
        :param pi: Weighting for balancing the two Gaussian distributions.
        :param sigma_1: The variance for the first Gaussian distribution.
        :param sigma_2: The variance for the second Gaussian distribution.
        :param device: The device the distribution will be initialised on.
        """

        self.pi = pi
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.device = device
        self.gaussian_1 = torch.distributions.Normal(torch.tensor(0).to(device), sigma_1, validate_args=True)
        self.gaussian_2 = torch.distributions.Normal(torch.tensor(0).to(device), sigma_2, validate_args=True)

    def log_prob(self, input: torch.Tensor) -> float:
        """
        Calculates the log_likelihood for each parameter sampled relative to a prior distribution.
        :param input: Torch.Tensor of the sampled parameters
        :return: The log probability of the input relative to the prior.
        """

        prob_1 = torch.exp(self.gaussian_1.log_prob(input))
        prob_2 = torch.exp(self.gaussian_2.log_prob(input))

        return (torch.log(self.pi * prob_1 + (1 - self.pi) * prob_2)).sum()


class BayesianLinearLayer(nn.Module):
    """
    Class for a Bayesian Fully Connected Layer that can be used to form a Bayesian Neural Network.
        init - Initialiser for the Bayesian layer that initialises the distributions for weights and biases.
        forward - Forward Propagation for the Bayesian Layer by sampling from the weight and bias distributions.
    """

    def __init__(self, in_features: int, out_features: int, device: torch.device) -> None:
        """
        Initialiser for the Bayesian Layer that initialises the distributions for the weights and biases.
        :param in_features: Integer for the number of input features in the layer.
        :param out_features: Integer for the number of output features in the layer.
        :param device: PyTorch Device the layer will be loaded on.
        """

        # Calls the super for the nn.Module.
        super(BayesianLinearLayer, self).__init__()

        # Saves the number of in and out features in the class.
        self.in_features = in_features
        self.out_features = out_features

        # The parameter initialisation for the weights.
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).normal_(-7, 0.1))
        self.weight = GaussianDistribution(self.weight_mu, self.weight_rho, device)

        # The parameter initialisation for the biases.
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).normal_(-7, 0.1))
        self.bias = GaussianDistribution(self.bias_mu, self.bias_rho, device)

        # The parameter initialisation for the prior distribution.
        sigma_1 = torch.FloatTensor([math.exp(0.1)]).to(device)
        sigma_2 = torch.FloatTensor([math.exp(0.4)]).to(device)
        pi = torch.tensor(0.5).to(device)

        # The initialisation for the weight and bias priors.
        self.weight_prior = ScaleMixtureGaussian(pi, sigma_1, sigma_2, device)
        self.bias_prior = ScaleMixtureGaussian(pi, sigma_1, sigma_2, device)

        # Creates the class member for the log prior and log variational posterior.
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward propagation with the layer by sampling the weights and biases from the distibutions.
        :param x: PyTorch Tensor for the input image batch.
        :return: PyTorch Tensor of logits.
        """

        # Samples the weight and bias from the distibutions.
        weight = self.weight.sample_distribution()
        bias = self.bias.sample_distribution()

        # Calculates the log prior and log variational posterior.
        self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
        self.log_variational_posterior = self.weight.log_posterior(weight) + self.bias.log_posterior(bias)

        # Performs forward propagation with the sampled weight and bias.
        return F.linear(x, weight, bias)
