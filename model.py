# -*- coding: utf-8 -*-


"""
The file for the definition of the SelectiveNet and Classifier models.
    Classifier - Class for a EfficientNet Classifier Model.
    BayesByBackpropClassifier - Class for a EfficientNet Bayes By Backprop Model.
"""


# Built-in/Generic Imports
import os

# Library Imports
import torch
import torch.nn as nn
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet

# Own Modules
from bayes_layers import BayesianLinearLayer


__author__    = ["Jacob Carse", "Andres Alvarez Olmo"]
__copyright__ = "Copyright 2022, Calibration"
__credits__   = ["Jacob Carse", "Andres Alvarez Olmo"]
__license__   = "MIT"
__version__   = "1.0.0"
__maintainer  = ["Jacob Carse", "Andres Alvarez Olmo"]
__email__     = ["j.carse@dundee.ac.uk", "alvarezolmoandres@gmail.com"]
__status__    = "Development"


class Classifier(nn.Module):
    """
    Class for the Classifier model that uses an EfficientNet encoder.
        init - Initialiser for the model.
        forward - Performs forward propagation.
        save_model - Saves the model.
    """

    def __init__(self, b: int = 0, class_num: int = 2, pretrained: bool = True) -> None:
        """
        Initialiser for the model that initialises the model's layers.
        :param b: The compound coefficient of the EfficientNet model to be loaded.
        :param class_num: The number of classes the model will be predicting.
        :param pretrained: Boolean if the pretrained weights should be loaded.
        """

        # Calls the super for the nn.Module.
        super(Classifier, self).__init__()

        # Loads the EfficientNet encoder.
        if pretrained:
            self.encoder = EfficientNet.from_pretrained(f"efficientnet-b{str(b)}")
        else:
            self.encoder = EfficientNet.from_name(f"efficientnet-b{str(b)}")

        # Defines the Pooling Layer for the Encoder outputs.
        self.encoder_pool = nn.AdaptiveAvgPool2d(1)

        # Defines the hidden Fully Connected Layer.
        self.hidden = nn.Linear(self.encoder._fc.in_features, 512)

        # Defines the output Fully Connected Layer.
        self.classifier = nn.Linear(512, class_num)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward propagation with the Classifier.
        :param x: Input image batch.
        :return: PyTorch Tensor of logits.
        """

        # Performs forward propagation with the encoder.
        x = self.encoder.extract_features(x)
        x = self.encoder_pool(x)
        x = x.view(x.shape[0], -1)

        # Performs forward propagation with the hidden layer.
        x = F.silu(self.hidden(x))

        # Get the output logits from the output layer.
        return self.classifier(x)

    def save_model(self, path: str, name: str, epoch: str = "best") -> None:
        """
        Method for saving the model.
        :param path: Directory path to save the model.
        :param name: The name of the experiment to be saved.
        :param epoch: Integer for the current epoch to be included in the save name.
        """

        # Checks if the save directory exists and if not creates it.
        os.makedirs(path, exist_ok=True)

        # Saves the model to the save directory.
        torch.save(self.state_dict(), os.path.join(path, f"{name}_{epoch}.pt"))


class BayesByBackpropClassifier(nn.Module):
    """
    Class for the Bayes By Backprop model that uses an EfficientNet encoder.
        init - Initialises for the model that initialises the model.
        forward - Performs forward propagation with the model.
        sample_elbo - Samples the Evidence Lower Bound from multiple samples.
        save_model - Saves the model.
    """

    def __init__(self, b: int = 0, class_num: int = 2, pretrained: bool = True,
                 device: torch.device = torch.device("cpu")) -> None:
        """
        Initialiser for the model that initialises the models layers.
        :param b: The compound coefficient of the EfficientNet model to be loaded.
        :param class_num: The number of classes to be predicted.
        :param pretrained: Boolean for if the EfficientNet encoder should be pretrained.
        :param device: The PyTorch Device that the model will be loaded on.
        """

        # Calls the super for the nn.Module.
        super(BayesByBackpropClassifier, self).__init__()

        # Saves the device and class num in the class.
        self.device = device
        self.class_num = class_num

        # Loads the EfficientNet encoder.
        if pretrained:
            self.encoder = EfficientNet.from_pretrained(f"efficientnet-b{str(b)}")
        else:
            self.encoder = EfficientNet.from_name(f"efficientnet-b{str(b)}")

        # Defines the Pooling Layer for the Encoder outputs.
        self.encoder_pool = nn.AdaptiveAvgPool2d(1)

        # Defines the hidden Bayesian Layer.
        self.hidden = BayesianLinearLayer(self.encoder._fc.in_features, 512, device)

        # Defines the output Bayesian Layer.
        self.classifier = BayesianLinearLayer(512, class_num, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward propagation with the Bayes by Backprop Model.
        :param x: PyTorch Tensor for the input image batch.
        :return: PyTorch Tensor of logits.
        """

        # Performs forward propagation with the en
        x = self.encoder.extract_features(x)
        x = self.encoder_pool(x)
        x = x.view(x.shape[0], -1)

        # Performs forward propagation with the hidden layer.
        x = F.silu(self.hidden(x))

        # Gets the output logits from the output layer.
        return self.classifier(x)

    def sample_elbo(self, x: torch.Tensor, samples: int = 1) -> (torch.Tensor, torch.Tensor):
        """
        Performs multiple iterations with the model to sample the Evidence Lower Bound.
        :param x: PyTorch Tensor for the input image batch.
        :param samples: Integer for the number of samples used to sample the ELBO.
        :return: PyTorch Tensors for the ELBO loss and the average output from the samples.
        """

        # Initialises the output, log priors and log variational posteriors.
        outputs, log_priors, log_variational_posteriors = [], [], []

        # Performs forward propagation with the encoder.
        x = self.encoder.extract_features(x)
        x = self.encoder_pool(x)
        x = x.view(x.shape[0], -1)

        # Gets the outputs, log priors and log variational posteriors from the samples.
        for i in range(samples):
            outputs.append(self.classifier(F.silu(self.hidden(x))))
            log_priors.append(self.hidden.log_prior + self.classifier.log_prior)
            log_variational_posteriors.append(self.hidden.log_variational_posterior +
                                              self.classifier.log_variational_posterior)

        # Gets the mean do the outputs, log priors and log variational posteriors.
        outputs = torch.stack(outputs).mean(0)
        log_prior = torch.squeeze(torch.stack(log_priors)).mean()
        log_variational_posterior = torch.squeeze(torch.stack(log_variational_posteriors)).mean()

        # Calculates the ELBO loss using the KL divergence.
        kl_divergence = log_variational_posterior - log_prior
        loss = kl_divergence / x.size()[0]

        # Returns the outputs and the loss.
        return loss, outputs

    def save_model(self, path: str, name: str, epoch: str = "best") -> None:
        """
        Method for saving the model.
        :param path: Directory path to save the model.
        :param name: The name of the experiment to be saved.
        :param epoch: Integer for the current epoch to be included in the save name.
        """

        # Checks if the save directory exists and if not creates it.
        os.makedirs(path, exist_ok=True)

        # Saves the model to the save directory.
        torch.save(self.state_dict(), os.path.join(path, f"{name}_{epoch}.pt"))
