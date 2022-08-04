# -*- coding: utf-8 -*-


"""
The file contains implementations of the functions used to test a CNN model using different options.
    test_cnn - Function used to test a Convolutional Neural Network.
    test_laplace - Function used to test a Bayesian Neural Network (Laplace Approximation).
    test_temperature - Function used to test a Convolutional Neural Network with temperature scaling.
    test_bnn - Function used to test a Bayesian Neural Network (Bayes by Backprop).
"""


# Library Imports
import laplace
import pandas as pd
from torch.cuda import amp
from torch.nn import functional as F
from torch.utils.data import DataLoader

# Own Modules
from utils import *
from dataset import get_datasets
from temperature import get_temperature
from model import Classifier, BayesByBackpropClassifier


__author__    = ["Jacob Carse", "Andres Alvarez Olmo"]
__copyright__ = "Copyright 2022, Calibration"
__credits__   = ["Jacob Carse", "Andres Alvarez Olmo"]
__license__   = "MIT"
__version__   = "1.0.0"
__maintainer  = ["Jacob Carse", "Andres Alvarez Olmo"]
__email__     = ["j.carse@dundee.ac.uk", "alvarezolmoandres@gmail.com"]
__status__    = "Development"


def test_cnn(arguments: Namespace, device: torch.device) -> None:
    """
    Function for testing the Convolutional Neural Network.
    :param arguments: ArgumentParser Namespace object with arguments used for training.
    :param device: PyTorch device that will be used for training.
    """

    # Loads the testing data.
    _, _, test_data = get_datasets(arguments)

    # Creates the testing data loader using the dataset objects.
    testing_data_loader = DataLoader(test_data, batch_size=arguments.batch_size * 2,
                                      shuffle=False, num_workers=arguments.data_workers,
                                      pin_memory=False, drop_last=False)

    log(arguments, "Loaded Datasets\n")

    # Initialises the classifier model.
    classifier = Classifier(arguments.efficient_net, test_data.num_class, pretrained=False)

    # Loads the trained model.
    classifier.load_state_dict(torch.load(os.path.join(arguments.model_dir, f"{arguments.experiment}_best.pt")))

    # Sets the classifier to evaluation mode.
    classifier.eval()

    # Moves the classifier to the selected device.
    classifier.to(device)

    # Defines the list of data to be used to collect results.
    batch_count = 0
    data_frame = [[] for _ in range(test_data.num_class + 2)]

    # Loops through the testing data batches with no gradient calculations.
    with torch.no_grad():
        for images, labels, file_names in testing_data_loader:
            # Adds to the current batch count.
            batch_count += 1

            # Moves the images to the selected device also moves the labels to the cpu.
            images = images.to(device)
            labels = labels.cpu().numpy()

            # Performs forward propagation using 16 bit precision.
            if arguments.precision == 16 and device != torch.device("cpu"):
                with amp.autocast():
                    logits = classifier(images)

            # Performs forward propagation using 32 bit precision.
            else:
                logits = classifier(images)

            # Gets the predictive probabilities from the neural network.
            predictions = F.softmax(logits, dim=1).cpu().numpy()

            # Adds all information to the dataframe.
            data_frame[0] += list(file_names)
            data_frame[1] += labels.tolist()
            for i in range(test_data.num_class):
                data_frame[2 + i] += predictions[:, i].tolist()

            # If the number of batches have been reached end testing.
            if batch_count == arguments.batches_per_epoch:
                break

    # Creates the output directory for the output files.
    os.makedirs(arguments.output_dir, exist_ok=True)

    # Creates the DataFrame from the output predictions.
    data_frame = pd.DataFrame(data_frame).transpose()

    # Outputs the output DataFrame to a csv file.
    data_frame.to_csv(os.path.join(arguments.output_dir, f"{arguments.experiment}_point.csv"))


def test_laplace(arguments: Namespace, device: torch.device) -> None:
    """
    Function for testing the Laplace Convolutional Neural Network.
    :param arguments: ArgumentParser Namespace object with arguments used for training.
    :param device: PyTorch device that will be used for training.
    """

    # Loads the training and testing data.
    train_data, _, test_data = get_datasets(arguments)

    # Creates the training data loader using the dataset objects.
    training_data_loader = DataLoader(train_data, batch_size=int(4 if arguments.dataset.lower() == "isic" else 16),
                                     shuffle=True, num_workers=arguments.data_workers,
                                     pin_memory=False, drop_last=False)

    # Creates the testing data loader using the dataset objects.
    testing_data_loader = DataLoader(test_data, batch_size=arguments.batch_size * 2,
                                     shuffle=False, num_workers=arguments.data_workers,
                                     pin_memory=False, drop_last=False)

    log(arguments, "Loaded Datasets\n")

    # Initialises the classifier model.
    classifier = Classifier(arguments.efficient_net, test_data.num_class, pretrained=False)

    # Loads the trained model.
    classifier.load_state_dict(torch.load(os.path.join(arguments.model_dir, f"{arguments.experiment}_best.pt")))

    # Sets the classifier to evaluation mode.
    classifier.eval()

    # Moves the classifier to the selected device.
    classifier.to(device)

    # Sets up the Laplace Approximation model using the trained model.
    la = laplace.Laplace(classifier, "classification", subset_of_weights="last_layer", hessian_structure="full")

    log(arguments, "Fitting Laplace Approximation to training data")

    # Fits Laplace Approximation using 16 bit precision.
    if arguments.precision == 16 and device != torch.device("cpu"):
        with amp.autocast():
            # Fits the Laplace Approximation to the training data.
            la.fit(training_data_loader)

            log(arguments, "Optimising Prior Precision")

            # Optimises the prior precision using Marginal-likelihood.
            la.optimize_prior_precision(method='marglik')

    # Fits Laplace Approximation using 32 bit precision.
    else:
        # Fits the Laplace Approximation to the training data.
        la.fit(training_data_loader)

        log(arguments, "Optimising Prior Precision")

        # Optimises the prior precision using Marginal-likelihood.
        la.optimize_prior_precision(method='marglik')

    # Defines the list of data to be used to collect results.
    batch_count = 0
    data_frame = [[] for _ in range(test_data.num_class + 2)]

    # Loops through the testing data batches with no gradient calculations.
    with torch.no_grad():
        for images, labels, file_names in testing_data_loader:
            # Adds to the current batch count.
            batch_count += 1

            # Moves the images to the selected device also moves the labels to the cpu.
            images = images.to(device)
            labels = labels.cpu().numpy()

            # Gets the predictive samples from the laplace model.
            predictions = la.predictive_samples(images, n_samples=arguments.testing_samples)
            predictions = torch.swapaxes(predictions, 0, 1)

            # Moves the predictions to the CPU.
            predictions = predictions.cpu().numpy()

            # Averages the Bayesian predictions.
            predictions = np.mean(predictions, axis=1)

            # Adds all information to the dataframe.
            data_frame[0] += list(file_names)
            data_frame[1] += labels.tolist()
            for i in range(test_data.num_class):
                data_frame[2 + i] += predictions[:, i].tolist()

            # If the number of batches have been reached end testing.
            if batch_count == arguments.batches_per_epoch:
                break

    # Creates the output directory for the output files.
    os.makedirs(arguments.output_dir, exist_ok=True)

    # Creates the DataFrame from the output predictions.
    data_frame = pd.DataFrame(data_frame).transpose()

    # Outputs the output DataFrame to a csv file.
    data_frame.to_csv(os.path.join(arguments.output_dir, f"{arguments.experiment}_laplace.csv"))


def test_temperature(arguments: Namespace, device: torch.device, temperature: str = "cross_entropy") -> None:
    """
    Function for testing the Convolutional Neural Network.
    :param arguments: ArgumentParser Namespace object with arguments used for training.
    :param device: PyTorch device that will be used for training.
    :param temperature: String for the mode used to optimise the temperature.
    """

    # Loads the validation and testing data.
    _, val_data, test_data = get_datasets(arguments)

    # Creates the validation data loader using the dataset objects.
    val_data_loader = DataLoader(val_data, batch_size=arguments.batch_size * 2,
                                 shuffle=False, num_workers=arguments.data_workers,
                                 pin_memory=False, drop_last=False)

    # Creates the testing data loader using the dataset objects.
    testing_data_loader = DataLoader(test_data, batch_size=arguments.batch_size * 2,
                                     shuffle=False, num_workers=arguments.data_workers,
                                     pin_memory=False, drop_last=False)

    log(arguments, "Loaded Datasets\n")

    # Initialises the classifier model.
    classifier = Classifier(arguments.efficient_net, test_data.num_class, pretrained=False)

    # Loads the trained model.
    classifier.load_state_dict(torch.load(os.path.join(arguments.model_dir, f"{arguments.experiment}_best.pt")))

    # Sets the classifier to evaluation mode.
    classifier.eval()

    # Moves the classifier to the selected device.
    classifier.to(device)

    # Gets the temperature using a specified mode.
    temp_value = get_temperature(arguments, classifier, val_data_loader, device, temperature)

    # Defines the list of data to be used to collect results.
    batch_count = 0
    data_frame = [[] for _ in range(test_data.num_class + 2)]

    # Loops through the testing data batches with no gradient calculations.
    with torch.no_grad():
        for images, labels, file_names in testing_data_loader:
            # Adds to the current batch count.
            batch_count += 1

            # Moves the images to the selected device also moves the labels to the cpu.
            images = images.to(device)
            labels = labels.cpu().numpy()

            # Performs forward propagation using 16 bit precision.
            if arguments.precision == 16 and device != torch.device("cpu"):
                with amp.autocast():
                    logits = classifier(images)

            # Performs forward propagation using 32 bit precision.
            else:
                logits = classifier(images)

            # Scales the logits using the provided temperature.
            logits = torch.div(logits, temp_value)

            # Gets the predictive probabilities from the neural network.
            predictions = F.softmax(logits, dim=1).cpu().numpy()

            # Adds all information to the dataframe.
            data_frame[0] += list(file_names)
            data_frame[1] += labels.tolist()
            for i in range(test_data.num_class):
                data_frame[2 + i] += predictions[:, i].tolist()

            # If the number of batches have been reached end testing.
            if batch_count == arguments.batches_per_epoch:
                break

    # Creates the output directory for the output files.
    os.makedirs(arguments.output_dir, exist_ok=True)

    # Creates the DataFrame from the output predictions.
    data_frame = pd.DataFrame(data_frame).transpose()

    # Outputs the output DataFrame to a csv file.
    data_frame.to_csv(os.path.join(arguments.output_dir, f"{arguments.experiment}_{temperature}.csv"))


def test_bnn(arguments, device) -> None:
    """
    Function for testing the Bayesian Neural Network.
    :param arguments: ArgumentParser Namespace object with arguments used for training.
    :param device: PyTorch device that will be used for training.
    """

    # Loads the testing data.
    _, _, test_data = get_datasets(arguments)

    # Creates the testing data loader using the dataset objects.
    testing_data_loader = DataLoader(test_data, batch_size=arguments.batch_size * 2,
                                     shuffle=False, num_workers=arguments.data_workers,
                                     pin_memory=False, drop_last=False)

    log(arguments, "Loaded Datasets\n")

    # Initialises the classifier model.
    classifier = BayesByBackpropClassifier(arguments.efficient_net, test_data.num_class,
                                           pretrained=False, device=device)

    # Loads the trained model.
    classifier.load_state_dict(torch.load(os.path.join(arguments.model_dir, f"{arguments.experiment}_best.pt")))

    # Sets the classifier to evaluation mode.
    classifier.eval()

    # Moves the classifier to the selected device.
    classifier.to(device)

    # Defines the list of data to be used to collect results.
    batch_count = 0
    data_frame = [[] for _ in range(test_data.num_class + 2)]

    # Loops through the testing data batches with no gradient calculations.
    with torch.no_grad():
        for images, labels, file_names in testing_data_loader:
            # Adds to the current batch count.
            batch_count += 1

            # Moves the images to the selected device also moves the labels to the cpu.
            images = images.to(device)
            labels = labels.cpu().numpy()

            # Declares a list for Monte-Carlo Samples.
            predictions = []

            # Performs Monte-Carlo Sampling using 16 bit precision.
            if arguments.precision == 16 and device != torch.device("cpu"):
                with amp.autocast():
                    for _ in range(arguments.testing_samples):
                        output = classifier(images)
                        predictions.append(output)

            # Performs Monte-Carlo Sampling using 32 bit precision.
            else:
                for _ in range(arguments.testing_samples):
                    output = classifier(images)
                    predictions.append(output)

            # Stacks the Monte-Carlo samples into a single tensor.
            predictions = torch.stack(predictions)
            predictions = np.moveaxis(F.softmax(predictions, dim=2).cpu().numpy(), 0, 1)

            # Finds the mean of predictive samples.
            predictions = np.mean(predictions, axis=1)

            # Adds all information to the dataframe.
            data_frame[0] += list(file_names)
            data_frame[1] += labels.tolist()
            for i in range(test_data.num_class):
                data_frame[2 + i] += predictions[:, i].tolist()

            # If the number of batches have been reached end testing.
            if batch_count == arguments.batches_per_epoch:
                break

    # Creates the output directory for the output files.
    os.makedirs(arguments.output_dir, exist_ok=True)

    # Creates the DataFrame from the output predictions.
    data_frame = pd.DataFrame(data_frame).transpose()

    # Outputs the output DataFrame to a csv file.
    data_frame.to_csv(os.path.join(arguments.output_dir, f"{arguments.experiment}_bbb.csv"))
