#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
Executable for calibration experiments.
Can be adjusted using either the configuration file or command line arguments.
"""


# Own Modules Imports
from utils import *
from config import *
from train_bbb import train_bnn
from train_cnn import train_cnn
from testing import test_cnn, test_bnn, test_laplace, test_temperature, grid_search_temp


__author__    = ["Jacob Carse", "Andres Alvarez Olmo"]
__copyright__ = "Copyright 2022, Calibration"
__credits__   = ["Jacob Carse", "Andres Alvarez Olmo"]
__license__   = "MIT"
__version__   = "1.0.0"
__maintainer  = ["Jacob Carse", "Andres Alvarez Olmo"]
__email__     = ["j.carse@dundee.ac.uk", "alvarezolmoandres@gmail.com"]
__status__    = "Development"


if __name__ == "__main__":
    # Loads the arguments from configurations file and command line.
    description = "Experiments on Calibration for Medical Images"
    arguments = load_configurations(description)

    # Displays the loaded arguments.
    log(arguments, "Loaded Arguments:")
    print_arguments(arguments)

    # Sets the random seed if specified.
    if arguments.seed != -1:
        set_random_seed(arguments.seed)
        log(arguments, f"Set Random Seed to {arguments.seed}")

    # Sets the default device to be used.
    device = get_device(arguments)
    log(arguments, f"Device set to {device}\n")

    # Trains and tests the CNN model.
    if arguments.task == "cnn":
        train_cnn(arguments, device)
        test_cnn(arguments, device)
        test_laplace(arguments, device)
        test_temperature(arguments, device, "cross_entropy")
        test_temperature(arguments, device, "ece")
        test_temperature(arguments, device, "mce")
        test_temperature(arguments, device, "combine_metric_1")
        test_temperature(arguments, device, "combine_metric_2")
        test_temperature(arguments, device, "combine_metric_3")
        test_temperature(arguments, device, "combine_all")
        test_temperature(arguments, device, "combine_temp")

    # Trains the CNN model.
    elif arguments.task == "train_cnn":
        train_cnn(arguments, device)

    # Test the CNN model.
    elif arguments.task == "test_cnn":
        test_cnn(arguments, device)
        test_laplace(arguments, device)
        test_temperature(arguments, device, "cross_entropy")
        test_temperature(arguments, device, "ece")
        test_temperature(arguments, device, "mce")
        test_temperature(arguments, device, "combine_metric_1")
        test_temperature(arguments, device, "combine_metric_2")
        test_temperature(arguments, device, "combine_metric_3")
        test_temperature(arguments, device, "combine_all")
        test_temperature(arguments, device, "combine_temp")

    # Trains and tests the Bayesian CNN.
    elif arguments.task == "bnn":
        train_bnn(arguments, device)
        test_bnn(arguments, device)

    # Trains the Bayesian CNN.
    elif arguments.task == "train_bnn":
        train_bnn(arguments, device)

    # Tests the Bayesian CNN.
    elif arguments.task == "test_bnn":
        test_bnn(arguments, device)

    # Asks for a valid task.
    else:
        log(arguments, "Select a correct task.")
