# -*- coding: utf-8 -*-


"""
The file contains the following utility functions for the application:
    log - Function to print and/or log messages to the console or logging file.
    str_to_bool - Function to convert an input string to a boolean value.
    set_random_seed - Function used to set the random seed for all libraries used to generate random numbers.
"""


# Built-in/Generic Imports
import os
import random
from argparse import ArgumentTypeError, Namespace

# Library Imports
import torch
import numpy as np


__author__    = ["Jacob Carse", "Andres Alvarez Olmo"]
__copyright__ = "Copyright 2022, Calibration"
__credits__   = ["Jacob Carse", "Andres Alvarez Olmo"]
__license__   = "MIT"
__version__   = "1.0.0"
__maintainer  = ["Jacob Carse", "Andres Alvarez Olmo"]
__email__     = ["j.carse@dundee.ac.uk", "alvarezolmoandres@gmail.com"]
__status__    = "Development"


def log(arguments: Namespace, message: str) -> None:
    """
    Logging function that will both print and log an input message.
    :param arguments: ArgumentParser object containing "log_dir" and "experiment".
    :param message: String containing the message to be printed and/or logged.
    """

    # Prints the message to the console if verbose is set to True.
    if arguments.verbose:
        print(message)

    if arguments.log_dir != '':
        # Creates the directory for the log file.
        os.makedirs(arguments.log_dir, exist_ok=True)

        # Logs the message to the log file.
        print(message, file=open(os.path.join(arguments.log_dir, f"{arguments.experiment}_log.txt"), 'a'))


def str_to_bool(argument: str) -> bool or ArgumentTypeError:
    """
    Function to convert a string to a boolean value.
    :param argument: String to be converted.
    :return: Boolean value.
    """

    # Checks if the argument is already a boolean value.
    if isinstance(argument, bool): return argument

    # Returns boolean depending on the input string.
    if argument.lower() in ["true", "t"]:
        return True
    elif argument.lower() in ["false", "f"]:
        return False

    # Returns an error if the value is not converted to a boolean value.
    return ArgumentTypeError(f"Boolean value expected. Got \"{argument}\".")


def set_random_seed(seed: int) -> None:
    """
    Sets the random seed for all libraries that are used to generate random numbers.
    :param seed: Integer for the seed that will be used.
    """

    # Sets the seed for the inbuilt Python functions.
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Sets the seed for the NumPy library.
    np.random.seed(seed)

    # Sets the seed for the PyTorch library.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_device(arguments: Namespace) -> torch.device:
    """
    Sets the device that will be used for training and testing.
    :param arguments: A ArgumentParser Namespace containing gpu.
    :return: A PyTorch device.
    """

    # Checks if the GPU is available to be used and sets the device.
    if arguments.use_gpu and torch.cuda.is_available():
        return torch.device(f"cuda:{0}")

    # Sets the device to CPU.
    else:
        return torch.device("cpu")
