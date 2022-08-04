# -*- coding: utf-8 -*-


"""
The file contains the code used for handling either the ISIC or PCAM dataset used to train and test the model.
    Dataset - Class for handling the dynamic loading and augmentation of data.
    get_datasets - Function to load the training, validation and testing datasets.
"""


# Built-in/Generic
import os
from argparse import Namespace

# Library Imports
import torch
import numpy as np
import pandas as pd
from torch.utils import data
from PIL import Image, ImageFile
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# Own Modules
from utils import log


__author__    = ["Jacob Carse", "Andres Alvarez Olmo"]
__copyright__ = "Copyright 2022, Calibration"
__credits__   = ["Jacob Carse", "Andres Alvarez Olmo"]
__license__   = "MIT"
__version__   = "1.0.0"
__maintainer  = ["Jacob Carse", "Andres Alvarez Olmo"]
__email__     = ["j.carse@dundee.ac.uk", "alvarezolmoandres@gmail.com"]
__status__    = "Development"


class Dataset(data.Dataset):
    """
    Class for handling the dataset used for training and testing.
        init - The initialiser for the class.
        len - Gets the size of the dataset.
        getitem - Gets an individual item from the dataset by index.
        num_class - Gets the number of classes of the dataset.
        augment - Method to augment an input image.
    """

    def __init__(self, arguments: Namespace, mode: str, df: pd.DataFrame) -> None:
        """
        Initialiser for the class that stores the filenames and labels used to load the images.
        :param arguments: ArgumentParser Namespace containing arguments.
        :param mode: String specifying the type of data loaded "train", "validation" and "test".
        :param df: Pandas DataFrame containing image and label columns.
        """

        # Calls the PyTorch Dataset Initialiser.
        super(Dataset, self).__init__()

        # Stores the arguments and mode in the object.
        self.arguments = arguments
        self.mode = mode

        # Sets the Pillow library to load truncated images.
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        # Stores the dataset data frame in the object.
        self.df = df

    def __len__(self) -> int:
        """
        Gets the length of the dataset.
        :return: Integer for the length of the dataset.
        """

        return self.df.shape[0]

    def __getitem__(self, index: int) -> (torch.Tensor, int):
        """
        Gets a given image and label from the datasets based on a given index.
        :param index: Integer representing the index of the data from the dataset.
        :return: A PyTorch Tensor with the augmented image, an Integer for the label and a filename for the image.
        """

        # Loads and augments the image.
        df_row = self.df.iloc[index]
        image = Image.open(df_row["image"])
        image = self.augment(image)

        # If training returns image and label.
        if self.mode == "train":
            return image, df_row["label"]

        # Returns the image, label and file name.
        return image, df_row["label"], df_row["image"]

    @property
    def num_class(self) -> int:
        """
        Method to return the number of classes in the loaded dataset.
        :return: Integer for the number of classes
        """

        return 8 if self.arguments.dataset.lower() == "isic" else 2

    def augment(self, image: Image) -> torch.Tensor:
        """
        Method for augmenting a given input image into a tensor.
        :param image: A Pillow Image.
        :return: An augmented image Tensor.
        """

        # Mean and Standard Deviation for normalising the dataset.
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

        # Crops the image into the square before resizing.
        if image.width != image.height and self.arguments.square_image:
            offset = int(abs(image.width - image.height) / 2)
            if image.width > image.height:
                image = image.crop([offset, 0, image.width - offset, image.height])
            else:
                image = image.crop([0, offset, image.width, image.height - offset])

        # Declares the list of standard transforms for the input image.
        augmentations = [transforms.Resize((self.arguments.image_x, self.arguments.image_y), InterpolationMode.LANCZOS),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=mean, std=std)]

        # Adds additional transformations if selected.
        if self.arguments.augmentation and self.mode == "train":
            # Class for Random 90 degree rotations.
            class RandomRotation:
                def __init__(self, angles): self.angles = angles

                def __call__(self, x, *args, **kwargs):
                    return transforms.functional.rotate(x, float(np.random.choice(self.angles)))

            # Adds the additional augmentations to the list of augmentations.
            augmentations = augmentations[:1] + [transforms.RandomVerticalFlip(),
                                                 transforms.RandomHorizontalFlip(),
                                                 RandomRotation([0, 90, 180, 270])] + augmentations[1:]

        # Applies the augmentations to the image.
        return transforms.Compose(augmentations)(image)


def get_datasets(arguments: Namespace) -> (Dataset, Dataset, Dataset):
    """
    Loads the ISIC or PCAM datasets and creates Dataset object for training, validation and testing.
    :param arguments: ArgumentParser Namespace containing arguments for data loading.
    :return: Three Dataset objects for training, validation and testing.
    """

    # Loads ISIC dataset.
    if arguments.dataset.lower() == "isic":
        # Reads the ISIC dataset csv file containing filenames and labels.
        df = pd.read_csv(os.path.join(arguments.dataset_dir, "ISIC_2019_Training_GroundTruth.csv"))

        # Gets the directory of the ISIC images.
        data_base = os.path.join(arguments.dataset_dir, "ISIC_2019_Training_Input")

        # Gets the full filenames and labels of the ISIC data.
        filenames = [os.path.join(data_base, x + ".jpg") for x in df["image"].tolist()]
        labels = np.argmax(df.drop(["image", "UNK"], 1).to_numpy(), axis=1)

        # Creates a DataFrame with the filenames and labels.
        df = pd.DataFrame([filenames, labels]).transpose()
        df.columns = ["image", "label"]

        # Gets the indices of all the data in the dataset.
        indices = np.array(range(df.shape[0]))

        # Shuffles the ISIC data.
        random_generator = np.random.default_rng(arguments.seed)
        random_generator.shuffle(indices)

        # Split data indices into training, testing and validation sets.
        split_point_1 = int(indices.shape[0] * arguments.test_split)
        split_point_2 = int(indices.shape[0] * (arguments.val_split + arguments.test_split))
        test_indices = indices[0:split_point_1]
        val_indices = indices[split_point_1:split_point_2]
        train_indices = indices[split_point_2::]

        # Creates the DataFrames for each of the data splits.
        train_df = df.take(train_indices)
        val_df = df.take(val_indices)
        test_df = df.take(test_indices)

    # Loads the PCAM dataset.
    elif arguments.dataset.lower() == "pcam":
        # Reads the PCAM dataset csv file containing filenames and labels.
        train_df = pd.read_csv(os.path.join(arguments.dataset_dir, "train.csv"), names=["image", "label"])
        test_df = pd.read_csv(os.path.join(arguments.dataset_dir, "test.csv"), names=["image", "label"])

        # Adds the full directory for each image.
        train_df["image"] = train_df["image"].apply(lambda x: os.path.join(arguments.dataset_dir, x))
        test_df["image"] = test_df["image"].apply(lambda x: os.path.join(arguments.dataset_dir, x))

        # Gets the indices of all the data in the dataset.
        indices = np.array(range(train_df.shape[0]))

        # Shuffles the PCAM data.
        random_generator = np.random.default_rng(arguments.seed)
        random_generator.shuffle(indices)

        # Split data indices into training and validation sets.
        split_point = int(indices.shape[0] * arguments.val_split)
        val_indices = indices[0:split_point]
        train_indices = indices[split_point::]

        # Creates the DataFrames for each of the data splits.
        val_df = train_df.take(val_indices)
        train_df = train_df.take(train_indices)

    # Exits script if ISIC or PCAM have not been selected.
    else:
        log(arguments, "Select either \'ISIC\' or \'PCAM\'.")
        quit()

    # Creates the training, validation and testing Dataset objects.
    train_data = Dataset(arguments, "train", train_df)
    val_data = Dataset(arguments, "validation", val_df)
    test_data = Dataset(arguments, "test", test_df)

    # Return the dataset objects.
    return train_data, val_data, test_data
