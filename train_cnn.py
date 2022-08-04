# -*- coding: utf-8 -*-


"""
The file contains implementations of the functions used to train a CNN model.
    train_cnn - Function used to train a Convolutional Neural Network.
"""


# Built-in/Generic Imports
import time

# Library Imports
from torch.cuda import amp
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler
from torch.utils.tensorboard import SummaryWriter

# Own Modules
from utils import *
from dataset import get_datasets
from model import Classifier


__author__    = ["Jacob Carse", "Andres Alvarez Olmo"]
__copyright__ = "Copyright 2022, Calibration"
__credits__   = ["Jacob Carse", "Andres Alvarez Olmo"]
__license__   = "MIT"
__version__   = "1.0.0"
__maintainer  = ["Jacob Carse", "Andres Alvarez Olmo"]
__email__     = ["j.carse@dundee.ac.uk", "alvarezolmoandres@gmail.com"]
__status__    = "Development"


def train_cnn(arguments: Namespace, device: torch.device) -> None:
    """
    Function for training the Convolutional Neural Network.
    :param arguments: ArgumentParser Namespace object with arguments used for training.
    :param device: PyTorch device that will be used for training.
    """

    # Loads a TensorBoard Summary Writer.
    if arguments.tensorboard_dir != "":
        writer = SummaryWriter(os.path.join(arguments.tensorboard_dir, arguments.task, arguments.experiment))

    # Loads the training and validation data.
    train_data, val_data, _ = get_datasets(arguments)

    # Creates the training data loader using the dataset objects.
    training_data_loader = DataLoader(train_data, batch_size=arguments.batch_size,
                                      shuffle=True, num_workers=arguments.data_workers,
                                      pin_memory=False, drop_last=False)

    # Creates the validation data loader using the dataset objects.
    validation_data_loader = DataLoader(val_data, batch_size=arguments.batch_size * 2,
                                        shuffle=False, num_workers=arguments.data_workers,
                                        pin_memory=False, drop_last=False)

    log(arguments, "Loader Datasets\n")

    # Initialises the classifier model.
    classifier = Classifier(arguments.efficient_net, train_data.num_class)

    # Sets the classifier to training mode.
    classifier.train()

    # Moves the classifier to the selected device.
    classifier.to(device)

    # Initialises the optimiser used to optimise the parameters of the model.
    optimiser = SGD(params=classifier.parameters(), lr=arguments.minimum_lr)

    # Initialises the learning rate scheduler to adjust the learning rate during training.
    scheduler = lr_scheduler.CyclicLR(optimiser, arguments.minimum_lr, arguments.maximum_lr, mode="triangular2")

    # Initialises the gradient scaler used 16 bit precision.
    if arguments.precision == 16 and device != torch.device("cpu"):
        scaler = amp.GradScaler()

    log(arguments, "Model Initialised")

    # Declares the main logging variables for training.
    start_time = time.time()
    best_loss, best_epoch, total_batches = 1e10, 0, 0

    # The beginning of the main training loop.
    for epoch in range(1, arguments.epochs + 1):
        # Declares the logging variables for the epoch.
        epoch_acc, epoch_loss, num_batches = 0., 0., 0

        # Loops through the training data batches.
        for images, labels in training_data_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Resets the gradients in the model.
            optimiser.zero_grad()

            # Performs training step with 16 bit precision.
            if arguments.precision == 16 and device != torch.device("cpu"):
                with amp.autocast():
                    # Performs forward propagation using the CNN model.
                    predictions = classifier(images)

                    if arguments.focal_loss:
                        # Calculates the focal loss.
                        ce_loss = F.cross_entropy(predictions, labels, reduction='none')
                        pt = torch.exp(-ce_loss)
                        loss = (arguments.focal_alpha * (1 - pt) ** arguments.focal_gamma * ce_loss).mean()
                    else:
                        # Calculates the cross entropy loss.
                        loss = F.cross_entropy(predictions, labels, label_smoothing=arguments.label_smoothing)

                # Using the gradient scaler performs backward propagation.
                scaler.scale(loss).backward()

                # Update the weights of the model using the optimiser.
                scaler.step(optimiser)

                # Updates the scale factor of the gradient scaler.
                scaler.update()

            # Performs training step with 32 bit precision.
            else:
                # Performs forward propagation using the CNN model.
                predictions = classifier(images)

                if arguments.focal_loss:
                    # Calculates the focal loss.
                    ce_loss = F.cross_entropy(predictions, labels, reduction='none')
                    pt = torch.exp(-ce_loss)
                    loss = (arguments.focal_alpha * (1 - pt) ** arguments.focal_gamma * ce_loss).mean()
                else:
                    # Calculates the cross entropy loss.
                    loss = F.cross_entropy(predictions, labels, label_smoothing=arguments.label_smoothing)

                # Performs backward propagation.
                loss.backward()

                # Updates the weights of the model using the optimiser.
                optimiser.step()

            # Updates the learning rate scheduler.
            scheduler.step()

            # Calculates the accuracy of the batch.
            batch_accuracy = (predictions.max(dim=1)[1] == labels).sum().double() / labels.shape[0]

            # Adds the number of batches, losses and accuracy to the epoch sum.
            num_batches += 1
            epoch_loss += loss.item()
            epoch_acc += batch_accuracy

            # Writes the batch loss and accuracy to TensorBoard logger.
            if arguments.tensorboard_dir != "":
                writer.add_scalar("Loss/batch", loss.item(), num_batches + total_batches)
                writer.add_scalar("Accuracy/batch", batch_accuracy, num_batches + total_batches)

            # Logs the details of the epoch progress.
            if num_batches % arguments.log_interval == 0:
                log(arguments, "Time: {}s\tTrain Epoch: {} [{}/{}] ({:.0f}%)\tLoss: {:.6f}\tAccuracy: {:.6f}".format(
                    str(int(time.time() - start_time)).rjust(6, '0'), str(epoch).rjust(2, '0'),
                    str(num_batches * arguments.batch_size).rjust(len(str(len(train_data))), '0'),
                    len(train_data), 100. * num_batches / (len(train_data) / arguments.batch_size),
                                     epoch_loss / num_batches, epoch_acc / num_batches
                ))

            # If the number of batches have been reached end epoch.
            if num_batches == arguments.batches_per_epoch:
                break

        # Updates the total number of batches (used for debugging).
        total_batches += num_batches

        # Writes epoch loss and accuracy to TensorBoard.
        if arguments.tensorboard_dir != "":
            writer.add_scalar("Loss/train", epoch_loss, epoch)
            writer.add_scalar("Accuracy/train", epoch_acc, epoch)

        # Declares the logging variables for validation.
        val_acc, val_loss, val_batches = 0., 0., 0

        # Performs the validation epoch with no gradient calculations.
        with torch.no_grad():
            # Loops through the validation data batches.
            for images, labels, _ in validation_data_loader:
                # Moves the images and labels to the selected device.
                images = images.to(device)
                labels = labels.to(device)

                # Performs forward propagation using 16 bit precision.
                if arguments.precision == 16 and device != torch.device("cpu"):
                    with amp.autocast():
                        # Performs forward propagation using the CNN model.
                        predictions = classifier(images)

                        if arguments.focal_loss:
                            # Calculates the focal loss.
                            ce_loss = F.cross_entropy(predictions, labels, reduction='none')
                            pt = torch.exp(-ce_loss)
                            loss = (arguments.focal_alpha * (1 - pt) ** arguments.focal_gamma * ce_loss).mean()
                        else:
                            # Calculates the cross entropy loss.
                            loss = F.cross_entropy(predictions, labels, label_smoothing=arguments.label_smoothing)

                # Performs forward propagation using 32 bit precision.
                else:
                    # Performs forward propagation using the CNN model.
                    predictions = classifier(images)

                    if arguments.focal_loss:
                        # Calculates the focal loss.
                        ce_loss = F.cross_entropy(predictions, labels, reduction='none')
                        pt = torch.exp(-ce_loss)
                        loss = (arguments.focal_alpha * (1 - pt) ** arguments.focal_gamma * ce_loss).mean()
                    else:
                        # Calculates the cross entropy loss.
                        loss = F.cross_entropy(predictions, labels, label_smoothing=arguments.label_smoothing)

                # Calculates the accuracy of the batch.
                batch_accuracy = (predictions.max(dim=1)[1] == labels).sum().double() / labels.shape[0]

                # Adds the number of batches, loss and accuracy to validation sum.
                val_batches += 1
                val_loss += loss.item()
                val_acc += batch_accuracy.item()

                # If the number of batches have been reached end validation.
                if val_batches == arguments.batches_per_epoch:
                    break

        # Writes validation loss and accuracy to TensorBoard
        if arguments.tensorboard_dir != "":
            writer.add_scalar("Loss/val", val_loss / val_batches, epoch)
            writer.add_scalar("Accuracy/val", val_acc / val_batches, epoch)

        # Logs the details of the training epoch.
        log(arguments, "\nEpoch: {}\tTraining Loss: {:.6f}\tTraining Accuracy: {:.6f}\n"
                       "Validation Loss: {:.6f}\tValidation Accuracy: {:.6f}\n".format(
            epoch, epoch_loss / num_batches, epoch_acc / num_batches, val_loss / val_batches, val_acc / val_batches
        ))

        # If the current epoch has the best validation loss then save the model with the prefix best.
        if val_loss / val_batches < best_loss:
            best_loss = val_loss / val_batches
            best_epoch = epoch
            classifier.save_model(arguments.model_dir, arguments.experiment)

    # Logs the final training information.
    log(arguments,
        f"\nTraining Finished with best loss of {best_loss} at epoch {best_epoch} in {int(time.time() - start_time)}s.")
