# -*- coding: utf-8 -*-


"""

"""


# Library Imports
import torch
import numpy as np
from KDEpy import FFTKDE
from scipy import optimize
from torch.cuda import amp

from utils import log

__author__    = ["Jacob Carse", "Andres Alvarez Olmo"]
__copyright__ = "Copyright 2022, Calibration"
__credits__   = ["Jacob Carse", "Andres Alvarez Olmo"]
__license__   = "MIT"
__version__   = "1.0.0"
__maintainer  = ["Jacob Carse", "Andres Alvarez Olmo"]
__email__     = ["j.carse@dundee.ac.uk", "alvarezolmoandres@gmail.com"]
__status__    = "Development"


def get_temperature(arguments, classifier, validation_data_loader, device, mode="cross_entropy") -> float:
    """

    :param arguments:
    :param classifier:
    :param validation_data_loader:
    :param device:
    :param mode:
    :param alpha:
    :return:
    """

    logit_list, label_list = [], []

    with torch.no_grad():
        for images, labels, _ in validation_data_loader:
            images = images.to(device)
            labels = labels.to(device)

            if arguments.precision == 16 and device != torch.device("cpu"):
                with amp.autocast():
                    logits = classifier(images)
            else:
                logits = classifier(images)

            logit_list.append(logits)
            label_list.append(labels)

            if len(logit_list) == arguments.batches_per_epoch:
                break

    logits = torch.cat(logit_list).cpu().numpy()
    labels = torch.cat(label_list).cpu().numpy()

    if mode == "combine_metric_1":
        temperature = optimize.minimize(combine_1, 1.0, args=(logits, labels, arguments.temp_alpha),
                                        method="L-BFGS-B", bounds=((0.05, 5.0),), tol=1e-12).x[0]
    elif mode == "combine_metric_2":
        temperature = optimize.minimize(combine_2, 1.0, args=(logits, labels, arguments.temp_alpha),
                                        method="L-BFGS-B", bounds=((0.05, 5.0),), tol=1e-12).x[0]
    elif mode == "combine_metric_3":
        temperature = optimize.minimize(combine_3, 1.0, args=(logits, labels, arguments.temp_alpha),
                                        method="L-BFGS-B", bounds=((0.05, 5.0),), tol=1e-12).x[0]
    elif mode == "combine_all":
        temperature = optimize.minimize(combine_all, 1.0, args=(logits, labels, arguments.temp_alpha),
                                        method="L-BFGS-B", bounds=((0.05, 5.0),), tol=1e-12).x[0]
    elif mode == "combine_temp":
        temp_1 = float(optimize.minimize(cross_entropy, 1.0, args=(logits, labels),
                                         method="L-BFGS-B", bounds=((0.05, 5.0),), tol=1e-12).x[0])
        temp_2 = float(optimize.minimize(kde_expected_calibration_error, 1.0, args=(logits, labels),
                                         method="L-BFGS-B", bounds=((0.05, 5.0),), tol=1e-12).x[0])
        temp_3 = float(optimize.minimize(maximum_calibration_error, 1.0, args=(logits, labels),
                                         method="L-BFGS-B", bounds=((0.05, 5.0),), tol=1e-12).x[0])
        temperature = np.mean([temp_1, temp_2, temp_3])
    elif mode == "ece":
        temperature = optimize.minimize(kde_expected_calibration_error, 1.0, args=(logits, labels),
                                        method="L-BFGS-B", bounds=((0.05, 5.0),), tol=1e-12).x[0]
    elif mode == "mce":
        temperature = optimize.minimize(maximum_calibration_error, 1.0, args=(logits, labels),
                                        method="L-BFGS-B", bounds=((0.05, 5.0),), tol=1e-12).x[0]
    else:
        temperature = optimize.minimize(cross_entropy, 1.0, args=(logits, labels),
                                        method="L-BFGS-B", bounds=((0.05, 5.0),), tol=1e-12).x[0]

    return temperature


def combine_all(temperature, *args):
    logits, labels, alpha = args

    labels = np.eye(np.amax(labels) + 1)[labels]

    logits = logits / temperature

    predictions = np.clip(np.exp(logits) / np.sum(np.exp(logits), 1)[:, None], 1e-20, 1. - 1e-20)

    items = predictions.shape[0]

    cross_entropy = -np.sum(labels * np.log(predictions)) / items

    ece = ece_kde_binary(predictions, labels)

    # Assign each prediction to a bin
    num_bins = 10
    bins = np.linspace(0.1, 1, num_bins)
    binned = np.digitize(predictions, bins)

    # Save the accuracy, confidence and size of each bin
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for bin in range(num_bins):
        bin_sizes[bin] = len(predictions[binned == bin])
        if bin_sizes[bin] > 0:
            bin_accs[bin] = (labels[binned == bin]).sum() / bin_sizes[bin]
            bin_confs[bin] = (predictions[binned == bin]).sum() / bin_sizes[bin]

    mce = 0
    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        mce = max(mce, abs_conf_dif)

    return np.mean([cross_entropy, ece, mce])


def combine_1(temperature, *args):
    logits, labels, alpha = args

    labels = np.eye(np.amax(labels) + 1)[labels]

    logits = logits / temperature

    predictions = np.clip(np.exp(logits) / np.sum(np.exp(logits), 1)[:, None], 1e-20, 1. - 1e-20)

    ece = ece_kde_binary(predictions, labels)

    # Assign each prediction to a bin
    num_bins = 10
    bins = np.linspace(0.1, 1, num_bins)
    binned = np.digitize(predictions, bins)

    # Save the accuracy, confidence and size of each bin
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for bin in range(num_bins):
        bin_sizes[bin] = len(predictions[binned == bin])
        if bin_sizes[bin] > 0:
            bin_accs[bin] = (labels[binned == bin]).sum() / bin_sizes[bin]
            bin_confs[bin] = (predictions[binned == bin]).sum() / bin_sizes[bin]

    mce = 0
    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        mce = max(mce, abs_conf_dif)

    return (ece * alpha) + (mce * (1 - alpha))


def combine_2(temperature, *args):
    logits, labels, alpha = args

    labels = np.eye(np.amax(labels) + 1)[labels]

    logits = logits / temperature

    predictions = np.clip(np.exp(logits) / np.sum(np.exp(logits), 1)[:, None], 1e-20, 1. - 1e-20)

    items = predictions.shape[0]

    cross_entropy = -np.sum(labels * np.log(predictions)) / items

    # Assign each prediction to a bin
    num_bins = 10
    bins = np.linspace(0.1, 1, num_bins)
    binned = np.digitize(predictions, bins)

    # Save the accuracy, confidence and size of each bin
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for bin in range(num_bins):
        bin_sizes[bin] = len(predictions[binned == bin])
        if bin_sizes[bin] > 0:
            bin_accs[bin] = (labels[binned == bin]).sum() / bin_sizes[bin]
            bin_confs[bin] = (predictions[binned == bin]).sum() / bin_sizes[bin]

    mce = 0
    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        mce = max(mce, abs_conf_dif)

    return (cross_entropy * alpha) + (mce * (1 - alpha))


def combine_3(temperature, *args):
    logits, labels, alpha = args

    labels = np.eye(np.amax(labels) + 1)[labels]

    logits = logits / temperature

    predictions = np.clip(np.exp(logits) / np.sum(np.exp(logits), 1)[:, None], 1e-20, 1. - 1e-20)

    items = predictions.shape[0]

    cross_entropy = -np.sum(labels * np.log(predictions)) / items

    ece = ece_kde_binary(predictions, labels)

    return (cross_entropy * alpha) + (ece * (1 - alpha))


def cross_entropy(temperature, *args) -> float:
    """

    """

    logits, labels = args

    labels = np.eye(np.amax(labels) + 1)[labels]

    logits = logits / temperature

    predictions = np.clip(np.exp(logits) / np.sum(np.exp(logits), 1)[:, None], 1e-20, 1. - 1e-20)
    items = predictions.shape[0]

    cross_entropy = -np.sum(labels * np.log(predictions)) / items

    return cross_entropy


def maximum_calibration_error(temperature, *args) -> float:
    logits, labels = args

    logits = logits / temperature

    labels = np.eye(np.amax(labels) + 1)[labels]

    predictions = np.clip(np.exp(logits) / np.sum(np.exp(logits), 1)[:, None], 1e-20, 1 - 1e-20)

    # Assign each prediction to a bin
    num_bins = 10
    bins = np.linspace(0.1, 1, num_bins)
    binned = np.digitize(predictions, bins)

    # Save the accuracy, confidence and size of each bin
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for bin in range(num_bins):
        bin_sizes[bin] = len(predictions[binned == bin])
        if bin_sizes[bin] > 0:
            bin_accs[bin] = (labels[binned == bin]).sum() / bin_sizes[bin]
            bin_confs[bin] = (predictions[binned == bin]).sum() / bin_sizes[bin]

    mce = 0
    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        mce = max(mce, abs_conf_dif)

    return mce


def expected_calibration_error(temperature, *args) -> float:
    logits, labels = args

    labels = np.eye(np.amax(labels) + 1)[labels]

    logits = logits / temperature

    predictions = np.clip(np.exp(logits) / np.sum(np.exp(logits), 1)[:, None], 1e-20, 1. - 1e-20)

    # Assign each prediction to a bin
    num_bins = 10
    bins = np.linspace(0.1, 1, num_bins)
    binned = np.digitize(predictions, bins)

    # Save the accuracy, confidence and size of each bin
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for bin in range(num_bins):
        bin_sizes[bin] = len(predictions[binned == bin])
        if bin_sizes[bin] > 0:
            bin_accs[bin] = (labels[binned == bin]).sum() / bin_sizes[bin]
            bin_confs[bin] = (predictions[binned == bin]).sum() / bin_sizes[bin]

    ece = 0
    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ece += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif

    return ece


def kde_expected_calibration_error(temperature, *args) -> float:
    logits, labels = args

    labels = np.eye(np.amax(labels) + 1)[labels]

    logits = logits / temperature

    predictions = np.clip(np.exp(logits) / np.sum(np.exp(logits), 1)[:, None], 1e-20, 1. - 1e-20)

    return ece_kde_binary(predictions, labels)


def ece_hist_binary(p, label, n_bins=15, order=1):
    p = np.clip(p, 1e-256, 1 - 1e-256)

    N = p.shape[0]
    label_index = np.array([np.where(r == 1)[0][0] for r in label])  # one hot to index
    with torch.no_grad():
        if p.shape[1] != 2:
            preds_new = torch.from_numpy(p)
            preds_b = torch.zeros(N, 1)
            label_binary = np.zeros((N, 1))
            for i in range(N):
                pred_label = int(torch.argmax(preds_new[i]).numpy())
                if pred_label == label_index[i]:
                    label_binary[i] = 1
                preds_b[i] = preds_new[i, pred_label] / torch.sum(preds_new[i, :])
        else:
            preds_b = torch.from_numpy((p / np.sum(p, 1)[:, None])[:, 1])
            label_binary = label_index

        confidences = preds_b
        accuracies = torch.from_numpy(label_binary)

        x = confidences.numpy()
        x = np.sort(x, axis=0)
        binCount = int(len(x) / n_bins)  # number of data points in each bin
        bins = np.zeros(n_bins)  # initialize the bins values
        for i in range(0, n_bins, 1):
            bins[i] = x[min((i + 1) * binCount, x.shape[0] - 1)]
            # print((i+1) * binCount)
        bin_boundaries = torch.zeros(len(bins) + 1, 1)
        bin_boundaries[1:] = torch.from_numpy(bins).reshape(-1, 1)
        bin_boundaries[0] = 0.0
        bin_boundaries[-1] = 1.0
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece_avg = torch.zeros(1)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            # print(prop_in_bin)
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece_avg += torch.abs(avg_confidence_in_bin - accuracy_in_bin) ** order * prop_in_bin
    return ece_avg.numpy()[0]


def mce_hist_binary(p, label, n_bins=15, order=1):
    p = np.clip(p, 1e-256, 1 - 1e-256)

    N = p.shape[0]
    label_index = np.array([np.where(r == 1)[0][0] for r in label])  # one hot to index
    with torch.no_grad():
        if p.shape[1] != 2:
            preds_new = torch.from_numpy(p)
            preds_b = torch.zeros(N, 1)
            label_binary = np.zeros((N, 1))
            for i in range(N):
                pred_label = int(torch.argmax(preds_new[i]).numpy())
                if pred_label == label_index[i]:
                    label_binary[i] = 1
                preds_b[i] = preds_new[i, pred_label] / torch.sum(preds_new[i, :])
        else:
            preds_b = torch.from_numpy((p / np.sum(p, 1)[:, None])[:, 1])
            label_binary = label_index

        confidences = preds_b
        accuracies = torch.from_numpy(label_binary)

        x = confidences.numpy()
        x = np.sort(x, axis=0)
        binCount = int(len(x) / n_bins)  # number of data points in each bin
        bins = np.zeros(n_bins)  # initialize the bins values
        for i in range(0, n_bins, 1):
            bins[i] = x[min((i + 1) * binCount, x.shape[0] - 1)]
            # print((i+1) * binCount)
        bin_boundaries = torch.zeros(len(bins) + 1, 1)
        bin_boundaries[1:] = torch.from_numpy(bins).reshape(-1, 1)
        bin_boundaries[0] = 0.0
        bin_boundaries[-1] = 1.0
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece_max = torch.zeros(1)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            # print(prop_in_bin)
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece = torch.abs(avg_confidence_in_bin - accuracy_in_bin) ** order * prop_in_bin
                if ece > ece_max:
                    ece_max = ece
    return ece_max


def mirror_1d(d, xmin=None, xmax=None):
    """If necessary apply reflecting boundary conditions."""
    if xmin is not None and xmax is not None:
        xmed = (xmin + xmax) / 2
        return np.concatenate(((2 * xmin - d[d < xmed]).reshape(-1, 1), d, (2 * xmax - d[d >= xmed]).reshape(-1, 1)))
    elif xmin is not None:
        return np.concatenate((2 * xmin - d, d))
    elif xmax is not None:
        return np.concatenate((d, 2 * xmax - d))
    else:
        return d


def ece_kde_binary(p, label, p_int=None, order=1):
    # points from numerical integration
    if p_int is None:
        p_int = np.copy(p)

    p = np.clip(p, 1e-256, 1 - 1e-256)
    p_int = np.clip(p_int, 1e-256, 1 - 1e-256)

    x_int = np.linspace(-0.6, 1.6, num=2 ** 14)

    N = p.shape[0]

    # this is needed to convert labels from one-hot to conventional form
    label_index = np.array([np.where(r == 1)[0][0] for r in label])
    with torch.no_grad():
        if p.shape[1] != 2:
            p_new = torch.from_numpy(p)
            p_b = torch.zeros(N, 1)
            label_binary = np.zeros((N, 1))
            for i in range(N):
                pred_label = int(torch.argmax(p_new[i]).numpy())
                if pred_label == label_index[i]:
                    label_binary[i] = 1
                p_b[i] = p_new[i, pred_label] / torch.sum(p_new[i, :])
        else:
            p_b = torch.from_numpy((p / np.sum(p, 1)[:, None])[:, 1])
            label_binary = label_index

    method = 'triweight'

    dconf_1 = (p_b[np.where(label_binary == 1)].reshape(-1, 1)).numpy()
    kbw = np.std(p_b.numpy()) * (N * 2) ** -0.2
    kbw = np.std(dconf_1) * (N * 2) ** -0.2
    # Mirror the data about the domain boundary
    low_bound = 0.0
    up_bound = 1.0
    dconf_1m = mirror_1d(dconf_1, low_bound, up_bound)
    # Compute KDE using the bandwidth found, and twice as many grid points
    pp1 = FFTKDE(bw=kbw, kernel=method).fit(dconf_1m).evaluate(x_int)
    pp1[x_int <= low_bound] = 0  # Set the KDE to zero outside of the domain
    pp1[x_int >= up_bound] = 0  # Set the KDE to zero outside of the domain
    pp1 = pp1 * 2  # Double the y-values to get integral of ~1

    p_int = p_int / np.sum(p_int, 1)[:, None]
    N1 = p_int.shape[0]
    with torch.no_grad():
        p_new = torch.from_numpy(p_int)
        pred_b_int = np.zeros((N1, 1))
        if p_int.shape[1] != 2:
            for i in range(N1):
                pred_label = int(torch.argmax(p_new[i]).numpy())
                pred_b_int[i] = p_int[i, pred_label]
        else:
            for i in range(N1):
                pred_b_int[i] = p_int[i, 1]

    low_bound = 0.0
    up_bound = 1.0
    pred_b_intm = mirror_1d(pred_b_int, low_bound, up_bound)
    # Compute KDE using the bandwidth found, and twice as many grid points
    pp2 = FFTKDE(bw=kbw, kernel=method).fit(pred_b_intm).evaluate(x_int)
    pp2[x_int <= low_bound] = 0  # Set the KDE to zero outside of the domain
    pp2[x_int >= up_bound] = 0  # Set the KDE to zero outside of the domain
    pp2 = pp2 * 2  # Double the y-values to get integral of ~1

    if p.shape[1] != 2:  # top label (confidence)
        perc = np.mean(label_binary)
    else:  # or joint calibration for binary cases
        perc = np.mean(label_index)

    integral = np.zeros(x_int.shape)
    reliability = np.zeros(x_int.shape)
    for i in range(x_int.shape[0]):
        conf = x_int[i]
        if np.max([pp1[np.abs(x_int - conf).argmin()], pp2[np.abs(x_int - conf).argmin()]]) > 1e-6:
            accu = np.min([perc * pp1[np.abs(x_int - conf).argmin()] / pp2[np.abs(x_int - conf).argmin()], 1.0])
            if np.isnan(accu) == False:
                integral[i] = np.abs(conf - accu) ** order * pp2[i]
                reliability[i] = accu
        else:
            if i > 1:
                integral[i] = integral[i - 1]

    ind = np.where((x_int >= 0.0) & (x_int <= 1.0))
    return np.trapz(integral[ind], x_int[ind]) / np.trapz(pp2[ind], x_int[ind])