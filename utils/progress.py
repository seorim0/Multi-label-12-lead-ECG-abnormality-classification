import os
import time
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import numpy as np


######################################################################################################################
#                                                   for dataset                                                      #
######################################################################################################################

# Load recording file as an array.
def load_recording(recording_file, header=None, leads=None, key='val'):
    from scipy.io import loadmat
    recording = loadmat(recording_file)[key]
    if header and leads:
        recording = choose_leads(recording, header, leads)
    return recording


# Select specific leads from the recording.
def choose_leads(recording, header, leads):
    num_leads = len(leads)
    num_samples = np.shape(recording)[1]
    chosen_recording = np.zeros((num_leads, num_samples), recording.dtype)
    available_leads = get_leads(header)
    for i, lead in enumerate(leads):
        if lead in available_leads:
            j = available_leads.index(lead)
            chosen_recording[i, :] = recording[j, :]
    return chosen_recording


# Extract lead names from the header.
def get_leads(header):
    leads = list()
    for i, l in enumerate(header.split('\n')):
        entries = l.split(' ')
        if i == 0:
            num_leads = int(entries[1])
        elif i <= num_leads:
            leads.append(entries[-1])
        else:
            break
    return tuple(leads)


# Create a new directory.
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Normalize the ECG data using Z-score normalization.
def normalize_ecg(ecg_data):
    mean = np.mean(ecg_data, axis=1, keepdims=True)
    std = np.std(ecg_data, axis=1, keepdims=True)
    return (ecg_data - mean) / (std + 1e-8)  # Prevent division by zero


######################################################################################################################
#                                                  Parser init                                                       #
######################################################################################################################
# Check if a variable is a number or represents a number.
def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False


# Check if a variable is a a finite number or represents a finite number.
def is_finite_number(x):
    if is_number(x):
        return np.isfinite(float(x))
    else:
        return False


# Load a table with row and column names.
def load_table(table_file):
    # The table should have the following form:
    # ,    a,   b,   c
    # a, 1.2, 2.3, 3.4
    # b, 4.5, 5.6, 6.7
    # c, 7.8, 8.9, 9.0
    table = list()
    with open(table_file, 'r') as f:
        for i, l in enumerate(f):
            arrs = [arr.strip() for arr in l.split(',')]
            table.append(arrs)

    # Define the numbers of rows and columns and check for errors.
    num_rows = len(table) - 1
    if num_rows < 1:
        raise Exception('The table {} is empty.'.format(table_file))
    row_lengths = set(len(table[i]) - 1 for i in range(num_rows))
    if len(row_lengths) != 1:
        raise Exception('The table {} has rows with different lengths.'.format(table_file))
    num_cols = min(row_lengths)
    if num_cols < 1:
        raise Exception('The table {} is empty.'.format(table_file))

    # Find the row and column labels.
    rows = [table[0][j + 1] for j in range(num_rows)]
    cols = [table[i + 1][0] for i in range(num_cols)]

    # Find the entries of the table.
    values = np.zeros((num_rows, num_cols), dtype=np.float64)
    for i in range(num_rows):
        for j in range(num_cols):
            value = table[i + 1][j + 1]
            if is_finite_number(value):
                values[i, j] = float(value)
            else:
                values[i, j] = float('nan')

    return rows, cols, values


######################################################################################################################
#                                                   for training                                                     #
######################################################################################################################
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


# Calculate total number of parameters in a model.
def cal_total_params(our_model):
    total_parameters = 0
    for variable in our_model.parameters():
        shape = variable.size()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        total_parameters += variable_parameters

    return total_parameters


# Display a progress bar during training/validation.
class Bar(object):
    def __init__(self, dataloader):
        if not hasattr(dataloader, 'dataset'):
            raise ValueError('Attribute `dataset` not exists in dataloder.')
        if not hasattr(dataloader, 'batch_size'):
            raise ValueError('Attribute `batch_size` not exists in dataloder.')

        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        self.dataset = dataloader.dataset
        self.batch_size = dataloader.batch_size
        self._idx = 0
        self._batch_idx = 0
        self._time = []
        self._DISPLAY_LENGTH = 50

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._time) < 2:
            self._time.append(time.time())

        self._batch_idx += self.batch_size
        if self._batch_idx > len(self.dataset):
            self._batch_idx = len(self.dataset)

        try:
            batch = next(self.iterator)
            self._display()
        except StopIteration:
            raise StopIteration()

        self._idx += 1
        if self._idx >= len(self.dataloader):
            self._reset()

        return batch

    def _display(self):
        if len(self._time) > 1:
            t = (self._time[-1] - self._time[-2])
            eta = t * (len(self.dataloader) - self._idx)
        else:
            eta = 0

        rate = self._idx / len(self.dataloader)
        len_bar = int(rate * self._DISPLAY_LENGTH)
        bar = ('=' * len_bar + '>').ljust(self._DISPLAY_LENGTH, '.')
        idx = str(self._batch_idx).rjust(len(str(len(self.dataset))), ' ')

        tmpl = '\r{}/{}: [{}] - ETA {:.1f}s'.format(
            idx,
            len(self.dataset),
            bar,
            eta
        )
        print(tmpl, end='')
        if self._batch_idx == len(self.dataset):
            print()

    def _reset(self):
        self._idx = 0
        self._batch_idx = 0
        self._time = []


######################################################################################################################
#                                               for validation and test                                              #
######################################################################################################################
# Compute a modified confusion matrix for multi-class, multi-label tasks.
def compute_modified_confusion_matrix(labels, outputs):
    # Compute a binary multi-class, multi-label confusion matrix, where the rows
    # are the labels and the columns are the outputs.
    num_recordings, num_classes = np.shape(labels)
    A = np.zeros((num_classes, num_classes))

    # Iterate over all of the recordings.
    for i in range(num_recordings):
        # Calculate the number of positive labels and/or outputs.
        normalization = float(max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1))
        # Iterate over all of the classes.
        for j in range(num_classes):
            # Assign full and/or partial credit for each positive class.
            if labels[i, j]:
                for k in range(num_classes):
                    if outputs[i, k]:
                        A[j, k] += 1.0 / normalization

    return A


# Determine the threshold for each class based on F1 score.
def find_threshold(targets, scores, num_classes=26):
    classes = ['164889003', '164890007', '6374002', '426627000', '733534002',
               '713427006', '270492004', '713426002', '39732003', '445118002',
               '164947007', '251146004', '111975006', '698252002', '426783006',
               '284470004', '10370003', '365413008', '427172004', '164917005',
               '47665007', '427393009', '426177001', '427084000', '164934002',
               '59931005']

    thresholds = []
    for class_idx in range(num_classes):
        pre_targets = targets[:, class_idx]
        pre_scores = scores[:, class_idx]

        # get best threshold based on f1 score
        best_threshold, best_f1 = 0, 0
        for threshold in range(0, 101):
            binary_preds = torch.tensor(pre_scores) > threshold / 100
            f1 = f1_score(pre_targets, binary_preds)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold / 100
        thresholds.append(best_threshold)
    return thresholds


# Get binary outputs based on a threshold.
def get_binary_outputs(q, threshold):
    labels = torch.zeros_like(q)
    for i in range(len(threshold)):
        pre_q = q[i]
        pre_threshold = threshold[i]

        labels[i] = pre_q >= pre_threshold
    return labels


# Get binary outputs (as numpy arrays) based on a threshold.
def get_binary_outputs_np(q, threshold):
    labels = np.zeros_like(q)
    for i in range(len(threshold)):
        pre_q = q[:, i]
        pre_threshold = threshold[i]

        labels[:, i] = pre_q >= pre_threshold
    return labels
