import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from torchvision.transforms import Resize
from torch.utils.data import Dataset, DataLoader
from PIL import Image
DATASET_INDEX_PATH = "./dataset/10234.txt"
SRC_PATH = "./dataset/"

transform_list = [
    Resize((32, 32)),
    transforms.ToTensor()
]

data_transforms = transforms.Compose(transform_list)


def default_loader(path):
    return Image.open(path).convert('RGB')


def format_input(input):
    """
    Format the input array by combining the time and input dimension of the input for feeding into ForecastNet.
    That is: reshape from [n_batches, in_seq_length, input_dim] to [n_batches, in_seq_length * input_dim]
    :param input: Input tensor with shape [n_batches, in_seq_length, input_dim]
    :return: input tensor reshaped to [n_batches, in_seq_length * input_dim]
    """
    N, T, D = input.shape
    input_reshaped = torch.reshape(input, (N, -1))
    return input_reshaped


class TrainDataset():
    def __init__(self, txt, transform=None, loader=default_loader, T_in_seq=10,
                 T_out_seq=5, time_major=False):
        fh = open(txt, 'r')
        imgseqsPath = []
        imgseqs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            imgseqsPath.append(SRC_PATH + line)
        self.seqSize = len(imgseqsPath)
        self.imgseqsPath = imgseqsPath
        self.transform = transform
        self.loader = loader
        self.T_in_seq = T_in_seq
        self.T_out_seq = T_out_seq
        self.time_major = time_major
        self.imgseqs = imgseqs

    def batch_format(self):
        T, n_dims = self.dataset.shape
        inputs = []
        targets = []
        # Loop over the indexes, extract a sample at that index and run it through the model
        for t in range(T - self.T_in_seq - self.T_out_seq + 1):
            # Extract the training and testing samples at the current permuted index
            inputs.append(self.dataset[t: t + self.T_in_seq, :])
            targets.append(self.dataset[t + self.T_in_seq:t + self.T_in_seq + self.T_out_seq, :])

        # Convert lists to arrays of size [n_samples, T_in, N] and [n_samples, T_out, N]
        inputs = np.array(inputs)
        targets = np.array(targets)

        if self.time_major:
            inputs = np.transpose(inputs, (1, 0, 2))
            targets = np.transpose(targets, (1, 0, 2))

        trainX_list = []
        trainY_list = []
        validX_list = []
        validY_list = []
        testX_list = []
        testY_list = []

        batch_size = self.seqSize - self.T_in_seq - self.T_out_seq + 1
        # The last 20% for test and valid
        test_idx = batch_size - int(0.2 * batch_size)
        valid_idx = batch_size - int(0.1 * batch_size)

        trainX_list.append(inputs[: test_idx, :, :])
        trainY_list.append(targets[: test_idx, :, :])
        validX_list.append(inputs[test_idx: valid_idx, :, :])
        validY_list.append(targets[test_idx: valid_idx, :, :])
        testX_list.append(inputs[valid_idx:, :, :])
        testY_list.append(targets[valid_idx:, :, :])

        trainX = np.concatenate(trainX_list, axis=0)
        trainY = np.concatenate(trainY_list, axis=0)
        validX = np.concatenate(validX_list, axis=0)
        validY = np.concatenate(validY_list, axis=0)
        testX = np.concatenate(testX_list, axis=0)
        testY = np.concatenate(testY_list, axis=0)

        return trainX, trainY, validX, validY, testX, testY

    def get_datasets(self):  # 每次迭代返回连续的10图像和第11张图像的标签
        for image in self.imgseqsPath:
            img = self.loader(image)
            if self.transform is not None:
                img = self.transform(img)
            self.imgseqs.append(img.view(-1))
        self.dataset = np.stack(self.imgseqs, axis=0)
        return self.batch_format()

    def __getitem__(self, index):
        # TO-DO
        pass

    def __len__(self):
        return self.seqSize


def generate_data(period=5):
    # Frequency
    #f = 1 / period
    T_in_seq = 2 * period
    T_out_seq = period

    train_data = TrainDataset(DATASET_INDEX_PATH, data_transforms, default_loader, T_in_seq, T_out_seq)
    return train_data.get_datasets()

generate_data()