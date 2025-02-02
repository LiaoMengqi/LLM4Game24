import random

from torch.utils.data import Dataset


class Dataset4Game24(Dataset):
    def __init__(self, data, shuffle_digit=False):
        self.data = data
        self.shuffle_digit = shuffle_digit

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        case = self.data[idx]
        sample = [str(i) for i in case]
        if self.shuffle_digit:
            random.shuffle(sample)
        sample = ' '.join(sample) + '\n'
        return {'prompt': sample, 'case': case}
