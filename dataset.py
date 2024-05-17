import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import pickle


class DecoderDataset(Dataset):
    def __init__(self, data, vectorizer):
        self.data = data
        self.vectorizer = vectorizer
        self.max_len = self.vectorizer.max_len
        self.load_data()

    def __len__(self):
        return len(self.data_frame)

    def load_data(self):
        self.data_frame = pd.DataFrame()
        self.tensors = None
        for path in tqdm(self.data):
            with open(path, 'rb') as f:
                temp_data = pickle.load(f)
            temp_data = temp_data[temp_data['tensor'].apply(
                lambda x: x.shape[0]) == 3]
            self.data_frame = pd.concat(
                (self.data_frame, temp_data.drop(columns=['tensor'])))
            if self.tensors is None:
                self.tensors = torch.stack(temp_data['tensor'].to_list())
            else:
                self.tensors = torch.concat(
                    (self.tensors, torch.stack(temp_data['tensor'].to_list())))
        del temp_data

    def __getitem__(self, i):
        x = self.tensors[i].float()
        y = self.vectorizer.encode(self.data_frame.iloc[i]['translations'])
        return x, y
