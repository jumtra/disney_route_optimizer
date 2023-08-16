import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class WaittimeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_len: int = 48):
        self.df: pd.DataFrame = df
        self.max_len: int = max_len
        self._set_data()

    def _set_data(self):
        pred_array = np.empty((0, self.max_len))
        tgt_array = np.empty((0, self.max_len))
        for date_attraction, d in self.df.groupby(["date", "attraction_name"]):
            pad_num = self.max_len - d.shape[0]
            pred = np.pad(
                d["pred"].values,
                [
                    (0, pad_num),
                ],
                constant_values=0,
            )
            tgt = np.pad(
                d["target"].values,
                [
                    (0, pad_num),
                ],
                constant_values=0,
            )
            pred_array = np.vstack((pred_array, pred))
            tgt_array = np.vstack((tgt_array, tgt))
        self.input = pred_array
        self.target = tgt_array

    def __getitem__(self, idx):
        _input = self.input[idx].reshape((self.max_len, 1))
        _target = self.target[idx].reshape((self.max_len, 1))

        return _input, _target

    def __len__(self):
        return self.input.shape[1]
