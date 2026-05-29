from torch.utils.data import Dataset
import pandas as pd
from typing import Tuple
import torch

class Loader(Dataset):
    def __init__(self, data_df: pd.DataFrame):
        user_col = 'user_idx'
        movie_col = 'movie_idx'
        rating_col = 'ratings'
        required_cols = {user_col,movie_col,rating_col}
        if not required_cols.issubset(data_df.columns):
            raise ValueError(f"Require correct column names")
        if data_df.isnull().values.any():
            raise ValueError(f"DataFrame contains NAN, clean the dataframe beforehand")
        
        self.user_id=torch.tensor(data_df[user_col].values, dtype=torch.int64)
        self.movie_id=torch.tensor(data_df[movie_col].values, dtype=torch.int64)
        self.ratings=torch.tensor(data_df[rating_col].values, dtype=torch.float32)

    def __len__(self)-> int:
        return len(self.user_id)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return(
            self.user_id[idx],
            self.movie_id[idx],
            self.ratings[idx]
        )