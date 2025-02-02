import numpy as np
import pandas as pd
import torch

from scipy import special
from torch.utils.data import Dataset


class ITSDataset(Dataset):
    """
    Dataset class for irregular data, originally taken from
    https://github.com/edebrouwer/gru_ode_bayes
    and modified
    """
    def __init__(self, in_df, validation=False, val_options=None):
        """
        Keyword arguments:
        in_df -- patient data (pd.DataFrame)
        validation -- if the constructed dataset is used for validation (bool)
        val_options -- options to specify forecasting time period (dict)
        """
        self.validation = validation
        self.df = in_df

        # Create Dummy covariates and labels if they are not fed.
        num_unique = np.zeros(self.df['ID'].nunique())
        self.init_cov_df = pd.DataFrame({'ID': self.df['ID'].unique(), 'Cov': num_unique})
        self.label_df = pd.DataFrame({'ID': self.df['ID'].unique(), 'label': num_unique})

        if self.validation:
            before_idx = self.df.loc[self.df['Time'] <= val_options['T_val'], 'ID'].unique()
            if val_options.get("T_stop"):
                after_idx = self.df.loc[(self.df['Time'] > val_options['T_val']) & (self.df['Time'] < val_options['T_stop']), 'ID'].unique()
            else:
                after_idx = self.df.loc[self.df['Time'] > val_options['T_val'], 'ID'].unique()

            valid_idx = np.intersect1d(before_idx, after_idx)
            self.df = self.df.loc[self.df['ID'].isin(valid_idx)].copy()
            self.label_df = self.label_df.loc[self.label_df['ID'].isin(valid_idx)].copy()
            self.init_cov_df = self.init_cov_df.loc[self.init_cov_df['ID'].isin(valid_idx)].copy()

        map_dict = dict(zip(self.df.loc[:, 'ID'].unique(), np.arange(self.df.loc[:, 'ID'].nunique())))
        self.map_dict = map_dict
        self.df.loc[:, 'ID'] = self.df.loc[:, 'ID'].map(map_dict)

        self.init_cov_df.loc[:, 'ID'] = self.init_cov_df.loc[:, 'ID'].map(map_dict)
        self.label_df.loc[:, 'ID'] = self.label_df['ID'].map(map_dict)

        assert self.init_cov_df.shape[0] == self.df['ID'].nunique()

        self.variable_num = sum([c.startswith('Value') for c in self.df.columns])
        self.init_cov_dim = self.init_cov_df.shape[1] - 1
        self.init_cov_df = self.init_cov_df.astype(np.float32)
        self.init_cov_df.set_index('ID', inplace=True)
        self.label_df.set_index('ID', inplace=True)
        self.df = self.df.astype(np.float32)

        if self.validation:
            assert val_options is not None, 'Validation set options should be fed'
            self.df_before = self.df.loc[self.df['Time'] <= val_options['T_val']].copy()
            self.df_after = self.df.loc[self.df['Time'] > val_options['T_val']].sort_values('Time').copy()
            if val_options.get("T_stop"):
                self.df_after = self.df_after.loc[self.df_after['Time'] < val_options['T_stop']].sort_values('Time').copy()
            self.df_after = self.df_after.groupby('ID').head(val_options['max_val_samples']).copy()
            self.df = self.df_before  # We remove observations after T_val
            self.df_after.ID = self.df_after.ID.astype(np.int)
            self.df_after.sort_values('Time', inplace=True)
        else:
            self.df_after = None

        self.length = self.df['ID'].nunique()
        self.df.ID = self.df.ID.astype(np.int)
        self.df.set_index('ID', inplace=True)
        self.df.sort_values('Time', inplace=True)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        subset = self.df.loc[idx]
        # print(subset.shape)
        # exit()
        if len(subset.shape) == 1:
            subset = self.df.loc[[idx]]
        init_covs = self.init_cov_df.loc[idx].values
        tag = self.label_df.loc[idx].astype(np.float32).values
        if self.validation:
            val_samples = self.df_after.loc[self.df_after['ID'] == idx]
        else:
            val_samples = None
        return {'idx': idx, 'y': tag, 'path': subset, 'init_cov': init_covs, 'val_samples': val_samples}

class SPEECHDataset(Dataset):
    def __init__(self, in_npz, validation=False, val_options=None):
        self.npz = in_npz
        self.length = self.npz.shape[0]
        self.variable_num = self.npz.shape[-1]
        self.init_cov_dim = 1
        self.validation = validation

        if self.validation:
            assert val_options is not None, 'Validation set options should be fed'
            self.df_before = self.df.loc[self.df['Time'] <= val_options['T_val']].copy()
            self.df_after = self.df.loc[self.df['Time'] > val_options['T_val']].sort_values('Time').copy()
            if val_options.get("T_stop"):
                self.df_after = self.df_after.loc[self.df_after['Time'] < val_options['T_stop']].sort_values('Time').copy()
            self.df_after = self.df_after.groupby('ID').head(val_options['max_val_samples']).copy()
            self.df = self.df_before  # We remove observations after T_val
            self.df_after.ID = self.df_after.ID.astype(np.int)
            self.df_after.sort_values('Time', inplace=True)

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        subset = self.npz[idx]
        # print(subset.shape)
        # exit()
        # print("Datset subset :", subset.shape)
        tag = None
        init_covs = 0
        if self.validation:
            val_samples = True
        else:
            val_samples = None
        return {'idx': idx, 'y': tag, 'path': subset, 'init_cov': init_covs, 'val_samples': val_samples}
        

def collate_GOB(batch):
    """
    Collate function used in the DataLoader to format data for GRU-ODE-Bayes,
    taken from https://github.com/edebrouwer/gru_ode_bayes
    """
    df = pd.concat([b['path'] for b in batch], axis=0)
    df.reset_index(inplace=True)
    map_dict = dict(zip(df['ID'].unique(), np.arange(df['ID'].nunique())))
    df['ID'] = df['ID'].map(map_dict)
    df.set_index('ID', inplace=True)
    df.sort_values(by=['ID', 'Time'], inplace=True)

    df_cov = torch.Tensor([b['init_cov'] for b in batch])
    labels = torch.tensor([b['y'] for b in batch])

    times = [df.loc[i].Time.values if isinstance(df.loc[i].Time, pd.Series)
             else np.array([df.loc[i].Time]) for i in df.index.unique()]

    num_observations = [len(x) for x in times]
    value_cols = [c.startswith('Value') for c in df.columns]
    mask_cols = [c.startswith('Mask') for c in df.columns]

    if batch[0]['val_samples'] is not None:
        df_after = pd.concat(b['val_samples'] for b in batch)
        df_after['ID'] = df_after['ID'].map(map_dict)
        df_after.sort_values(by=['ID', 'Time'], inplace=True)
        df_after.set_index('ID', inplace=True)
        value_cols_val = [c.startswith('Value') for c in df_after.columns]
        mask_cols_val = [c.startswith('Mask') for c in df_after.columns]
        x_val = torch.tensor(df_after.iloc[:, value_cols_val].values)
        m_val = torch.tensor(df_after.iloc[:, mask_cols_val].values)
        times_val = [df_after.loc[i].Time.values if isinstance(df_after.loc[i].Time, pd.Series)
                     else np.array([df_after.loc[i].Time]) for i in df_after.index.unique()]
    else:
        x_val = None
        m_val = None
        times_val = None
    
    # print(times.shape)
    print(times)

    res = dict()
    res['times'] = np.array(times, dtype=object)
    res['num_obs'] = torch.Tensor(num_observations)
    res['X'] = torch.tensor(df.iloc[:, value_cols].values)
    res['M'] = torch.tensor(df.iloc[:, mask_cols].values)
    res['y'] = labels
    res['cov'] = df_cov
    res['X_val'] = x_val
    res['M_val'] = m_val
    res['times_val'] = times_val

    # print(res['times'].shape)
    # print(res['times'])
    # exit()

    for k in res.keys():
        print(k)
        try:
            print(res[k].shape)
        except:
            print(res[k])
    # exit()
    

    return res

def collate_SPEECH(batch):
    """
    Collate function used in the DataLoader to format data for GRU-ODE-Bayes,
    taken from https://github.com/edebrouwer/gru_ode_bayes
    """
    subset = torch.stack([torch.tensor(b["path"]) for b in batch], 0)
    b, t, d = subset.shape
    # times1d = np.ones((b,54))
    # times = np.cumsum(times, axis = 1)

    times1d = np.array(np.arange(54)).astype(np.float32)
    times = [times1d for i in range(b)]

    num_observations = [54] * b

    if batch[0]['val_samples']:
        x_val = subset[:, 48:, :]
        m_val = torch.ones_like(x_val)
        times_val = 

    subset = torch.reshape(subset, (b * t, d))
    

    # print(times)

    res = dict()
    res['times'] = np.array(times, dtype=object)
    res['num_obs'] = torch.Tensor(num_observations)
    res['X'] = torch.tensor(subset)
    res['M'] = torch.ones_like(subset)
    res['y'] = torch.zeros((b,1))
    res['cov'] = torch.zeros((b,1))
    res['X_val'] = None
    res['M_val'] = None
    res['times_val'] = None
    
    # print(res['times'].shape)
    # print(res['times'])
    # exit()

    # for k in res.keys():
    #     print(k)
    #     try:
    #         print(res[k].shape)
    #     except:
    #         print(res[k])
    # exit()

    
    return res

def compute_corr(x_true, x_hat, mask):
    means_true = x_true.sum(0) / mask.sum(0)
    means_hat = x_hat.sum(0) / mask.sum(0)
    corr_num = ((x_true - means_true) * (x_hat - means_hat) * mask).sum(0)
    corr_denum1 = ((x_true - means_true).pow(2) * mask).sum(0).sqrt()
    corr_denum2 = ((x_hat - means_hat).pow(2) * mask).sum(0).sqrt()
    return corr_num / (corr_denum1 * corr_denum2)


def sort_array_on_other(x1, x2):
    """
    This function returns the permutation y needed to transform x2 in x1 s.t. x2[y]=x1
    https://github.com/edebrouwer/gru_ode_bayes
    """
    temp_dict = dict(zip(x1, np.arange(len(x1))))
    index = np.vectorize(temp_dict.get)(x2)
    perm = np.argsort(index)
    return perm


def log_lik_gaussian(x, mu, logvar):
    """
    Return loglikelihood of x in gaussian specified by mu and logvar, taken from
    https://github.com/edebrouwer/gru_ode_bayes
    """
    return np.log(np.sqrt(2 * np.pi)) + (logvar / 2) + ((x - mu).pow(2) / (2 * logvar.exp()))


def tail_fun_gaussian(x, mu, logvar):
    """
    Returns the probability that the given distribution is HIGHER than x
    https://github.com/edebrouwer/gru_ode_bayes
    """
    return 0.5 - 0.5 * special.erf((x - mu) / ((0.5 * logvar).exp() * np.sqrt(2)))
