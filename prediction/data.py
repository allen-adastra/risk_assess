"""
Create data loader
"""

# Argoverse related imports
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap

# Pytorch related imports
import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

# Project imports
from prediction.utils import get_nearby_lanes

# Other imports
import numpy as np
import os

class ArgoverseDataset(Dataset):
    """Argoverse dataset."""

    def __init__(self, data_dir, obs_len=20, position_downscaling_factor=100):
        """
        Args:
            inp_dir: Directory with all trajectories
            obs_len: length of observed trajectory
        """
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.position_downscaling_factor = position_downscaling_factor

        assert os.path.isdir(data_dir), 'Invalid Data Directory'
        self.afl = ArgoverseForecastingLoader(data_dir)
        self.avm = ArgoverseMap()

    def __len__(self):
        return len(self.afl)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        traj = self.afl[idx].agent_traj
        # city = self.afl[idx].seq_df["CITY_NAME"][0]
        # lanes = get_nearby_lanes(self.avm, traj, city)

        # first transform the trajectory for better training and visualization
        R, t = self.get_normalize_vectors(traj)
        traj_normalized = self.normalize_track(traj, R, t)
        traj_normalized = traj_normalized/self.position_downscaling_factor

        past_traj = traj_normalized[:self.obs_len]
        future_traj = traj_normalized[self.obs_len:]

        # compute acc and alpha (angular rate)
        dt = 0.1
        assert self.obs_len >= 2, 'observed length too small'
        # we need more info from past trajectory to compute controls
        future_traj_extended = traj_normalized[self.obs_len-2:]
        future_acc, future_alpha, vel0, theta0 = traj_to_accels_yawrates(future_traj_extended, dt)

        # create multi-entry target
        target = {}
        target['traj'] = torch.Tensor(future_traj)
        target['acc'] = torch.Tensor(future_acc)
        target['alpha'] = torch.Tensor(future_alpha)
        target['vel0'] = vel0
        target['theta0'] = theta0

        # add additional information for visualization and testing
        target['past_traj'] = past_traj
        target['past_traj_origin'] = traj[:self.obs_len] # unnormalized past trajectory
        target['future_traj_origin'] = traj[self.obs_len:] # unnormalized future trajectory
        target['acc_gt'] = future_acc
        target['alpha_gt'] = future_alpha
        target['R'] = R
        target['t'] = t
        # target['normalized_lanes'] = [self.normalize_track(ln, R, t) for ln in lanes]

        return torch.Tensor(past_traj), target

    def get_normalize_vectors(self, traj):
        """
        Given a trajectory with shape nx2,
        get a pair of normalizing vectors (R, t) for the track
        such that the most recent observed position is at the origin,
        and the first observed position lies on x-axis
        For more details, see Argoverse paper
        :param traj: a trajectory with shape nx2
        :return R: rotational matrix with shape 2x2
        :return t: translational vector with shape 2
        """
        pos_s, pos_e = traj[0], traj[self.obs_len-1]
        velocity_v = pos_e - pos_s

        assert np.linalg.norm(velocity_v) > 0.001, 'need to filter out stop tracks'

        # compute R and t
        v1 = velocity_v / np.linalg.norm(velocity_v)
        v2 = np.array([-v1[1], v1[0]])
        R = np.array([v1, v2])
        t = -np.dot(R, pos_e)

        return R, t

    def normalize_track(self, traj, R, t):
        """
        Normalize a trajectory with shape nx2 based on (R, t)
        :param traj: a trajectory with shape nx2
        :param R: rotational matrix with shape 2x2
        :param t: translational vector with shape 2
        :return normalized_traj: normalized trajectory with shape nx2
        """
        normalized_traj = R@traj.transpose() + np.array([t]).transpose()
        normalized_traj = normalized_traj.transpose()

        return normalized_traj

class SubsetSequentialSampler(Sampler):
    """
    Sequential subsampler for dataloader
    Modified from: https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html
    """
    def __init__(self, indices):
        self.num_samples = len(indices)
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self.num_samples

def get_data_loaders(data_dir, obs_len, train_batch_size, val_batch_size, val_ratio=0.2,
    subsampling_factor=1.0, position_downscaling_factor=100.0, num_workers=1):
    """
    Get training and validation dataloaders
    :param data_dir: directory of data files
    :param obs_len: length of observation trajectory
    :param train_batch_size: batch size of training data laoder
    :param val_batch_size: batch size of validation data loader
    :param val_ratio: ratio of validation set among all data
    :param subsampling_factor: data subsampling factor for training and validation set
    :param position_downscaling_factor: downscaling factor for trajectories
    :param num_workers: number of workers for both dataloaders
    :return: (training data loader, validation data loader)
    """
    dataset = ArgoverseDataset(data_dir, obs_len, position_downscaling_factor)

    n_total = len(dataset)
    n_val = int(n_total*val_ratio)
    n_train = n_total - n_val
    train_dataset, val_dataset = data.random_split(dataset, [n_train, n_val])

    val_indices = list(range(int(n_val*subsampling_factor)))
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        num_workers=num_workers,
        sampler=SubsetSequentialSampler(val_indices))

    train_indices = list(range(int(n_train*subsampling_factor)))
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=num_workers,
        sampler=SubsetSequentialSampler(train_indices))

    print('Loading data with {} train samples and {} val samples'.format(len(train_indices), len(val_indices)))
    print('Batching     with {} train batches and {} val batches'.format(len(train_loader), len(val_loader)))

    return train_loader, val_loader

def get_test_data_loader(test_dir, obs_len, test_batch_size,
    subsampling_factor=1.0, position_downscaling_factor=100.0, num_workers=1):
    """
    Get training and validation dataloaders
    :param test_dir: directory of test files
    :param obs_len: length of observation trajectory
    :param test_batch_size: batch size of test data loader
    :param subsampling_factor: data subsampling factor for training and validation set
    :param position_downscaling_factor: downscaling factor for trajectories
    :param num_workers: number of workers for both dataloaders
    :return: test data loader
    """
    dataset = ArgoverseDataset(test_dir, obs_len, position_downscaling_factor)

    n_total = len(dataset)

    test_indices = list(range(int(n_total*subsampling_factor)))
    test_loader = DataLoader(
        dataset,
        batch_size=test_batch_size,
        num_workers=num_workers,
        sampler=SubsetSequentialSampler(test_indices))

    print('Loading test data with {} samples and {} val batches'.format(len(test_indices), len(test_loader)))

    return test_loader

def traj_to_speeds_headings(traj, dt):
    """
    Given a sequence of n samples from a trajectory, compute arrays of speeds and headings for times t = 1 to t = n - 1
    Args:
        traj: the (n x 2) numpy array you get from the agent_traj property from ArgoverseForecastingLoader
        dt: time between samples
    """
    # Traj should be a n x 2 numpy array
    n_samples = traj.shape[0]
    assert traj.shape[1] == 2

    # Compute the offset vectors between positions from time step to time step.
    offset_vectors = np.diff(traj, axis = 0)

    # Compute the distance travelled using the offset vectors and divide by the sampling time to get speed.
    speeds = (1.0/dt) * np.linalg.norm(offset_vectors, axis = 1)

    # Use the dy and dxs from the offset vectors to compute the headings based off the formula:
    #   heading = arctan2(dy, dx)
    headings = np.arctan2(offset_vectors[:, 1], offset_vectors[:, 0])
    assert speeds.size == n_samples - 1
    assert headings.size == n_samples - 1
    return speeds, headings

def traj_to_accels_yawrates(traj, dt):
    """
    Given a sequence of n samples from a trajectory, compute arrays of accels and yawrates for times t = 1 to t = n - 2
    Args:
        traj: the (n x 2) numpy array you get from the agent_traj property from ArgoverseForecastingLoader
        dt: time between samples
    """
    n_samples = traj.shape[0]
    assert traj.shape[1] == 2
    speeds, headings = traj_to_speeds_headings(traj, dt)
    accels = (1.0/dt) * np.diff(speeds)
    yawrates = (1.0/dt) * np.diff(headings)
    assert accels.size == n_samples - 2
    assert yawrates.size == n_samples - 2
    return accels, yawrates, speeds[0], headings[0]
