# utility function

import io
import numpy as np
import os
import torch
from torch import nn

def get_nearby_lanes(avm, traj, city):
    """
    Get nearby centerlines that are close to a given trajectory
    :param avm: ArgoverseMap class
    :param traj: input trajectory
    :param city: city of the trajectory
    """
    candidate_centerlines = avm.get_candidate_centerlines_for_traj(traj, city, viz=False)

    lanes = []
    for lane_centerline in candidate_centerlines:
        lane = lane_centerline.tolist()
        lanes.append(np.array(lane))

    return lanes

def get_ego_car_traj(data):
    """
    Get ego car trajectory from a pandas DataFrame
    :param data: a pandas DataFrame containing ego car trajectory
    """
    ego_x = data.seq_df[data.seq_df["OBJECT_TYPE"] == "AV"]["X"]
    ego_y = data.seq_df[data.seq_df["OBJECT_TYPE"] == "AV"]["Y"]
    ego_traj = np.column_stack((ego_x, ego_y))
    return ego_traj

def save_model(session_id,
               model,
               model_args=None,
               training_args=None,
               loss=None,
               epoch=None,):
    # define directory to save the model
    session_dir = os.path.join('models', session_id)
    try:
        os.mkdir(session_dir)
    except FileExistsError:
        print('Need to create models directory')
        pass

    # save parameters
    info = {
        'model_args': model_args,
        'training_args': training_args,
        'loss': loss,
        'epoch': epoch,
    }
    fname_params = os.path.join(session_dir, 'params.pth')
    torch.save(info, fname_params)

    # save model
    fname = os.path.join(session_dir, 'model.pth')
    print("Saving model to {}".format(fname))
    torch.save(model.state_dict(), fname)


def load_model(session_id, model):
    # load model parameters
    session_dir = os.path.join('models', session_id)

    if not os.path.isdir(session_dir):
        raise ValueError("Bad session directory! [{}]".format(session_dir))

    fname_info = os.path.join(session_dir, 'params.pth')
    if not os.path.isfile(fname_info):
        raise ValueError("Bad params filename! [{}]".format(fname_info))

    with io.open(fname_info, 'rb') as f:
        info = torch.load(f)

    fname_model = os.path.join(session_dir, 'model.pth')
    if os.path.isfile(fname_model):
        if torch.cuda.is_available():
            model_resumed = torch.load(fname_model)
        else:
            model_resumed = torch.load(fname_model, map_location='cpu')
    else:
        print("Could not load state_dict")
        model_resumed = None

    if isinstance(model, torch.nn.Module):
        model.load_state_dict(model_resumed, strict=False)
    else:
        print("Could not load model")

    return info, model


def init_weights(m):
    """
    Initialize weights for linear layers using Kaiming normal
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def propogate_controls_to_trajectories(prediction, vel0, theta0, dt, scale_k, R, t):
    """
    Given a sequence of predicted control signals and transformation information,
    reconstruct nominal trajectories
    Args:
        prediction: a dictionary containing control predictions as GMM parameters
        vel0: initial velocity
        theta0: initial heading
        dt: time between samples
        scale_k: scaling factor
        R,t: transformation matrices
    Return:
        w_trajs: weights associated with each nominal trajectory
        trajs: reconstructed nominal trajectory
    """
    # extract prediction values
    mus_acc = [p['mus_acc'].tolist()[0] for p in prediction]
    weights_acc = [p['lweights_acc'].exp().tolist()[0] for p in prediction]
    mus_acc, weights_acc = np.array(mus_acc), np.array(weights_acc)[0]

    mus_alpha = [p['mus_alpha'].tolist()[0] for p in prediction]
    weights_alpha = [p['lweights_alpha'].exp().tolist()[0] for p in prediction]
    mus_alpha, weights_alpha = np.array(mus_alpha), np.array(weights_alpha)[0]
    # recontruct trajectories for all combos of acc and heading
    mus_n = weights_acc.shape[0]
    alphas_n = weights_alpha.shape[0]
    w_trajs = []
    trajs = []
    for i in range(mus_n):
        for j in range(alphas_n):
            w_acc, w_alpha = weights_acc[i], weights_alpha[j]
            mu_acc, mu_alpha = mus_acc[:,i], mus_alpha[:,j]

            vel = np.cumsum(mu_acc*dt) + vel0
            theta = np.cumsum(mu_alpha*dt) + theta0

            traj_x = np.cumsum(vel*np.cos(theta)*dt)
            traj_y = np.cumsum(vel*np.sin(theta)*dt)
            traj   = np.vstack((traj_x, traj_y))

            # unnormalize traj
            traj = traj.T * scale_k
            traj = traj - t
            traj = np.linalg.inv(R) @ traj.T

            w_trajs.append(w_acc*w_alpha)
            trajs.append(traj)

    return w_trajs, trajs
