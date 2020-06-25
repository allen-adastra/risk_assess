"""
Script to test an RNN-based encoder decoder predictor
Usage:
    python tests/test_predicting_positions.py --test_dir dataset/argoverse_train1_filtered --session_id 8e66f991fe824ad49fee0981a01d2090
"""

from __future__ import print_function

from argparse import ArgumentParser
import functools
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
import torch
from tqdm import tqdm
import time
from datetime import datetime
import pickle
import yaml

from prediction.data import ArgoverseDataset
from prediction.loss import loss_ade, loss_nll, loss_std, loss_weight, regularized_loss_nll
from prediction.model import RNNEncoderDecoder
from prediction.utils import load_model
from prediction.visualize import draw_lane_centerlines, draw_traj, draw_prediction_gmm

from risk_assess.random_objects.gmm_trajectory import GmmTrajectory
from risk_assess.random_objects.gmm_control_sequence import GmmControlSequence
from risk_assess.deterministic import simulate_deterministic
import risk_assess.risk_assessors as ra

# fix random seed
torch.manual_seed(0)

def compute_absolute_errors(true_risks, estimated_risks):
    assert len(true_risks) == len(estimated_risks)
    return [abs(est_risk - true_risk) for true_risk, est_risk in zip(true_risks, estimated_risks)]

def compute_relative_errors(true_risks, estimated_risks):
    assert len(true_risks) == len(estimated_risks)
    return [est_risk/true_risk - 1 if true_risk > 0 else 0 for true_risk, est_risk in zip(true_risks, estimated_risks)]

def predict(data, model, scale_k):
    # Load some data and make a prediction off it.
    past_traj, target = data
    past_traj_origin = target['past_traj_origin'] # Past trajectory of the agent.
    past_traj = past_traj.unsqueeze(0) # build a batch with size 1
    prediction = model(past_traj)

    # Unnormalize predictions
    R = target['R']
    t = np.asarray(target['t'])

    gmm_traj = GmmTrajectory.from_prediction(prediction, scale_k)
    
    # Change the frame into the global frame.
    gmm_traj_global = gmm_traj.in_frame(t, R.T)

    gmm_control_seq = GmmControlSequence.from_prediction(prediction)
    
    return gmm_traj_global, gmm_control_seq, past_traj_origin

def generate_ego_trajectory(past_traj_origin, steps, dt):
    # TODO: initial random crap put in
    x0 = past_traj_origin[-1][0] + 2.5
    y0 = past_traj_origin[-1][1] + 2.5
    v0 = 7.0
    theta0 = 0.0
    accels = np.asarray([0.1 for i in range(steps)])
    steers = np.asarray([0.1 for i in range(steps)])
    xs, ys, vs, thetas = simulate_deterministic(x0, y0, v0, theta0, accels, steers, dt)
    ego_xys = np.vstack((xs, ys))
    return ego_xys, thetas

def main(test_dir, session_id, save_results):
    """
    Test the predictor using saved parameters and models
    """

    """
    Configure.
    """
    with open(dir_path + "/params.yaml") as file:
        params = yaml.full_load(file)

    # Output file name
    file_name = datetime.now().strftime("%m%d%Y-%H%M") + "_" + session_id 
    device = 'cpu'
    
    # load parameters and models
    model = RNNEncoderDecoder(device)
    model_info, model = load_model(session_id, model)
    model_args, training_args = model_info['model_args'], model_info['training_args']

    # create data
    scale_k = training_args['position_downscaling_factor']
    dataset = ArgoverseDataset(test_dir, training_args['obs_len'], scale_k)

    # Specify a method to treat as ground truth and specify a list of methods to test.
    # Each method specification is of the form (name, kwargs)
    ground_truth_method = ('imhof', {'eps_abs' : 1e-10, 'eps_rel' : 1e-10, 'limit' : 500})
    methods = [ground_truth_method, 
               ('ltz', {}),
               ('monte_carlo', {'n_samples' : int(1e4)}),
               ('monte_carlo', {'n_samples' : int(5e4)}),
               ('monte_carlo', {'n_samples' : int(1e5)})]
    results = dict()
    Q = np.asarray(params["ellipse_Q"])
    for i in tqdm(range(len(dataset))):
        gmm_traj, _, past_traj_origin = predict(dataset[i], model, scale_k)
        # TODO: transform gmm_traj into ego vehicle body frame

        ego_xys, thetas = generate_ego_trajectory(past_traj_origin, len(gmm_traj), params["dt"])

        body_frame_gmm_traj = gmm_traj.in_body_frame(ego_xys, thetas)

        ground_truth_risks, t_ground_truth = ra.assess_risk_gmms(body_frame_gmm_traj, Q, ground_truth_method[0], **ground_truth_method[1])

        # Assess risk with the test methods.
        for method, method_kwargs in methods:
            risks, t_method = ra.assess_risk_gmms(body_frame_gmm_traj, Q, method, **method_kwargs)
            if method == 'monte_carlo':
                key_name = method + '_' + str(method_kwargs['n_samples'])
            else:
                key_name = method
            if key_name not in results.keys():
                results[key_name] = {'time' : list(), 'risks' : list(), 'max_absolute_error' : list(), 'max_relative_error' : list()}
            results[key_name]['time'].append(t_method)
            results[key_name]['risks'].append(risks)
            results[key_name]['max_absolute_error'].append(max(compute_absolute_errors(ground_truth_risks, risks)))
            results[key_name]['max_relative_error'].append(max(compute_relative_errors(ground_truth_risks, risks)))

        # Generate a plot
        fig = plt.figure("traj_pred_test")
        ax = plt.gca()
        draw_traj(ego_xys, marker="x", color="#FF0000" )
        draw_traj(past_traj_origin.T, marker="s", color="#d33e4c")
        draw_prediction_gmm(ax, gmm_traj)
        plt.legend(['Ego Vehicle Planned Trajectory', 'Agent Observed Trajectory'], fontsize = 14)

        if save_results:
            # Save plots
            fname = os.path.join("/tmp/argoverse/prediction_viz_{}_{}.png".format(session_id, str(i)))
            fig.tight_layout()
            plt.savefig(fname, dpi=600)
            plt.close(fig)
            with open('dataset/test_results/' + file_name + '.pkl', 'wb') as out_file:
                    pickle.dump(results, out_file, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--test_dir', type=str,
                        help='test directory')
    parser.add_argument('--session_id', type=str,
                        help='model name')
    parser.add_argument('--save_results', default = False, type = bool)
    args = parser.parse_args()

    main(args.test_dir, args.session_id, args.save_results)
