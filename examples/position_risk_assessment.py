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

import risk_assess.risk_assessors as ra

from utils import *

# fix random seed
torch.manual_seed(0)


def load(test_dir, session_id):
    """
    Test the predictor using saved parameters and models
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

    return model, dataset, params, scale_k, file_name

def main(test_dir, session_id, save_results):
    """
    Test the predictor and assess risk. Running this will output pickle files in
    dataset/test_results and plots in /tmp/argoverse.
    """

    model, dataset, params, scale_k, file_name = load(test_dir, session_id)

    #
    # Specify a method to treat as ground truth and specify a list of methods to test.
    # Each method specification is of the form (name, kwargs). We will collect
    # the data for each method and dump it later.
    #
    ground_truth_method = ('imhof', {'eps_abs' : 1e-10, 'eps_rel' : 1e-10, 'limit' : 500})
    methods = [ground_truth_method, 
               ('ltz', {}),
               ('monte_carlo', {'n_samples' : int(1e4)}),
               ('monte_carlo', {'n_samples' : int(5e4)}),
               ('monte_carlo', {'n_samples' : int(1e5)})]
    results = dict()
    Q = np.asarray(params["ellipse_Q"])


    # Run risk assessment for every data point in the dataset.
    for i in tqdm(range(len(dataset))):
        gmm_traj, _, past_traj_origin = predict(dataset[i], model, scale_k)
        
        # Generate the ego vehicle trajectory.
        ego_xys, thetas = generate_ego_trajectory(past_traj_origin, len(gmm_traj), params["dt"])

        # Transform the prediction into the body frame of the ego vehicle and assess risk.
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

        if save_results:
            # Generate and save a plot and save the results in a pickle file.
            fig = plt.figure("traj_pred_test")
            ax = plt.gca()
            draw_traj(ego_xys, marker="x", color="#FF0000" )
            draw_traj(past_traj_origin.T, marker="s", color="#d33e4c")
            draw_prediction_gmm(ax, gmm_traj)
            plt.legend(['Ego Vehicle Planned Trajectory', 'Agent Observed Trajectory'], fontsize = 14)
            fname = os.path.join("/tmp/argoverse/prediction_viz_{}_{}.png".format(session_id, str(i)))
            fig.tight_layout()
            plt.savefig(fname, dpi=600)
            plt.close(fig)
            with open('dataset/test_results/' + file_name + '.pkl', 'wb') as out_file:
                    pickle.dump(results, out_file, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--test_dir', type=str, default=dir_path + '/../dataset/argoverse_filtered', help='test directory')
    parser.add_argument('--session_id', type=str, default='345ab5bce047400e8a3913784511f547', help='model name')
    parser.add_argument('--save_results', default = True, type = bool)
    args = parser.parse_args()

    main(args.test_dir, args.session_id, args.save_results)
