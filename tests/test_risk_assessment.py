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
import torch
from tqdm import tqdm
import time
from datetime import datetime
import pickle

from prediction.data import ArgoverseDataset
from prediction.loss import loss_ade, loss_nll, loss_std, loss_weight, regularized_loss_nll
from prediction.model import RNNEncoderDecoder
from prediction.utils import load_model
from prediction.visualize import draw_lane_centerlines, draw_traj, draw_prediction_gmm

from risk_assess.random_objects.gmm_trajectory import GmmTrajectory
# from risk_assess.geom_utils import Ellipse
from risk_assess.risk_assessor import RiskAssessor
# fix random seed
torch.manual_seed(0)

def compute_absolute_errors(true_risks, estimated_risks):
    assert len(true_risks) == len(estimated_risks)
    return [abs(est_risk - true_risk) for true_risk, est_risk in zip(true_risks, estimated_risks)]

def compute_relative_errors(true_risks, estimated_risks):
    assert len(true_risks) == len(estimated_risks)
    return [est_risk/true_risk - 1 if true_risk > 0 else 0 for true_risk, est_risk in zip(true_risks, estimated_risks)]

def main(test_dir, session_id, save_results):
    """
    Test the predictor using saved parameters and models
    """
    # Time step size
    dt = 0.1

    # Output file name
    file_name = datetime.now().strftime("%m%d%Y-%H%M") + "_" + session_id 

    # Device to use
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
    ground_truth_method = ('imhof', {'eps_abs' : 1e-10, 'eps_rel' : 1e-10})
    methods = [ground_truth_method, 
               ('noncentral_chisquare_cqf', {}),
               ('noncentral_chisquare', {}),
               ('monte_carlo', {'n_samples' : int(1e4)}),
               ('monte_carlo', {'n_samples' : int(5e4)}),
               ('monte_carlo', {'n_samples' : int(1e5)})]
    results = dict()

    # for i in tqdm(range(len(dataset))):
    #     # Load some data and make a prediction off it.
    #     past_traj, target = dataset[i]
    #     past_traj_origin = target['past_traj_origin'] # Past trajectory of the agent.
    #     past_traj = past_traj.unsqueeze(0) # build a batch with size 1
    #     prediction = model(past_traj)

    #     # Unnomralize predictions
    #     R = target['R']
    #     t = np.asarray(target['t']).reshape((2, 1)) # This should be a column array
    #     gmm_traj = GmmTrajectory.from_prediction(prediction, scale_k)
    #     gmm_traj.change_frame(t, R.T)
    #     gmm_traj.save_as_matfile("/home/allen/Desktop", "test.mat")
        
    #     # TODO: initial random crap put in
    #     initial_state = CarState(past_traj_origin[-1][0] + 2.5, past_traj_origin[-1][1] + 2.5, 7.0, 0.0)
    #     accels = np.asarray([0.1 for i in range(len(gmm_traj))])
    #     steers = np.asarray([0.1 for i in range(len(gmm_traj))])
    #     car_frame_ellipse = Ellipse(3, 2, 0, 0, 0)
    #     xs, ys, vs, thetas = simulate_deterministic(initial_state.x, initial_state.y, initial_state.v, initial_state.theta, accels, steers, dt)
    #     pv = RiskAssessor(xs.tolist()[0], ys.tolist()[0], vs.tolist()[0], thetas.tolist()[0], car_frame_ellipse)
    #     xs = pv.xs
    #     ys = pv.ys
    #     ego_traj = np.vstack((xs, ys)).T

    #     # Assess the risk with the ground truth method.
    #     ground_truth_risks, t_ground_truth_method = pv.assess_risk_gmms(gmm_traj, ground_truth_method[0], **ground_truth_method[1])
    #     # Assess risk with the test methods.
    #     for method, method_kwargs in methods:
    #         risks, t_method = pv.assess_risk_gmms(gmm_traj, method, **method_kwargs)
    #         if method == 'monte_carlo':
    #             key_name = method + '_' + str(method_kwargs['n_samples'])
    #         else:
    #             key_name = method
    #         if key_name not in results.keys():
    #             results[key_name] = {'time' : list(), 'risks' : list(), 'max_absolute_error' : list(), 'max_relative_error' : list()}
    #         result_dic = results[key_name]
    #         result_dic['time'].append(t_method)
    #         result_dic['risks'].append(risks)
    #         result_dic['max_absolute_error'].append(max(compute_absolute_errors(ground_truth_risks, risks)))
    #         result_dic['max_relative_error'].append(max(compute_relative_errors(ground_truth_risks, risks)))

    #     # Generate a plot
    #     fig = plt.figure("traj_pred_test")
    #     ax = plt.gca()
    #     draw_traj(ego_traj, marker="x", color="#FF0000" )
    #     draw_traj(past_traj_origin, marker="s", color="#d33e4c")
    #     draw_prediction_gmm(ax, gmm_traj)
    #     plt.legend(['Ego Vehicle Planned Trajectory', 'Agent Observed Trajectory'], fontsize = 14)
    #     if save_results:
    #         # Save plots
    #         fname = os.path.join("/tmp/argoverse/prediction_viz_{}_{}.png".format(session_id, str(i)))
    #         fig.tight_layout()
    #         plt.savefig(fname, dpi=600)
    #         plt.close(fig)
    #         with open('dataset/test_results/' + file_name + '.pkl', 'wb') as out_file:
    #                 pickle.dump(results, out_file, pickle.HIGHEST_PROTOCOL)
    #     # Clean up...
    #     del pv

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--test_dir', type=str,
                        help='test directory')
    parser.add_argument('--session_id', type=str,
                        help='model name')
    parser.add_argument('--save_results', default = False, type = bool)
    args = parser.parse_args()

    main(args.test_dir, args.session_id, args.save_results)
