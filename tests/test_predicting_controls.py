"""
Script to test an RNN-based encoder decoder control predictor
Usage:
    python tests/test_predicting_controls.py --test_dir dataset/argoverse_filtered_5000/ --session_id 24d694464a6742208f6cca0668f7bad0
"""

from __future__ import print_function

from argparse import ArgumentParser
import functools
import logging
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import torch
from tqdm import tqdm
from datetime import datetime
import pickle

from prediction.data import ArgoverseDataset
from prediction.loss import loss_ade, loss_nll, loss_std, loss_weight, regularized_loss_nll
from prediction.model import RNNEncoderDecoder
from prediction.utils import load_model, propogate_controls_to_trajectories
from prediction.visualize import draw_lane_centerlines, draw_traj, draw_prediction_gmm

from risk_assess.random_objects.gmm_control_sequence import GmmControlSequence
from risk_assess.deterministic import CarState, simulate_deterministic
from risk_assess.uncertain_agent.state_objects import AgentMomentState
from risk_assess.uncertain_agent.moment_dynamics import propagate_moments
from risk_assess.geom_utils import Ellipse
from risk_assess.risk_assessor import RiskAssessor
# fix random seed
torch.manual_seed(0)

def main(test_dir, session_id, save_results):
    """
    Test the predictor using saved parameters and models
    """
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # load parameters and models
    model = RNNEncoderDecoder(device)
    model_info, model = load_model(session_id, model)
    model_args, training_args = model_info['model_args'], model_info['training_args']
    model.to(device)

    # create data
    scale_k = training_args['position_downscaling_factor']
    dataset = ArgoverseDataset(test_dir, training_args['obs_len'], scale_k)
    dt = 0.1
    n_half_spaces = 12

    # Set number of Monte Carlo samples to use.
    n_samps = int(1e4)
    mc_idx = "monte_carlo_" + '{:.0e}'.format(n_samps)

    # Output dictionary.
    results = {"chebyshev_halfspace" : {"risks" : list(), "times" : list()},
               mc_idx : {"risks" : list(), "times" : list()}}

    # Output file name
    file_name = "test_predict_controls_" + datetime.now().strftime("%m%d%Y-%H%M") + "_" + session_id 

    for i in tqdm(range(len(dataset))):
        past_traj, target = dataset[i]
        past_traj = past_traj.unsqueeze(0) # build a batch with size 1
        past_traj = past_traj.to(device)
        prediction = model(past_traj)

        # Frame change params.
        R = target['R']
        offset_theta = -math.atan2(R[1][0], R[0][0]) # Agent heading in global frame is offset by this many radians.

        # Initial state of the agent.
        agent_vel0_scaled = target['vel0'] * scale_k
        agent_theta0 = target['theta0'] + offset_theta
        past_traj_origin = target['past_traj_origin'] # past_traj_origin is global frame

        # HACK: start the agent at origin.
        agent_x0 = 0
        agent_y0 = 0

        initial_agent_state = AgentMomentState.from_deterministic_state(agent_x0, agent_y0, agent_vel0_scaled, agent_theta0)

        # A hard coded control sequence and initial state to produce a trajectory for the ego vehicle.
        initial_ego_state = CarState(agent_x0 + 3.0, agent_y0 + 3.0, 5.0, 0.0)
        accels = np.asarray([0.0 for i in range(len(prediction))])
        steers = np.asarray([0.1 for i in range(len(prediction))])
        car_frame_ellipse = Ellipse(3, 2, 0, 0, 0)

        # Simulate the trajectory of the ego vehicle.
        xs, ys, vs, thetas = simulate_deterministic(initial_ego_state.x, initial_ego_state.y, initial_ego_state.v, initial_ego_state.theta, accels, steers, dt)

        # Initialize PlanVerifier and an instance of GmmControlSequence
        ra = RiskAssessor(xs.tolist()[0], ys.tolist()[0], vs.tolist()[0], thetas.tolist()[0], car_frame_ellipse)
        gmm_control_seq = GmmControlSequence.from_prediction(prediction)

        # Assess risk with the Monte Carlo and the Chebyshev methods.
        monte_carlo_risks, agent_xs, agent_ys, t_monte_carlo = ra.control_assess_risk_monte_carlo(gmm_control_seq, agent_x0, agent_y0, agent_vel0_scaled, agent_theta0, n_samps, dt)
        
        moments_risks, t_moments = ra.control_assess_risk_chebyshev_halfspace(gmm_control_seq, initial_agent_state, n_half_spaces, dt)

        results[mc_idx]["risks"].append(monte_carlo_risks)
        results[mc_idx]["times"].append(t_monte_carlo)

        results["chebyshev_halfspace"]["risks"].append(moments_risks)
        results["chebyshev_halfspace"]["times"].append(t_moments)

        fig = plt.figure("traj_pred_test")
        ax = plt.gca()
        
        # save plots
        if save_results:
            fname = os.path.join("/tmp/argoverse/prediction_viz_{}_{}.png".format(session_id, str(i)))
            fig.tight_layout()
            plt.axis('equal')
            plt.savefig(fname, dpi=600)
            plt.close(fig)

            with open('dataset/test_results/' + file_name + '.pkl', 'wb') as out_file:
                pickle.dump(results, out_file, pickle.HIGHEST_PROTOCOL)
        
        del pv

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--test_dir', type=str,
                        help='test directory')
    parser.add_argument('--session_id', type=str,
                        help='model name')
    parser.add_argument('--save_results', default = False, type = bool)

    args = parser.parse_args()

    main(args.test_dir, args.session_id, args.save_results)
