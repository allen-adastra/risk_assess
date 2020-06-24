# helper functions to visualize things in argoverse

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

def draw_lane_centerlines(lane_centerlines):
    """
    Modified from argoverse.visualization.visualize_sequences.viz_sequence
    """
    for lane_cl in lane_centerlines:
        plt.plot(lane_cl[:, 0], lane_cl[:, 1], "--", color="grey", alpha=1, linewidth=1, zorder=0)
        plt.text(lane_cl[0, 0], lane_cl[0, 1], "s")
        plt.text(lane_cl[-1, 0], lane_cl[-1, 1], "e")


def draw_traj(traj, marker, color, alpha=1, markersize=6, zorder=15):
    """
    Modified from argoverse.visualization.visualize_sequences.viz_sequence
    """
    plt.plot(traj[:,0], traj[:,1],
        marker,
        color=color,
        alpha=alpha,
        markersize=markersize,
        zorder=zorder,
        )
    plt.xlabel("Map X")
    plt.ylabel("Map Y")

def draw_prediction_gmm(ax, gmm_trajectory):
    """
    Args:
        ax (output of plt.gca())
        gmm_trajectory (instance of GmmTrajectory)
    """
    seq_len = gmm_trajectory._n_steps
    num_component = gmm_trajectory._n_components
    mean_trajs, covariance_trajs, weights = gmm_trajectory.array_rep

    for k in range(num_component):
        mean_traj = mean_trajs[k]
        covariance_traj = covariance_trajs[k]
        draw_traj(mean_traj, marker="o", color='b', alpha=0.6, markersize=1)
        # plot weights
        plt.text(mean_traj[-1, 0], mean_traj[-1, 1], "%.2f" % weights[k])

        # plot uncertainties
        for i in range(seq_len):
            cov = covariance_traj[i]

            [eigval, eigvec] = np.linalg.eig(cov)
            angle = np.arctan2(eigvec[1, 0], eigvec[0, 0]) * 180 / np.pi

            ell_width = np.sqrt(eigval[0]) * 2
            ell_height = np.sqrt(eigval[1]) * 2

            ell = Ellipse(
                [mean_traj[i,0], mean_traj[i,1]],
                width=ell_width,
                height=ell_height,
                angle=angle,
                color='b',
                fill=False)
            ax.add_artist(ell)