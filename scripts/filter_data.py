# usage: python filter_data.py -i ~/intent/argoverse_raw/forecasting_sample/data/ -o ~/intent/argoverse_filtered -l 500

import argparse
from scipy.spatial import distance
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import tqdm
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.centerline_utils import get_normal_and_tangential_distance_point
from argoverse.visualization.visualize_sequences import viz_sequence

from prediction.visualize import draw_lane_centerlines, draw_traj
from prediction.utils import get_nearby_lanes, get_ego_car_traj

# Add prediction/data.py to the path
import os 
import sys
from os import path
sys.path.append(path.dirname(path.abspath(__file__)) + '/../prediction/')
from data import traj_to_accels_yawrates

def filter_nonstop(traj, obs_len=20, min_dist=2.0):
    """
    filter nonstop track
    :return: boolean - true if vehicle has moved for more than *min_dist*
    """
    start, middle, end = traj[0], traj[obs_len-1], traj[-1]

    dist1 = np.sqrt(np.sum((start-end) ** 2))
    dist2 = np.sqrt(np.sum((start-middle) ** 2))

    return dist1 >= min_dist and dist2 >= 0.1

def filter_nonstraight(traj, angle_threshold=0.1):
    """
    filter nonstraight track
    Args:
        traj: trajectory with dimension length x spatial dim
        angle_threshold (radians): we need the two vectors to have at least this difference in heading.
    :return: boolean - true if the track is nonstraight
    """
    l, dim = traj.shape
    start, end, middle = traj[0], traj[-1], traj[l//2]
    vec1 = end - middle
    vec2 = middle - start

    # The magnitude of the angle between the two vectors can be determined by:
    #   cos(angle) = (vec1 dot vec2) / ||vec1|| ||vec2||
    try:
        angle = abs(math.acos(np.dot(vec1, vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2))))
    except:
        angle = 0.0

    return angle >= angle_threshold

def filter_reasonable_accel_yawrates(traj,dt = 0.1, accel_threshold = 10.0, yawrate_threshold = 2.0):
    """
    Args:
        traj:
        dt (s): time in between samples of traj
        accel_threshold (m/s^2): the maximum acceleration in the traj must be less than this value
        yawrate_threshold (rad/s): the maximum yawrate in the traj mmust be less than this value
    """
    accels, yawrates = traj_to_accels_yawrates(traj, dt)
    accels_reasonable = max(np.abs(accels)) <= accel_threshold
    yawrates_reasonable = max(np.abs(yawrates)) <= yawrate_threshold
    return accels_reasonable and yawrates_reasonable

def filter_lane_change(traj, lanes):
    """
    Args:
        traj: target vehicle trajectory
        lanes: coordinates of nearby lanes relative to the target vehicle
    """

    # compute the distance to the closest lane from start and end positions
    i_start, d_start = -1, 100000
    i_end, d_end = -1, 100000
    for i, lane in enumerate(lanes):
        # ignore lane if it is too short
        if len(lane) < 20:
            continue

        # compute distance to start point
        d_s_t, d_s_n = get_normal_and_tangential_distance_point(traj[0][0], traj[0][1], lane)
        d_s_n = abs(d_s_n)
        if d_s_n < d_start:
            i_start = i
            d_start = d_s_n

        d_e_t, d_e_n = get_normal_and_tangential_distance_point(traj[-1][0], traj[-1][1], lane)
        d_e_n = abs(d_e_n)
        if d_e_n < d_end:
            i_end = i
            d_end = d_e_n

    # return true if the closest lane to start and the closest lane to end is the same
    if i_start == i_end:
        return False
    else:
        return True

def filter_turning(traj, theta_buffer = 20):
    """
    Args:
        traj: target vehicle trajectory
        theta_diff_threshold (degrees): buffer to decide whether a turn is valid
    """
    # get start and end orientations of the target vehicle in degrees
    theta_start = np.arctan2(traj[4][1]-traj[0][1], traj[4][0]-traj[0][0])/np.pi*180
    theta_end   = np.arctan2(traj[-1][1]-traj[-4][1], traj[-1][0]-traj[-4][0])/np.pi*180
    theta_diff  = theta_end - theta_start

    # normalize angle in degrees to [0, 360]
    while theta_diff < 0:
        theta_diff += 360
    while theta_diff >= 360:
        theta_diff -= 360

    # check left turn
    if theta_diff >= 90-theta_buffer and theta_diff <= 90+theta_buffer:
        return True

    # check right turn
    elif theta_diff >= 270-theta_buffer and theta_diff <= 270+theta_buffer:
        return True

    return False

def filter_close_interaction(traj, ego_traj, interaction_threshold=20.0):
    """
    Args:
        traj: target vehicle trajectory
        ego_traj: ego vehicle trajectory
        interaction_threshold: maximum distance that is allowed between two
                               interactive vehicles
    """
    # skip if ego car does not move
    delta_d_ego = np.sum((ego_traj[-1] - ego_traj[0]) ** 2)
    if np.sqrt(delta_d_ego) <= 2: return False

    # compute pairwise distance between two trajectories
    distance = (traj - ego_traj) ** 2
    closest_distance = np.sqrt(np.min(np.sum(distance, axis=1)))
    return closest_distance <= interaction_threshold

def filter_argoverse_data(input_dir, output_dir, convert_limit=0, filters=[], visualize=False):
    """
    filter argoverse data
    :param input_dir: directory of input files (usually downlowaded from argoverse website)
    :param output_dir: directory of filtered files
    :param convert_limit: maximum number of files to save. (0 means save all filtered files)
    :return: none
    """
    afl = ArgoverseForecastingLoader(input_dir)
    avm = ArgoverseMap()

    # create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    track_counter = 0
    print('Extracting {} files'.format(len(afl)))

    items = [f.__name__ for f in filters]
    statistics = {item:[] for item in items}

    for data in tqdm.tqdm(afl):
        # get trajectory and city of the target vehicle
        traj = data.agent_traj
        city = data.seq_df["CITY_NAME"][0]

        # get trajectory of (ego) autonomous vehicle
        av_traj = get_ego_car_traj(data)

        # go through filters and save data if all filters return True
        pass_all_filters = True
        for f in filters:
            if f == filter_lane_change:
                # get lanes (for visualization, TODO: add to features)
                lanes = get_nearby_lanes(avm, traj, city)

                result = f(traj, lanes)
            elif f == filter_close_interaction:
                result = f(traj, av_traj)
            else:
                result = f(traj)

            if not result:
                pass_all_filters = False
                statistics[f.__name__].append(0)
            else:
                statistics[f.__name__].append(1)

        if pass_all_filters:
            track_counter += 1

            # create symlink for filtered data
            file_name = os.path.split(data.current_seq)[-1]
            path_src = os.path.join(input_dir, file_name)
            path_dst = os.path.join(output_dir, file_name)
            os.symlink(path_src, path_dst)

        if visualize and pass_all_filters:
            fig, ax = plt.subplots()
            plt.axis('equal')
            draw_lane_centerlines(lanes)
            draw_traj(traj, marker="o", color="#d33e4c", alpha=0.5)
            draw_traj(av_traj, marker="s", color='b', alpha=0.4)

            if not os.path.exists("/tmp/argoverse/"):
                os.mkdir("/tmp/argoverse/")
            file_id = os.path.split(data.current_seq)[-1].split('.')[0]
            label = filters[0].__name__
            fname = os.path.join("/tmp/argoverse/prediction_viz_{}_{}.png".format(str(file_id), label))
            fig.tight_layout()
            plt.savefig(fname, dpi=600)
            plt.close(fig)

        if convert_limit > 0 and track_counter >= convert_limit:
            break

    for item in statistics:
        print('Stats - {}: {}/{}'.format(item, np.sum(statistics[item]), len(statistics[item])))
    import IPython; IPython.embed(header='done filtering')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--input_dir', help='input data files to read')
    parser.add_argument('-o', '--output_dir', help='filtered data symlinks to write')
    parser.add_argument('-l', '--limit', type=int, default=0, help='maximum number of files to save')
    parser.add_argument('-v', '--visualize', default=False, help='visualize tracks')

    args = parser.parse_args()
    # filters = [filter_nonstop, filter_nonstraight, filter_reasonable_accel_yawrates] # TODO: add this as an argument
    filters = [filter_nonstop, filter_nonstraight]

    filter_argoverse_data(input_dir=args.input_dir,
        output_dir=args.output_dir,
        convert_limit=args.limit,
        filters=filters,
        visualize=args.visualize)

if __name__ == '__main__':
    main()
