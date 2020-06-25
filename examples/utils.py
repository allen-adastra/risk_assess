import numpy as np
from risk_assess.random_objects.gmm_trajectory import GmmTrajectory
from risk_assess.random_objects.gmm_control_sequence import GmmControlSequence
from risk_assess.deterministic import simulate_deterministic

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
    """ Generate an ego vehicle trajectory in the vicinity of the agent.

    Args:
        past_traj_origin ([type]): [description]
        steps ([type]): [description]
        dt ([type]): [description]

    Returns:
        [type]: [description]
    """
    x0 = past_traj_origin[-1][0] + 2.5
    y0 = past_traj_origin[-1][1] + 2.5
    v0 = 7.0
    theta0 = 0.0
    accels = np.asarray([0.1 for i in range(steps)])
    steers = np.asarray([0.1 for i in range(steps)])
    xs, ys, vs, thetas = simulate_deterministic(x0, y0, v0, theta0, accels, steers, dt)
    ego_xys = np.vstack((xs, ys))
    return ego_xys, thetas

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