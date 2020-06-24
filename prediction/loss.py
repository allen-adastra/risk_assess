"""
Define model losses
"""
import torch
import numpy as np

def loss_ade_mtp(y_pred, ys, position_downscaling_factor=100):
    '''
    Compute weighted average displacement error between acausal trajectory and the best predicted trajectory
    :param y_pred: a set of GMM parameters including
        'lweights': component log weights with shape batch x num_component
        'mus':      component means with shape batch x num_component x spatial_dim
        'lsigs':    component log stds with shape batch x num_component x spatial_dim
    :param ys: a set of targets including
        'traj':  acausal trajectory with shape batch x n x spatial_dim
        'acc':   acausal acceleration with shape batch x n x spatial_dim
        'alpha': acausal angular rate with shape batch x n x spatial_dim
    '''
    spatial_dim = ys['traj'].shape[-1]
    mses = []

    # compute displacement error at each horizon and take average over all batch items
    for i, y_p in enumerate(y_pred):
        y = ys['traj'][:,i]

        y_p_w = y_p['lweights']
        y_p_m = y_p['mus']
        num_component = y_p_w.shape[1]
        mse = 0
        errors = []
        for k in range(num_component):
            diff = (y_p_m[:,k] - y) * position_downscaling_factor # we need to upscale the difference
            error = torch.norm(diff,p=2,dim=1)
            errors.append(error)

        mses.append(torch.stack(errors))

    # for each batch item, select the mode with the best loss
    mses_avg = torch.stack(mses).mean(dim=0)
    mses_best = mses_avg.min(dim=1)[0]

    # return average among all batch items
    return mses_best.mean()

def loss_ade(y_pred, ys, position_downscaling_factor=100):
    '''
    Compute weighted average displacement error between acausal trajectory and predicted trajectory
    :param y_pred: a set of GMM parameters including
        'lweights': component log weights with shape batch x num_component
        'mus':      component means with shape batch x num_component x spatial_dim
        'lsigs':    component log stds with shape batch x num_component x spatial_dim
    :param ys: a set of targets including
        'traj':  acausal trajectory with shape batch x n x spatial_dim
        'acc':   acausal acceleration with shape batch x n x spatial_dim
        'alpha': acausal angular rate with shape batch x n x spatial_dim
    '''
    spatial_dim = ys['traj'].shape[-1]
    mses = []

    # compute displacement error at each horizon and take average over all batch items
    for i, y_p in enumerate(y_pred):
        y = ys['traj'][:,i]

        y_p_w = y_p['lweights']
        y_p_m = y_p['mus']
        num_component = y_p_w.shape[1]
        mse = 0
        for k in range(num_component):
            diff = (y_p_m[:,k] - y) * position_downscaling_factor # we need to upscale the difference
            mse += y_p_w[k].exp() * torch.mean(torch.norm(diff,p=2,dim=1))
        mses.append(mse)

    return torch.stack(mses).mean()

def loss_fde(y_pred, ys, position_downscaling_factor=100):
    '''
    Compute weighted final displacement error between acausal trajectory and predicted trajectory
    :param y_pred: a set of GMM parameters including
        'lweights': component log weights with shape batch x num_component
        'mus':      component means with shape batch x num_component x spatial_dim
        'lsigs':    component log stds with shape batch x num_component x spatial_dim
    :param ys: a set of targets including
        'traj':  acausal trajectory with shape batch x n x spatial_dim
        'acc':   acausal acceleration with shape batch x n x spatial_dim
        'alpha': acausal angular rate with shape batch x n x spatial_dim
    '''
    spatial_dim = ys['traj'].shape[-1]
    fdes = []

    # compute displacement error at each horizon and take average over all batch items
    for i, y_p in enumerate(y_pred):
        y = ys['traj'][:,i]

        y_p_w = y_p['lweights']
        y_p_m = y_p['mus']
        num_component = y_p_w.shape[1]
        fde = 0
        for k in range(num_component):
            diff = (y_p_m[:,k] - y) * position_downscaling_factor # we need to upscale the difference
            fde += y_p_w[k].exp() * torch.mean(torch.norm(diff[-1:],p=2,dim=1))
        fdes.append(fde)

    return torch.stack(fdes).mean()

def loss_nll_mtp(y_pred, ys, position_downscaling_factor=100):
    '''
    Compute average negative log-likelihoods (NLL) of acausal trajectory over the best predicted GMM distribution
    :param y_pred: a set of GMM parameters including
        'lweights': component log weights with shape batch x num_component
        'mus':      component means with shape batch x num_component x spatial_dim
        'lsigs':    component log stds with shape batch x num_component x spatial_dim
    :param ys: a set of targets including
        'traj':  acausal trajectory with shape batch x n x spatial_dim
        'acc':   acausal acceleration with shape batch x n x spatial_dim
        'alpha': acausal angular rate with shape batch x n x spatial_dim
    '''
    spatial_dim = ys['traj'].shape[-1]
    nllikes = []
    nlls = []

    # compute nll for i-th predicted distribution
    for i, y_p in enumerate(y_pred):
        y = ys['traj'][:,i]

        y_p_w = y_p['lweights']
        y_p_m = y_p['mus'] * position_downscaling_factor
        y_p_s = y_p['lsigs']
        # y_p_s = torch.ones(y_p_s.shape).to(y.device)

        y = torch.unsqueeze(y, 1).expand_as(y_p_m) * position_downscaling_factor

        # compute log-likelihood for individual components
        # see slide 2: at https://mas-dse.github.io/DSE210/Additional%20Materials/gmm.pdf
        # center by substracting offset
        y_diff = y - y_p_m
        exponentials = (y_diff / (y_p_s.exp()*position_downscaling_factor)).pow(2) / 2
        scalars = np.log(2 * np.pi) / 2 + y_p_s + np.log(position_downscaling_factor)
        llikes = y_p_w - scalars.sum(dim=2) - exponentials.sum(dim=2)

        nllikes.append(-llikes)

    # for each batch item, select the mode with the best weight
    nllikes_avg = torch.stack(nllikes).mean(dim=0)
    nllikes_best = nllikes_avg.min(dim=1)[0]

    # return average among all batch items
    return nllikes_best.mean()

def loss_nll(y_pred, ys, position_downscaling_factor=100):
    '''
    Compute average negative log-likelihoods (NLL) of acausal trajectory over predicted GMM distribution
    :param y_pred: a set of GMM parameters including
        'lweights': component log weights with shape batch x num_component
        'mus':      component means with shape batch x num_component x spatial_dim
        'lsigs':    component log stds with shape batch x num_component x spatial_dim
    :param ys: a set of targets including
        'traj':  acausal trajectory with shape batch x n x spatial_dim
        'acc':   acausal acceleration with shape batch x n x spatial_dim
        'alpha': acausal angular rate with shape batch x n x spatial_dim
    '''
    spatial_dim = ys['traj'].shape[-1]
    nlls = []

    # compute nll for i-th predicted distribution
    for i, y_p in enumerate(y_pred):
        y = ys['traj'][:,i]

        y_p_w = y_p['lweights']
        y_p_m = y_p['mus'] * position_downscaling_factor
        y_p_s = y_p['lsigs']
        # y_p_s = torch.ones(y_p_s.shape).to(y.device)

        y = torch.unsqueeze(y, 1).expand_as(y_p_m) * position_downscaling_factor

        # compute log-likelihood for individual components
        # see slide 2: at https://mas-dse.github.io/DSE210/Additional%20Materials/gmm.pdf
        # center by substracting offset
        y_diff = y - y_p_m
        exponentials = (y_diff / (y_p_s.exp()*position_downscaling_factor)).pow(2) / 2
        scalars = np.log(2 * np.pi) / 2 + y_p_s + np.log(position_downscaling_factor)
        llikes = y_p_w - scalars.sum(dim=2) - exponentials.sum(dim=2)

        # compute overall log-likelihood by finding the maximum cluster likelihood to avoid numerical problems
        # see: https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
        maxv0, maxi = llikes.max(dim=1, keepdim=True)
        llikes_ = torch.logsumexp(llikes - maxv0, dim=1) + maxv0.squeeze()

        # add negative nll
        nlls.append(-1.0*llikes_)

    # print(y_p['lweights'])

    return torch.stack(nlls).mean()

def loss_std(y_pred, ys, std_mean=1.0, position_downscaling_factor=100):
    '''
    Compute regularized std loss
    :param y_pred: a set of GMM parameters including
        'lweights': component log weights with shape batch x num_component
        'mus':      component means with shape batch x num_component x spatial_dim
        'lsigs':    component log stds with shape batch x num_component x spatial_dim
    :param std_mean: desired value of std_mean
    '''
    std_losses = []
    for y_p in y_pred:
        y_p_s = y_p['lsigs']
        std   = torch.ones(y_p_s.shape)*std_mean
        std   = std.to(y_p_s.device)

        # compute mse between predicted std and expected std for each batch
        diff  = y_p_s.exp()*position_downscaling_factor - std
        diff_se = (diff**2).view(diff.shape[0],-1)
        diff_mse = diff_se.mean()

        std_losses.append(diff_mse)
    return torch.stack(std_losses).mean()

def loss_weight(y_pred, ys):
    '''
    Compute regularized weight loss
    :param y_pred: a set of GMM parameters including
        'lweights': component log weights with shape batch x num_component
        'mus':      component means with shape batch x num_component x spatial_dim
        'lsigs':    component log stds with shape batch x num_component x spatial_dim
    '''

    # encourage weight to be nonzero
    loss = y_pred[0]['lweights']
    loss = (loss.exp()**2).mean()
    return loss

def regularized_loss_nll(y_pred, ys,
    std_regularization_factor=0.0, std_mean=1.0,
    weight_regularization_factor=0.0,
    ade_regularization_factor=0.0,
    control_regularization_factor=0.0,
    position_downscaling_factor=100.0):
    '''
    Compute regularized negative log-likelihoods (NLL) loss
    :param y_pred: a set of GMM parameters including
        'lweights': component log weights with shape batch x num_component
        'mus':      component means with shape batch x num_component x spatial_dim
        'lsigs':    component log stds with shape batch x num_component x spatial_dim
    :param ys: a set of targets including
        'traj':  acausal trajectory with shape batch x n x spatial_dim
        'acc':   acausal acceleration with shape batch x n x spatial_dim
        'alpha': acausal angular rate with shape batch x n x spatial_dim
    :param lstd_regularization_factor: importance weight of lstd regularization
    :param std_mean: desired value of std_mean
    :param lweight_regularization_factor: importance weight of lweight regularization
    '''
    result = loss_nll(y_pred, ys)

    if std_regularization_factor > 0:
        # add an additional loss for lstd to prevent it from being too large or too small
        result += std_regularization_factor*loss_std(y_pred, None, std_mean, position_downscaling_factor)

    if weight_regularization_factor > 0:
        # add an additional loss for lweight
        result += weight_regularization_factor*loss_weight(y_pred, None)

    if ade_regularization_factor > 0:
        result += ade_regularization_factor*loss_ade(y_pred, ys, position_downscaling_factor)

    if control_regularization_factor > 0:
        control_result = regularized_loss_nll_control(y_pred, ys,
                            std_regularization_factor=std_regularization_factor,
                            weight_regularization_factor=weight_regularization_factor,
                            ade_regularization_factor=ade_regularization_factor)
        result += control_regularization_factor * control_result

    # import IPython; IPython.embed(header='nll')
    return result

###############################
# loss functions for controls
###############################

def loss_nll_control(y_pred, ys, control_type='', position_downscaling_factor=100):
    '''
    Compute average negative log-likelihoods (NLL) of acausal control signals
        over predicted GMM distribution
    :param y_pred: a set of GMM parameters including
        'lweights': component log weights with shape batch x num_component
        'mus':      component means with shape batch x num_component
        'lsigs':    component log stds with shape batch x num_component
    :param ys: a set of targets including
        'traj':  acausal trajectory with shape batch x n
        'acc':   acausal acceleration with shape batch x n
        'alpha': acausal angular rate with shape batch x n
    '''
    spatial_dim = ys[control_type].shape[-1]
    nlls = []
    n_control = ys[control_type].shape[1]

    # compute nll for i-th predicted distribution
    for i in range(n_control):
        y_p = y_pred[i]
        y = ys[control_type][:,i]

        y_p_w = y_p['lweights_'+control_type]
        y_p_m = y_p['mus_'+control_type]
        y_p_s = y_p['lsigs_'+control_type]
        y = torch.unsqueeze(y, 1).expand_as(y_p_m)

        # compute log-likelihood for individual components
        # see slide 2: at https://mas-dse.github.io/DSE210/Additional%20Materials/gmm.pdf
        # center by substracting offset
        y_diff = y - y_p_m
        exponentials = (y_diff / (y_p_s.exp())).pow(2) / 2
        scalars = np.log(2 * np.pi) / 2 + y_p_s
        llikes = y_p_w - scalars - exponentials

        # compute overall log-likelihood by finding the maximum cluster likelihood to avoid numerical problems
        # see: https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
        maxv0, maxi = llikes.max(dim=1, keepdim=True)
        llikes_ = torch.logsumexp(llikes - maxv0, dim=1) + maxv0.squeeze()

        # add negative nll
        nlls.append(-1.0*llikes_)

    return torch.stack(nlls).mean()

def loss_l2_control(y_pred, ys, control_type=''):
    '''
    Compute weighted l2 error between acausal controls and predicted controls
    :param y_pred: a set of GMM parameters including
        'lweights': component log weights with shape batch x num_component
        'mus':      component means with shape batch x num_component
        'lsigs':    component log stds with shape batch x num_component
    :param ys: a set of targets including
        'traj':  acausal trajectory with shape batch x n
        'acc':   acausal acceleration with shape batch x n
        'alpha': acausal angular rate with shape batch x n
    '''
    spatial_dim = ys['traj'].shape[-1]
    mses = []
    n_control = ys[control_type].shape[1]

    # compute l2 for i-th predicted distribution
    for i in range(n_control):
        y_p = y_pred[i]
        y = ys[control_type][:,i]

        y_p_w = y_p['lweights_'+control_type]
        y_p_m = y_p['mus_'+control_type]
        num_component = y_p_w.shape[1]
        mse = 0
        for k in range(num_component):
            diff = (y_p_m[:,k] - y) # we need to upscale the difference
            mse += y_p_w[k].exp() * torch.mean(torch.norm(diff,p=2))
        mses.append(mse)

    return torch.stack(mses).mean()


def loss_std_control(y_pred, ys, std_mean=1.0, control_type=''):
    '''
    Compute regularized std loss
    :param y_pred: a set of GMM parameters including
        'lweights': component log weights with shape batch x num_component
        'mus':      component means with shape batch x num_component
        'lsigs':    component log stds with shape batch x num_component
    :param std_mean: desired value of std_mean
    '''
    std_losses = []
    n_control = ys[control_type].shape[1]

    # compute l2 for i-th predicted distribution
    for i in range(n_control):
        y_p = y_pred[i]
        y_p_s = y_p['lsigs_'+control_type]
        std   = torch.ones(y_p_s.shape)*std_mean
        std   = std.to(y_p_s.device)

        # compute mse between predicted std and expected std for each batch
        diff_se  = (y_p_s.exp() - std)**2
        diff_mse = diff_se.mean()

        std_losses.append(diff_mse)
    return torch.stack(std_losses).mean()

def loss_weight_control(y_pred, ys, control_type=''):
    '''
    Compute regularized weight loss
    :param y_pred: a set of GMM parameters including
        'lweights': component log weights with shape batch x num_component
        'mus':      component means with shape batch x num_component x spatial_dim
        'lsigs':    component log stds with shape batch x num_component x spatial_dim
    '''

    # encourage weight to be nonzero
    loss = y_pred[0]['lweights_'+control_type]
    loss = (loss.exp()**2).mean()
    return loss

def regularized_loss_nll_control(y_pred, ys,
    std_regularization_factor=0.0, std_mean=1.0,
    weight_regularization_factor=0.0,
    ade_regularization_factor=0.0,
    position_downscaling_factor=100.0):
    '''
    Compute regularized negative log-likelihoods (NLL) loss for control inputs
    :param y_pred: a set of predicted GMM parameters
    :param ys: a set of groundtruth control targets
    :param lstd_regularization_factor: importance weight of lstd regularization
    :param std_mean: desired value of std_mean
    :param lweight_regularization_factor: importance weight of lweight regularization
    '''
    result = loss_nll_control(y_pred, ys, control_type='acc')
    result+= loss_nll_control(y_pred, ys, control_type='alpha')

    if std_regularization_factor > 0:
        # add an additional loss for lstd to prevent it from being too large or too small
        result += std_regularization_factor*loss_std_control(y_pred, ys, std_mean=0.1, control_type='acc')
        result += std_regularization_factor*loss_std_control(y_pred, ys, std_mean=0.5, control_type='alpha')

    if weight_regularization_factor > 0:
        # add an additional loss for lweight
        result += weight_regularization_factor*loss_weight_control(y_pred, None, control_type='acc')
        result += weight_regularization_factor*loss_weight_control(y_pred, None, control_type='alpha')

    if ade_regularization_factor > 0:
        result += ade_regularization_factor*loss_l2_control(y_pred, ys, control_type='acc')
        result += ade_regularization_factor*loss_l2_control(y_pred, ys, control_type='alpha')

    # import IPython; IPython.embed(header='nll')
    return result
