# evaluator to log and visualize the statistics during training

import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from prediction.visualize import draw_lane_centerlines, draw_traj, draw_prediction_gmm
from prediction.utils import save_model

from plan_verification.random_objects import MultivariateNormal, GMM, GmmTrajectory

class ValidationEvent(object):
    def __init__(self,
                 session_id,
                 val_loader,
                 logger,
                 evaluator,
                 model,
                 writer,
                 model_args=None,
                 training_args=None):
        self.session_id = session_id
        self.val_loader = val_loader
        self.logger = logger
        self.evaluator = evaluator
        self.model = model
        self.writer = writer
        self.model_args = model_args
        self.training_args = training_args
        self.position_downscaling_factor = training_args['position_downscaling_factor']

        self.best_loss = 1e20

    def log_validation_results(self, engine):
        '''
        Run evluation and log results to tensorboard
        :param engine: ignite engine
        :param prob_evaluator: ignite evaluator
        :return:
        '''
        self.evaluator.run(self.val_loader)
        metrics = self.evaluator.state.metrics

        # print metrics to console and log to tensorboard writer
        metrics_to_print = ['ade', 'fde', 'nll', 'std', 'wgt', 'nll_acc', 'nll_alpha']
        for metric, value in metrics.items():
            if metric in metrics_to_print:
                self.logger.info("Validation Results - Epoch: {}  Avg {}: {:.2f}"
                  .format(engine.state.epoch, metric, value))
            self.writer.add_scalar("valdation/{}".format(metric), value, engine.state.epoch)

        # obtain one prediction sample and visualize
        y_pred, ys = self.evaluator.state.output

        batch_sz = ys['traj'].shape[0]
        # visualize each sample of the batch
        # import IPython; IPython.embed()

        for batch_idx in range(batch_sz):
            y_acausal_single = np.array(ys['traj'][batch_idx].tolist())*self.position_downscaling_factor
            y_past_single = np.array(ys['past_traj'][batch_idx].tolist())*self.position_downscaling_factor

            # visualize predictions in tb
            fig_trajectory = plt.figure("traj_pred")
            ax = plt.gca()

            # plot past and acausal trajectory
            # TODO: plot nearby lanes
            draw_traj(y_past_single, marker="s", color="#d33e4c")
            draw_traj(y_acausal_single, marker="o", color="#d33e4c")

            # get predictions from single batch element
            prediction = []
            for i, y_p in enumerate(y_pred):
                y_pred_mu_single = [y_p['mus'][batch_idx]]
                y_pred_lstd_single = [y_p['lsigs'][batch_idx]]
                y_pred_lweight_single = [y_p['lweights'][batch_idx]]

                y_pred_single = {'mus': y_pred_mu_single, 'lsigs': y_pred_lstd_single, 'lweights': y_pred_lweight_single}
                prediction.append(y_pred_single)

            gmm_traj = GmmTrajectory.from_prediction(prediction, self.position_downscaling_factor)

            # visualize predictions
            draw_prediction_gmm(ax, gmm_traj)
            plt.axis('equal')

            self.writer.add_figure("Traj_pred/sample_{}".format(batch_idx), fig_trajectory, engine.state.epoch)

        # Save model with the best loss
        current_loss = metrics['nll']

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.logger.info("Saving model {}".format(self.session_id))
            save_model(
                self.session_id,
                self.model,
                self.model_args,
                self.training_args,
                self.best_loss,
                epoch=engine.state.epoch
                )
