"""
Script to train an RNN-based encoder decoder predictor
Usage:
    Start tensorboard:
    tensorboard --logdir=logs/
    Run the example:
    python train.py --data_dir dataset/argoverse_train1_filtered_1000/ --log_dir=logs/
"""

from __future__ import print_function

from argparse import ArgumentParser
import functools
import logging
import os
import subprocess
import uuid

import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD, Adam

try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

from prediction.evaluation import ValidationEvent
from prediction.data import get_data_loaders
from prediction.loss import loss_ade, loss_fde
from prediction.loss import loss_nll, loss_std, loss_weight, regularized_loss_nll
from prediction.loss import regularized_loss_nll_control, loss_nll_control
from prediction.model import RNNEncoderDecoder
from prediction.utils import init_weights

# fix random seed
torch.manual_seed(0)

def create_summary_writer(model, data_loader, log_dir, session_id):
    """
    Create writer in Tensorboard
    """
    writer = SummaryWriter(log_dir=os.path.join(log_dir, session_id))
    return writer

def main(training_args):
    """
    Train the predictor using the parameters specified at the bottom
    """

    # creat a session id for every training session
    session_id = uuid.uuid4().hex

    # create a logger to log training progress
    logger = logging.getLogger('rss')
    logging.basicConfig(level=logging.INFO)
    logger.info('Start training session {}'.format(session_id))

    # obtain training and validation data loaders
    train_loader, val_loader = get_data_loaders(training_args['data_dir'],
        training_args['obs_len'],
        training_args['train_batch_size'],
        training_args['val_batch_size'],
        training_args['val_ratio'],
        training_args['subsampling_factor'],
        training_args['position_downscaling_factor'])
    device = 'cpu'

    # use cuda when possible
    if torch.cuda.is_available():
        device = 'cuda'

    # create prediction model
    model = RNNEncoderDecoder(device, training_args['mlp_dropout'])
    model.apply(init_weights)

    # create a writer log training and validation progress in tensorboard
    writer = create_summary_writer(model, train_loader, training_args['log_dir'], session_id)

    # save arguments and git info
    writer.add_text(tag='args', text_string=str(training_args), global_step=0)
    git_info = subprocess.check_output(["git", "log", "-v", "-1"])
    writer.add_text(tag='git_info', text_string=str(git_info), global_step=0)

    # create optiimzer
    # TODO: test SGD vs. Adam
    # optimizer = SGD(model.parameters(), lr=training_args['lr'], momentum=training_args['momentum'])
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=training_args['lr'],
        betas=(0.5, 0.999))

    # create supervised trainer and evaluator in ignite
    regularized_loss_nll_full = functools.partial(regularized_loss_nll,
        std_regularization_factor=training_args['std_regularization_factor'],
        std_mean=training_args['std_regularization_mean'],
        weight_regularization_factor=training_args['weight_regularization_factor'],
        ade_regularization_factor=training_args['ade_regularization_factor'],
        position_downscaling_factor=training_args['position_downscaling_factor'],
        control_regularization_factor=training_args['control_regularization_factor'])
    loss_ade_full = functools.partial(loss_ade,
        position_downscaling_factor=training_args['position_downscaling_factor'])
    loss_fde_full = functools.partial(loss_fde,
        position_downscaling_factor=training_args['position_downscaling_factor'])
    loss_std_full = functools.partial(loss_std,
        std_mean=training_args['std_regularization_mean'],
        position_downscaling_factor=training_args['position_downscaling_factor'])
    loss_nll_acc = functools.partial(loss_nll_control,
        control_type='acc')
    loss_nll_alpha = functools.partial(loss_nll_control,
        control_type='alpha')
    trainer = create_supervised_trainer(model, optimizer, regularized_loss_nll_full, device=device)
    loss_batch = lambda x: x['traj'].shape[0]
    evaluator = create_supervised_evaluator(model,
                                            metrics={'nll': Loss(loss_nll, batch_size=loss_batch),
                                                     'ade': Loss(loss_ade_full, batch_size=loss_batch),
                                                     'fde': Loss(loss_fde_full, batch_size=loss_batch),
                                                     'std': Loss(loss_std_full, batch_size=loss_batch),
                                                     'wgt': Loss(loss_weight, batch_size=loss_batch),
                                                     'nll_acc': Loss(loss_nll_acc, batch_size=loss_batch),
                                                     'nll_alpha': Loss(loss_nll_alpha, batch_size=loss_batch),},
                                            device=device)

    # print training loss after every iteration
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iteration = engine.state.iteration%len(train_loader)
        iteration = len(train_loader) if iteration == 0 else iteration
        logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
              "".format(engine.state.epoch, iteration,
                        len(train_loader), engine.state.output))
        writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)

    # print metrics after every epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_ade = metrics['ade']
        avg_fde = metrics['fde']
        avg_nll = metrics['nll']
        avg_std = metrics['std']
        avg_wgt = metrics['wgt']
        avg_nll_acc = metrics['nll_acc']
        avg_nll_alpha = metrics['nll_alpha']
        logger.info("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(engine.state.epoch, avg_ade, avg_nll))
        writer.add_scalar("training/avg_nll", avg_nll, engine.state.epoch)
        writer.add_scalar("training/avg_ade", avg_ade, engine.state.epoch)
        writer.add_scalar("training/avg_fde", avg_fde, engine.state.epoch)
        writer.add_scalar("training/avg_std", avg_std, engine.state.epoch)
        writer.add_scalar("training/avg_wgt", avg_wgt, engine.state.epoch)
        writer.add_scalar("training/avg_nll_acc", avg_nll_acc, engine.state.epoch)
        writer.add_scalar("training/avg_nll_alpha", avg_nll_alpha, engine.state.epoch)

    # log validation metrics to tensorboard after every epoch
    validation_event = ValidationEvent(
        session_id,
        val_loader,
        logger,
        evaluator,
        model,
        writer,
        None, # TODO: add model args
        training_args,
        )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, validation_event.log_validation_results)

    # kick off trainer
    trainer.run(train_loader, max_epochs=training_args['epochs'])

    writer.close()

    return session_id


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-i', '--data_dir', type=str,
                        help='input directory')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--val_batch_size', type=int, default=64,
                        help='input batch size for validation (default: 64)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument("--log_dir", type=str, default="tensorboard_logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument('--obs_len', type=int, default=20,
                        help='length of observed trajetory')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='percentage of test set')
    parser.add_argument('--subsampling_factor', type=float, default=1.0,
                        help='percentage of subsampling')
    parser.add_argument('--position_downscaling_factor', type=float, default=100.0,
                        help='downscaling factor for trajectories')
    parser.add_argument('--std_regularization_factor', type=float, default=0.1,
                        help='regularization factor for standard deviation loss')
    parser.add_argument('--std_regularization_mean', type=float, default=1.0,
                        help='prior mean for standard deviation')
    parser.add_argument('--weight_regularization_factor', type=float, default=1.0,
                        help='regularization factor for weight loss')
    parser.add_argument('--ade_regularization_factor', type=float, default=0.1,
                        help='regularization factor for ade loss')
    parser.add_argument('--mlp_dropout', type=float, default=0.1,
                        help='dropout rate in mlp layers')
    parser.add_argument('--control_regularization_factor', type=float, default=0.0,
                        help='regularization factor for control losses')

    args = parser.parse_args()

    training_args = {'data_dir': args.data_dir, 'obs_len': args.obs_len,
        'train_batch_size': args.batch_size, 'val_batch_size': args.val_batch_size,
        'epochs': args.epochs, 'lr': args.lr, 'momentum': args.momentum,
        'log_interval': args.log_interval, 'log_dir': args.log_dir,
        'val_ratio': args.val_ratio, 'subsampling_factor': args.subsampling_factor,
        'position_downscaling_factor': args.position_downscaling_factor,
        'std_regularization_factor': args.std_regularization_factor,
        'std_regularization_mean': args.std_regularization_mean,
        'weight_regularization_factor': args.weight_regularization_factor,
        'ade_regularization_factor': args.ade_regularization_factor,
        'mlp_dropout': args.mlp_dropout,
        'control_regularization_factor': args.control_regularization_factor,}

    session_id = main(training_args)

    import IPython; IPython.embed(header='Done with {}'.format(session_id))
