"""
Predictor model
"""
import torch
from torch import nn

class RNNEncoder(nn.Module):
    def __init__(self, device,
        spatial_dim=2, embedding_dim=16, h_dim=32, num_layers=1, dropout=0.0):
        """
        :param device: which device to train model (e.g., cuda, cpu)
        :spatical_dim: dimension of predicted positions
        :embedding_dim: dimension of position embeddings
        :h_dim: dimension of hidden state in RNN
        :num_layers: number of layers in RNN
        :dropout: percentage of non-dropout
        """
        super(RNNEncoder, self).__init__()

        self.device = device
        self.num_layers = num_layers
        self.h_dim = h_dim

        # embedding trajectory
        self.spatial_embedding = nn.Linear(spatial_dim, embedding_dim)

        # embedding encoder input
        self.encoder_input_embedding = nn.Linear(embedding_dim, embedding_dim)

        # RNN cell for encoder
        self.encoder = nn.GRU(
            embedding_dim, h_dim, num_layers
        )

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Encode past trajectory into latent vector
        """
        batch, seq_len = x.shape[0], x.shape[1]
        initial_h = torch.zeros(self.num_layers, batch, self.h_dim).to(self.device)

        x_ = x.float()
        ebd_x = []
        # embed positions
        for i in range(seq_len):
            ebd_x_i = self.spatial_embedding(x_[:,i])
            ebd_x_i = self.relu(ebd_x_i)
            ebd_x_i = self.dropout(ebd_x_i)
            ebd_x.append(ebd_x_i)

        ebd_x = torch.stack(ebd_x)
        ebd_x = self.encoder_input_embedding(ebd_x)
        ebd_x = self.dropout(self.relu(ebd_x))

        # run encoder RNN on past trajectory and return hidden from the last RNN cell
        output, state = self.encoder(ebd_x, initial_h)
        return state

class RNNDecoder(nn.Module):
    def __init__(self, device, num_components=3, pred_len=30,
        spatial_dim=2, input_dim=8, embedding_dim=16, h_dim=32, num_layers=1, dropout=0.0):
        """
        :param device: which device to train model (e.g., cuda, cpu)
        :param num_components: number of components in GMM
        :pred_len: predicting horizon
        :spatical_dim: dimension of predicted positions
        :input_dim: dimension of input tensor to RNN
        :embedding_dim: dimension of position embeddings
        :h_dim: dimension of hidden state in RNN
        :num_layers: number of layers in RNN
        :dropout: percentage of non-dropout
        """
        super(RNNDecoder, self).__init__()

        self.device = device
        self.num_components = num_components
        self.pred_len = pred_len
        self.spatial_dim = spatial_dim
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.h_dim = h_dim

        # TODO: embed input
        # # embedding trajectory
        # self.spatial_embedding = nn.Linear(spatial_dim, embedding_dim)

        # RNN cell for decoder
        self.decoder = nn.GRU(
            input_dim, h_dim, num_layers
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

        # define gmm parameters (weight, mu, lsig)
        # TODO: use mlp w/ dropouts and batchnorms
        self.latent2lweight = nn.Linear(h_dim, num_components)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        # assume independence between x and y
        self.latent2lsig    = nn.Linear(h_dim, num_components*spatial_dim)

        self.latent2mu      = nn.Linear(h_dim, num_components*spatial_dim)

        self.latent2input1 = nn.Linear(h_dim, input_dim)
        self.latent2input2 = nn.Linear(input_dim, input_dim)

        # linear layers for control predictions
        self.latent2lweight_acc = nn.Linear(h_dim, num_components)
        self.latent2lsig_acc = nn.Linear(h_dim, num_components)
        self.latent2mu_acc = nn.Linear(h_dim, num_components)

        self.latent2lweight_alpha = nn.Linear(h_dim, num_components)
        self.latent2lsig_alpha = nn.Linear(h_dim, num_components)
        self.latent2mu_alpha = nn.Linear(h_dim, num_components)


    def forward(self, x):
        """
        Encode past trajectory into latent vector
        """
        batch = x.shape[1]
        predictions = []

        # initialize decoder input with 0s
        decoder_input = torch.zeros(1, batch, self.input_dim).to(self.device)

        # assume weight is fixed for all t's
        lweights = None
        lweights_acc = None
        lweights_alpha = None
        hidden = x

        # run decoder iteratively over horizon pred_len
        for i in range(self.pred_len):
            output, hidden = self.decoder(decoder_input, hidden)
            output = output.view(batch, -1)
            # get GMM parameters (log-weight, mean, log-sigma) from hidden state
            mus     = self.latent2mu(output)
            lsigs   = self.latent2lsig(output)

            if lweights is None:
                lweights = self.latent2lweight(output)
                lweights = self.logsoftmax(lweights) # make sure weights are summed up to 1 after exp

            # reshape GMM parameters
            mus = mus.view(-1, self.num_components, self.spatial_dim)
            lsigs = lsigs.view(-1, self.num_components, self.spatial_dim)

            # obtain control GMM parameters
            mus_acc = self.latent2mu_acc(output)
            lsigs_acc = self.latent2lsig_acc(output)
            if lweights_acc is None:
                lweights_acc = self.logsoftmax(self.latent2lweight_acc(output))

            mus_alpha = self.latent2mu_alpha(output)
            lsigs_alpha = self.latent2lsig_alpha(output)
            if lweights_alpha is None:
                lweights_alpha = self.logsoftmax(self.latent2lweight_alpha(output))

            # reformat position to feed into next iteration
            decoder_input = self.latent2input1(output)
            decoder_input = self.relu(decoder_input.unsqueeze(0))
            decoder_input = self.dropout(decoder_input)
            decoder_input = self.latent2input2(decoder_input)
            decoder_input = self.relu(decoder_input)
            decoder_input = self.dropout(decoder_input)
            predictions.append({'lweights': lweights, 'mus': mus, 'lsigs': lsigs,
                'lweights_acc': lweights_acc, 'mus_acc': mus_acc, 'lsigs_acc': lsigs_acc,
                'lweights_alpha': lweights_alpha, 'mus_alpha': mus_alpha, 'lsigs_alpha': lsigs_alpha})

        return predictions

class RNNEncoderDecoder(nn.Module):
    def __init__(self, device, dropout=0.0):
        super(RNNEncoderDecoder, self).__init__()
        self.mlp_dropout = dropout
        self.encoder = RNNEncoder(device, dropout=self.mlp_dropout)
        self.decoder = RNNDecoder(device, dropout=self.mlp_dropout)

    def forward(self, past_traj):
        """
        Encode past trajectory into a latent state, and then decode into future trajectory
        """
        latent = self.encoder(past_traj)
        # import IPython; IPython.embed(header='after encoder')
        future_pred = self.decoder(latent)

        return future_pred
