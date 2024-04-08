import torch
import torch.nn as nn

from d3rlpy.models.encoders import EncoderFactory
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ConvEncoder(nn.Module):
    def __init__(self, observation_shape, feature_size):
        super(ConvEncoder, self).__init__()
        self.feature_size = feature_size
        self.fc1 = nn.Linear(96*64, 10240)
        self.fc2 = nn.Linear(10240, feature_size)

    def forward(self, x):

        conv1=nn.Conv1d(x.shape[0],64, kernel_size=80).to(device)
        h = torch.relu(conv1(x.reshape(1,x.shape[0],x.shape[1])))
        h = h.view(-1)
        h = torch.relu(self.fc1(h))
        h = torch.relu(self.fc2(h))
        return h

    # THIS IS IMPORTANT!
    def get_feature_size(self):
        return self.feature_size
#
#   encoder factory
class ConvEncoderFactory(EncoderFactory):
    TYPE = 'custom' # this is necessary

    def __init__(self, feature_size):
        self.feature_size = feature_size

    def create(self, observation_shape):
        return ConvEncoder(observation_shape, self.feature_size)

    def create_with_action(self,observation_shape, action_size):
        return ConvEncoderWithAction(observation_shape, action_size, self.feature_size)


    def get_params(self, deep=False):
        return {'feature_size': self.feature_size}


class ConvEncoderWithAction(nn.Module):
    def __init__(self, observation_shape, action_size, feature_size):
        super(ConvEncoderWithAction, self).__init__()

        self.feature_size = feature_size
        self.action_size = action_size
        self.fc1 = nn.Linear(64*96+1, 10240)

        self.fc2 = nn.Linear(10240, feature_size)

    def forward(self, x, action): # action is also given


        conv1 = nn.Conv1d(x.shape[0],64, kernel_size=80).to(device)

        h = torch.relu(conv1(x.reshape(1, x.shape[0], x.shape[1])))
        h = h.view(-1)

        if action.ndim == 3:
             action = action.reshape(action.shape[2])
        elif action.ndim == 2:
             action = action.reshape(action.shape[1])
        h = torch.cat([h, action])
        h = torch.relu(self.fc1(h))
        h = torch.relu(self.fc2(h))
        h = h.reshape(1, 1, h.shape[0])
        return h

    def get_feature_size(self):
        return self.feature_size

