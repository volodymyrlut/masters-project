import pprint
from collections import OrderedDict

import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class StateSpace:
    '''
    State Space manager

    Provides utilit functions for holding "states" / "actions" that the controller
    must use to train and predict.

    Also provides a more convenient way to define the search space
    '''
    def __init__(self):
        self.states = OrderedDict()
        self.state_count_ = 0
        self.total_combinations = 1

    def add_state(self, name, values):
        '''
        Adds a "state" to the state manager, along with some metadata for efficient
        packing and unpacking of information required by the RNN Controller.

        Stores metadata such as:
        -   Global ID
        -   Name
        -   Valid Values
        -   Number of valid values possible
        -   Map from value ID to state value
        -   Map from state value to value ID

        Args:
            name: name of the state / action
            values: valid values that this state can take

        Returns:
            Global ID of the state. Can be used to refer to this state later.
        '''
        index_map = {}
        for i, val in enumerate(values):
            index_map[i] = val

        value_map = {}
        for i, val in enumerate(values):
            value_map[val] = i

        metadata = {
            'id': self.state_count_,
            'name': name,
            'values': values,
            'size': len(values),
            'index_map_': index_map,
            'value_map_': value_map,
        }
        self.states[self.state_count_] = metadata
        self.state_count_ += 1

        self.total_combinations *= len(values)

        return self.state_count_ - 1

    def embedding_encode(self, id, value):
        '''
        Embedding index encode the specific state value

        Args:
            id: global id of the state
            value: state value

        Returns:
            embedding encoded representation of the state value
        '''
        state = self[id]
        size = state['size']
        value_map = state['value_map_']
        value_idx = value_map[value]

        one_hot = np.zeros((1, size), dtype=np.float32)
        one_hot[np.arange(1), value_idx] = value_idx + 1
        return one_hot

    def get_state_value(self, id, index):
        '''
        Retrieves the state value from the state value ID

        Args:
            id: global id of the state
            index: index of the state value (usually from argmax)

        Returns:
            The actual state value at given value index
        '''
        state = self[id]
        index_map = state['index_map_']

        if (type(index) == list or type(index) == np.ndarray) and len(index) == 1:
            index = index[0]

        value = index_map[index]
        return value

    def get_random_state_space(self, num_layers):
        '''
        Constructs a random initial state space for feeding as an initial value
        to the Controller RNN

        Args:
            num_layers: number of layers to duplicate the search space

        Returns:
            A list of one hot encoded states
        '''
        states = []

        for id in range(self.size * num_layers):
            state = self[id]
            size = state['size']

            sample = np.random.choice(size, size=1)
            sample = state['index_map_'][sample[0]]
            state = self.embedding_encode(id, sample)
            states.append(state)
        return states

    def parse_state_space_list(self, state_list):
        '''
        Parses a list of one hot encoded states to retrieve a list of state values

        Args:
            state_list: list of one hot encoded states

        Returns:
            list of state values
        '''
        state_values = []
        for id, state_one_hot in enumerate(state_list):
            state_val_idx = np.argmax(state_one_hot, axis=-1)[0]
            value = self.get_state_value(id, state_val_idx)
            state_values.append(value)

        return state_values

    def print_state_space(self):
        ''' Pretty print the state space '''
        print('*' * 40, 'STATE SPACE', '*' * 40)

        pp = pprint.PrettyPrinter(indent=2, width=100)
        for id, state in self.states.items():
            pp.pprint(state)
            print()

    def print_actions(self, actions):
        ''' Print the action space properly '''
        print('Actions :')

        for id, action in enumerate(actions):
            if id % self.size == 0:
                print("*" * 20, "Layer %d" % (((id + 1) // self.size) + 1), "*" * 20)

            state = self[id]
            name = state['name']
            vals = [(n, p) for n, p in zip(state['values'], *action)]
            print("%s : " % name, vals)
        print()

    def __getitem__(self, id):
        return self.states[id % self.size]

    @property
    def size(self):
        return self.state_count_

class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        print("LOOOOL")
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 48, kernel_size=2)
        self.bn2 = nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(48, 48, kernel_size=2)
        self.bn3 = nn.BatchNorm2d(48)
        input_size = (h + w + 1) * 48

        self.relu = nn.ReLU()
        self.distribution_mu = nn.Linear(outputs, 1)
        self.distribution_presigma = nn.Linear(outputs, 1)
        self.distribution_sigma = nn.Softplus()

        self.head = nn.Linear(input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        pre_sigma = self.distribution_presigma(x)
        mu = self.distribution_mu(x)
        sigma = self.distribution_sigma(pre_sigma)
        return sigma, mu, self.head(x.view(x.size(0), -1))

#
# state_space.parse_state_space_list(res.view(8, 18, 6).max(1)[0].view(8, 1, 6).detach().numpy())