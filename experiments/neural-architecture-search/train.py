import numpy as np
from keras.datasets import cifar10, cifar100, mnist
from keras.utils import to_categorical
import math
from controller import DQN, StateSpace, ReplayMemory, Transition
import torch.optim as optim
import torch
import random
from manager import NetworkManager
from model import model_fn
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

NUM_LAYERS = 4  # number of layers of the state space
STATE_DIMENSIONALITY = 2  # CONST; basically filters and kernels
NUM_ACTIONS = 6  # number of available filter and kernel sizes

MAX_TRIALS = 80  # maximum number of models generated
MAX_EPOCHS = 8  # maximum number of epochs to train
CHILD_BATCHSIZE = 128  # batchsize of the child models
EXPLORATION = 0.9  # high exploration for the first 1000 steps
REGULARIZATION = 1e-3  # regularization strength
CONTROLLER_CELLS = 32  # number of cells in RNN controller
EMBEDDING_DIM = 20  # dimension of the embeddings for each state
ACCURACY_BETA = 0.8  # beta value for the moving average of the accuracy
CLIP_REWARDS = 0.0  # clip rewards in the [-0.05, 0.05] range
RESTORE_CONTROLLER = True  # restore controller to continue training
GAMMA = 0.999

TARGET_UPDATE = 10
BATCH_SIZE = 1

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# construct a state space
state_space = StateSpace()

# add states
state_space.add_state(name='kernel', values=[1, 3, 6, 9, 12, 24])
state_space.add_state(name='filters', values=[2, 4, 8, 16, 32, 64])

# print the state space being searched
state_space.print_state_space()


def construct_network_manager(type, controller):
    # prepare the training data for the NetworkManager
    dataset = cifar10
    if type == 'cifar100':
        dataset = cifar100
    if type == 'mnist':
        dataset = mnist
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    if type == 'cifar100':
        y_train = to_categorical(y_train, 100)
        y_test = to_categorical(y_test, 100)
    else:
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
    packed = [x_train, y_train, x_test, y_test]  # pack the dataset for the NetworkManager
    manager = NetworkManager(dataset=packed, type=type, controller=controller, epochs=MAX_EPOCHS,
                             child_batchsize=CHILD_BATCHSIZE, clip_rewards=CLIP_REWARDS,
                             acc_beta=ACCURACY_BETA)
    return manager


# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()

# Get number of actions from gym action space
n_actions = math.factorial(NUM_LAYERS) * (state_space.total_combinations)

policy_net = DQN(NUM_LAYERS * STATE_DIMENSIONALITY, NUM_ACTIONS, n_actions).to(device)
target_net = DQN(NUM_LAYERS * STATE_DIMENSIONALITY, NUM_ACTIONS, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(100)

policy_net_gaussian = DQN(NUM_LAYERS * STATE_DIMENSIONALITY, NUM_ACTIONS, n_actions).to(device)
target_net_gaussian = DQN(NUM_LAYERS * STATE_DIMENSIONALITY, NUM_ACTIONS, n_actions).to(device)
target_net_gaussian.load_state_dict(policy_net_gaussian.state_dict())
target_net_gaussian.eval()

optimizer_gaussian = optim.RMSprop(policy_net_gaussian.parameters())
memory_gaussian = ReplayMemory(100)

steps_done = 0

logs = pd.DataFrame(
    {'action': [], 'dataset': [], 'accuracy': [], 'reward': [], 'total_reward': [], 'loss': [], 'iteration': [],
     'mu': [], 'sigma': [], 'ucb_reward': [], 'controller': [], 'random': []})
logs['action'] = logs.action.astype(str)
logs['dataset'] = logs.dataset.astype(str)
logs['controller'] = logs.controller.astype(str)
state = state_space.get_random_state_space(NUM_LAYERS)
print("Initial Random State : ", state_space.parse_state_space_list(state))
print()


# create the Network Manager

def get_action(state, controller='classic'):
    '''
    Gets a one hot encoded action list, either from random sampling or from
    the Controller RNN

    Args:
        state: a list of one hot encoded states, whose first value is used as initial
            state for the controller RNN

    Returns:
        A one hot encoded action list
    '''
    global steps_done
    global EXPLORATION_RATE
    sample = random.random()

    steps_done += 1
    if sample <= EXPLORATION_RATE:
        logs.at[logs.index[-1], 'random'] = 1
        # print("Generating random action to explore")
        actions = []

        for i in range(STATE_DIMENSIONALITY * NUM_LAYERS):
            state_ = state_space[i]
            size = state_['size']

            sample = np.random.choice(size, size=1)
            sample = state_['index_map_'][sample[0]]
            action = state_space.embedding_encode(i, sample)
            actions.append(action)
        return actions

    else:
        logs.at[logs.index[-1], 'random'] = 0
        pn = policy_net
        if controller == 'gaussian':
            pn = policy_net_gaussian
        # print("Prediction action from Controller")
        input_height = NUM_LAYERS * STATE_DIMENSIONALITY
        reshaped_state = torch.FloatTensor(state).unsqueeze(0).reshape(1, 1, input_height, NUM_ACTIONS)
        res = pn(reshaped_state.to(device))
        reshaped_res = res.view(input_height, int(n_actions / (input_height * NUM_ACTIONS)), NUM_ACTIONS).max(1)[
            0].view(input_height, 1, NUM_ACTIONS).detach().cpu().numpy()
        actions_that_will_be_performed = state_space.parse_state_space_list(reshaped_res)
        # print(actions_that_will_be_performed)
        actions = []
        for i in range(STATE_DIMENSIONALITY * NUM_LAYERS):
            sample = actions_that_will_be_performed[i]
            action = state_space.embedding_encode(i, sample)
            actions.append(action)
        return actions


distribution_mu = nn.Linear(n_actions, 1).to(device)
distribution_presigma = nn.Linear(n_actions, 1).to(device)
distribution_sigma = nn.Softplus().to(device)


def optimize_model(controller):
    if len(memory) < BATCH_SIZE:
        return
    pn = policy_net
    tn = target_net
    m = memory
    o = optimizer
    if controller == 'gaussian':
        pn = policy_net_gaussian
        m = memory_gaussian
        tn = target_net_gaussian
        o = optimizer_gaussian
    transitions = m.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            state_batch)), device=device, dtype=torch.uint8)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    input_height = NUM_LAYERS * STATE_DIMENSIONALITY
    state_batch_reshaped = torch.FloatTensor(state_batch).unsqueeze(0).reshape(1, 1, input_height, NUM_ACTIONS).to(
        device)
    state_action_values = pn(state_batch_reshaped)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = tn(state_batch_reshaped)
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.to(device)

    loss = 0

    if controller == 'gaussian':
        # Compute gaussian loss
        pre_sigma = distribution_presigma(next_state_values.to(device))
        mu = distribution_mu(next_state_values.to(device))
        sigma = distribution_sigma(next_state_values.to(device))

        zero_index = (next_state_values != 0)
        distribution = torch.distributions.normal.Normal(mu, sigma[zero_index])
        likelihood = distribution.log_prob(next_state_values[zero_index])
        loss = -torch.mean(likelihood)
        logs.at[logs.index[-1], 'loss'] = loss
        logs.at[logs.index[-1], 'mu'] = mu
        logs.at[logs.index[-1], 'sigma'] = torch.mean(sigma)
    else:
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        logs.at[logs.index[-1], 'loss'] = loss
    # Optimize the model
    o.zero_grad()
    loss.backward()

    for param in pn.parameters():
        if param.grad is not None:
            param.grad.data.clamp_(-1, 1)
    o.step()


for c in ['gaussian', 'classic']:
    PERFORMED_ACTIONS_LIST = []
    m = memory_gaussian
    tn = target_net_gaussian
    pn = policy_net_gaussian
    total_reward = 0
    steps_done = 0
    previous_acc = 0.0
    total_reward = 0.0
    global EXPLORATION_RATE
    EXPLORATION_RATE = 0.9
    if c == 'classic':
        m = memory
        tn = target_net
        pn = policy_net
    for d in ['cifar10', 'cifar100']:
        manager = construct_network_manager(d, c)
        for i_episode in range(MAX_TRIALS):
            logs = logs.append(pd.Series(), ignore_index=True)
            action = get_action(state, c)
            # build a model, train and get reward and accuracy from the network manager
            current_action = state_space.parse_state_space_list(action)
            current_action_str = "-".join(map(str, current_action))
            PERFORMED_ACTIONS_LIST.append(current_action_str)
            times_action_was_played = PERFORMED_ACTIONS_LIST.count(current_action_str)
            reward, previous_acc = manager.get_rewards(model_fn, current_action)
            print("Reward received from network manager: ", reward, "Accuracy of CNN trained: ", previous_acc)
            ucb_reward = 0
            if (i_episode > 0):
                ucb_reward = math.sqrt((2.0 * math.log(i_episode)) / times_action_was_played)
            # Because of append to PERFORMED_ACTIONS_LIST above, division by 0 is impossible
            updated_reward = reward + ucb_reward
            # print("Number of time action was played: ", times_action_was_played, "Updared_reward: ", updated_reward)
            total_reward += reward
            print("Total reward: ", total_reward)

            # Saving results to a dataframe
            logs.at[logs.index[-1], 'action'] = current_action_str
            logs.at[logs.index[-1], 'reward'] = reward
            logs.at[logs.index[-1], 'ucb_reward'] = ucb_reward
            logs.at[logs.index[-1], 'total_reward'] = total_reward
            logs.at[logs.index[-1], 'accuracy'] = previous_acc
            logs.at[logs.index[-1], 'dataset'] = d
            logs.at[logs.index[-1], 'controller'] = c
            logs.at[logs.index[-1], 'iteration'] = i_episode
            # Store the transition in memory
            m.push(torch.FloatTensor(state), torch.FloatTensor(action), torch.FloatTensor([reward]))
            optimize_model(c)

            if i_episode % 10 == 0:
                if EXPLORATION_RATE > 0.1:
                    EXPLORATION_RATE -= 0.1
                logs.to_csv("logs.csv")
            # Update the target network, copying all weights and biases in DQN
            if i_episode % TARGET_UPDATE == 0:
                tn.load_state_dict(pn.state_dict())
            # Move to the next state
            # actions and states are equivalent, save the state and reward
            state = action

logs.to_csv("logs.csv")
print('Complete')
