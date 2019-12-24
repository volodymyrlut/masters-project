import numpy as np
from keras.datasets import cifar10
from keras.utils import to_categorical
import csv
import math
from controller import DQN, StateSpace, ReplayMemory, Transition
import torch.optim as optim
import torch
import random
import matplotlib
import matplotlib.pyplot as plt
from manager import NetworkManager
from model import model_fn
import torch.nn.functional as F

NUM_LAYERS = 4  # number of layers of the state space
STATE_DIMENSIONALITY = 2  # CONST; basically filters and kernels
MAX_TRIALS = 100  # maximum number of models generated
NUM_ACTIONS = 6  # number of available filter and kernel sizes

MAX_EPOCHS = 20  # maximum number of epochs to train
CHILD_BATCHSIZE = 128  # batchsize of the child models
EXPLORATION = 0.1  # high exploration for the first 1000 steps
REGULARIZATION = 1e-3  # regularization strength
CONTROLLER_CELLS = 32  # number of cells in RNN controller
EMBEDDING_DIM = 20  # dimension of the embeddings for each state
ACCURACY_BETA = 0.8  # beta value for the moving average of the accuracy
CLIP_REWARDS = 0.0  # clip rewards in the [-0.05, 0.05] range
RESTORE_CONTROLLER = True  # restore controller to continue training
GAMMA = 0.999
EPS_START = 0.1
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 2
BATCH_SIZE = 1

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# construct a state space
state_space = StateSpace()

# add states
state_space.add_state(name='kernel', values=[1, 3, 6, 9, 12, 24])
state_space.add_state(name='filters', values=[2, 4, 8, 16, 32, 64])

# print the state space being searched
state_space.print_state_space()

# prepare the training data for the NetworkManager
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

dataset = [x_train, y_train, x_test, y_test]  # pack the dataset for the NetworkManager

previous_acc = 0.0
total_reward = 0.0

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
memory = ReplayMemory(25)

steps_done = 0

# episode_durations = []


# def plot_durations():
#     plt.figure(2)
#     plt.clf()
#     durations_t = torch.tensor(episode_durations, dtype=torch.float)
#     plt.title('Training...')
#     plt.xlabel('Episode')
#     plt.ylabel('Duration')
#     plt.plot(durations_t.numpy())
#     # Take 100 episode averages and plot them too
#     if len(durations_t) >= 100:
#         means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
#         means = torch.cat((torch.zeros(99), means))
#         plt.plot(means.numpy())
#
#     plt.pause(0.001)  # pause a bit so that plots are updated
#     if is_ipython:
#         display.clear_output(wait=True)
#         display.display(plt.gcf())


state = state_space.get_random_state_space(NUM_LAYERS)
print("Initial Random State : ", state_space.parse_state_space_list(state))
print()

# create the Network Manager
manager = NetworkManager(dataset, epochs=MAX_EPOCHS, child_batchsize=CHILD_BATCHSIZE, clip_rewards=CLIP_REWARDS,
                         acc_beta=ACCURACY_BETA)


def get_action(state):
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
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                              math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample <= eps_threshold:
        print("Generating random action to explore")
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
        print("Prediction action from Controller")
        input_height = NUM_LAYERS * STATE_DIMENSIONALITY
        reshaped_state = torch.FloatTensor(state).unsqueeze(0).reshape(1, 1, input_height, NUM_ACTIONS)
        res = policy_net(reshaped_state.to(device))
        reshaped_res = res.view(input_height, int(n_actions / (input_height * NUM_ACTIONS)), NUM_ACTIONS).max(1)[
            0].view(input_height, 1, NUM_ACTIONS).detach().cpu().numpy()
        actions_that_will_be_performed = state_space.parse_state_space_list(reshaped_res)
        print(actions_that_will_be_performed)
        actions = []
        for i in range(STATE_DIMENSIONALITY * NUM_LAYERS):
            sample = actions_that_will_be_performed[i]
            action = state_space.embedding_encode(i, sample)
            actions.append(action)
        return actions


PERFORMED_ACTIONS_LIST = []


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
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
    state_action_values = policy_net(state_batch_reshaped)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = target_net(state_batch_reshaped)
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.to(device)

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


for i_episode in range(MAX_TRIALS):
    # Initialize the environment and state
    action = get_action(state)
    state_space.print_actions(action)
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
    print("Number of time action was played: ", times_action_was_played, "Updared_reward: ", updated_reward)
    total_reward += updated_reward
    print("Total reward: ", total_reward)

    # Store the transition in memory
    memory.push(torch.FloatTensor(state), torch.FloatTensor(action), torch.FloatTensor([updated_reward]))
    optimize_model()

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    # Move to the next state
    # actions and states are equivalent, save the state and reward
    state = action

    # if done:
    #     episode_durations.append(t + 1)
    #     plot_durations()
    #     break

print('Complete')
plt.ioff()
plt.show()