import math
import random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import matplotlib.pyplot as plt
from collections import deque

# Use CUDA
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


# Define the Experience Replay Buffer
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)

# Load the "highway-v0" Environment
# env = gym.make("CartPole-v0")
from client import signal_handler
import sys, traceback
try:
    env = gym.make("highway-v1")
except Exception as e:
    traceback.print_exc(file=sys.stdout)
    signal_handler()

# Define the Epsilon Greedy Exploration
epsilon_start = 1.0
epsilon_final = 0.1#0.01
epsilon_decay = 1000
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

# Define the Double Deep Q Network
class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n)
        )

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            # print(q_value.max(1))
            action = q_value.max(1)[1].data[0]
            # action = torch.numpy(q_value.max(1)[1].data[0])
            # print('a:',action)
        else:
            action = random.randrange(env.action_space.n)
        return action

current_model = DQN(env.observation_space .shape[0], env.action_space.n)
target_model = DQN(env.observation_space.shape[0], env.action_space.n)

if USE_CUDA:
    current_model = current_model.cuda()
    target_model = target_model.cuda()
optimizer = optim.Adam(current_model.parameters())
replay_buffer = ReplayBuffer(2000)

# Synchronize current policy net and target net
def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())
update_target(current_model, target_model)


# Computing Temporal Difference Loss
def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    q_values = current_model(state)
    next_q_values = current_model(next_state)
    next_q_state_values = target_model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


# Plot the Results
def plot(frame_idx, rewards, losses):
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()
    plt.pause(1.0)
    plt.close()


# Start the Training
num_frames = 20000
batch_size = 128
gamma = 0.99
losses = []
all_rewards = []
episode_reward = 0
state = env.reset()
print('Start training...')
for frame_idx in range(1, num_frames + 1):
    epsilon = epsilon_by_frame(frame_idx)
    action = current_model.act(state, epsilon)
#
    if isinstance(action, torch.Tensor):
        action_1 = action.cpu().numpy()
        action_1 = int(action_1)
    else:
        action_1 = action
    # print('......')
    # print('action:', action_1)
    # print("before\n")
    next_state, reward, done, _ = env.step(action_1)
    # print('reward:', reward)
    replay_buffer.push(state, action, reward, next_state, done)
    # print("after\n")
    state = next_state
    episode_reward += reward
#
    if done:
        print('episode_reward:', episode_reward)
        print('----------')
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0
#
    if len(replay_buffer) > batch_size:
        print("1\n")
        loss = compute_td_loss(batch_size)
        losses.append(loss.data[0])
#
    if frame_idx % 500 == 0:
        print("2\n")
        plot(frame_idx, all_rewards, losses)
        print('result plot')
#
    if frame_idx % 500 == 0:
        print("3\n")
        update_target(current_model, target_model)
        torch.save(current_model, './ddqn_model.pkl')
        print('model saved',frame_idx)
#
    env.render()
print('training finished')
#
#
    # plot(frame_idx, all_rewards, losses)
"""
model = torch.load('./ddqn_model.pkl')
epsilon = 0
episode_reward = 0
all_rewards = []
state = env.reset()
while 1:
    action = model.act(state, epsilon)

    if isinstance(action, torch.Tensor):
        action_1 = action.cpu().numpy()
        action_1 = int(action_1)
    else:
        action_1 = action
    next_state, reward, done, _ = env.step(action_1)

    state = next_state
    episode_reward += reward
    env.render()

    if done:
        state = env.reset()
        print(episode_reward)
        all_rewards.append(episode_reward)
        episode_reward = 0

    if len(all_rewards) > 0 and len(all_rewards) % 10 == 0:
        print('plot')
        plt.plot(all_rewards)
        plt.show()
"""