from PySide2.QtWidgets import QApplication
from GameEngine import GameEngine
from itertools import count
import sys
import numpy as np
from collections import deque
import pandas as pd
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
import torch.functional as F
from agent import Actor_Critic


def stack_frames(stacked_frames, state, new_episode):
    frame = state
    if new_episode:
        stacked_frames = deque([np.zeros((84, 84)) for _ in range(4)], maxlen=4)
        maxframe = np.maximum(frame, frame)
        stacked_frames.append(maxframe)
        stacked_frames.append(maxframe)
        stacked_frames.append(maxframe)
        stacked_frames.append(maxframe)
        stacked_state = np.stack(stacked_frames)
    else:
        maxframe = np.maximum(stacked_frames[-1], frame)
        stacked_frames.append(maxframe)
        stacked_state = np.stack(stacked_frames)
    return np.squeeze(stacked_state), stacked_frames


if __name__ == "__main__":
    app = QApplication(sys.argv)
    env = GameEngine()
    env.start()

    #Hyperparameters
    num_episodes = 200
    learning_rate = 0.01
    GAMMA = 0.99
    writer = SummaryWriter()
    actor_critic = Actor_Critic()
    actor_critic.train()
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)
    episode_rewards = []
    episode_length = []
    episode_entropy = []
    episode_actor_loss = []
    episode_critic_loss = []
    episode_total_loss = []
    for i_episode in range(num_episodes):
        state = env.reset()
        stacked_frames = deque([np.zeros((84, 84)) for _ in range(4)], maxlen=4)
        stacked_state, stacked_frames = stack_frames(stacked_frames, state, new_episode=True)
        state = stacked_state
        log_probs = []
        values = []
        advantages = []
        rewards = []
        entropys = []
        critics = []
        actors = []
        overall_entropy = 0
        for t in count():
            state = np.reshape(state, (1, 4, 84, 84))
            policy, value = actor_critic.forward(torch.from_numpy(state).float().to(device))
            probs = Categorical(F.softmax(policy, dim=1))
            # value = critic.forward(torch.from_numpy(state).float().to(device))
            action = probs.sample()
            log_prob = probs.log_prob(action)
            entropy = probs.entropy()
            log_probs.append(log_prob)
            values.append(value)
            entropys.append(entropy)
            # print("entropy {} and probs {}".format(entropy, probs))
            overall_entropy += entropy.item()
            # step in environment
            next_state, reward, done, _ = env.step(action)
            stacked_next_state, stacked_frames = stack_frames(stacked_frames, next_state, new_episode=False)
            next_state = stacked_next_state
            next_state = next_state.reshape((1, 4, 84, 84))
            # define V value for next state
            _, next_value = actor_critic.forward(torch.from_numpy(next_state).float().to(device))
            # compute advantage A = r + gamma * V(s) - V(s')
            advantage = reward + GAMMA * next_value - value
            advantages.append(advantage)
            # switch to next state
            state = next_state
            rewards.append(reward)
            if done:
                episode_rewards.append(np.sum(rewards))
                episode_length.append(t)
                episode_critic_loss.append(np.sum(critics))
                print("episode: {}, reward: {}, total length: {}, entropy {:.2f}".format(i_episode, np.sum(rewards),
                                                                                         t, overall_entropy))
                break

        advantage = torch.stack(advantages)
        # update actor critic
        entropys = torch.stack(entropys)
        log_probs = torch.stack(log_probs)

        critic_loss = 0.5 * advantage.pow(2).sum()
        actor_loss = (-log_probs * advantage.detach()).sum()
        ac_loss = actor_loss + critic_loss - 0.01 * entropys.sum()
        episode_actor_loss.append(critic_loss.detach().numpy())
        episode_total_loss.append(ac_loss.detach().numpy())
        episode_entropy.append(entropys.mean())
        # print("episode loss {:.2f} \n".format(ac_loss))

        ac_optimizer.zero_grad()
        ac_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), 1)
        ac_optimizer.step()

        app.exec_()
        sys.exit(app.exit())
