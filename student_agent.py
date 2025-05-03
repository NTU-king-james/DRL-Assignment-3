from training.train_4 import Mario
from training.wrappers import FrameDownsampleWrapper
import numpy as np
import cv2
from collections import deque
import torch

# Do not modify the input of the 'act' function and the '__init__' function. 

model_path = "./training/checkpoints/mario_net_episode_7000.chkpt"

class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        # create wrapped Mario environment matching training pipeline
        self.state_dim = (4, 84, 84)
        self.action_dim = 12
        self.frame_buffer = deque(maxlen=4)
        self.frame_count = 0
        self.current_action = 0
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        self.model = Mario(state_dim=self.state_dim, action_dim=self.action_dim, save_dir="results")
        self.model.net.load_state_dict(checkpoint['model'])
        self.model.net.eval()

    def act(self, observation):
        # increment frame counter
        self.frame_count += 1
        # every 4th frame, select a new action
        if self.frame_count % 4 == 1:
            # clear buffer so we start fresh
            # self.frame_buffer.clear()
            state = self.preprocess(observation)
            # compute new action
            self.current_action = self.model.act(state)
        # return the same action for frames without recomputation
        return self.current_action

    def preprocess(self, obs):
        # convert to grayscale
        frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # resize to 84x84
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        # normalize pixel values to [0,1]
        frame = frame.astype(np.float32) / 255.0
        # append and pad buffer
        self.frame_buffer.append(frame)
        while len(self.frame_buffer) < 4:
            self.frame_buffer.append(frame)
        # stack frames into a single state
        state = np.stack(self.frame_buffer, axis=0)
        return state