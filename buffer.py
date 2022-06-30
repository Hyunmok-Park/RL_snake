import collections
import random

import torch

class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, num_sample):
        mini_batch = random.sample(self.buffer, num_sample)
        s_list = []
        a_list = []
        r_list = []
        next_s_list = []
        done_list = []

        for tran in mini_batch:
            s_list.append(tran[0])
            a_list.append([tran[1]])
            r_list.append([tran[2]])
            next_s_list.append(tran[3])
            done_list.append([tran[4]])

        s_list = torch.tensor(s_list, dtype=torch.float)
        a_list = torch.tensor(a_list)
        r_list = torch.tensor(r_list)
        next_s_list = torch.tensor(next_s_list, dtype=torch.float)
        done_list = torch.tensor(done_list)

        return s_list, a_list, r_list, next_s_list, done_list

    def len(self):
        return len(self.buffer)