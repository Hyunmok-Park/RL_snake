import torch
import torch.nn.functional as F
import torch.optim

import numpy as np
from tqdm import tqdm

from net import Qnet
from grid_world import myenv
from buffer import ReplayBuffer

def main():
    device = torch.device('mps')

    q = Qnet(128)
    q_target = Qnet(128)
    env = myenv(10, 10)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer(buffer_limit=10000)

    update_interval = 20
    optimizer = torch.optim.Adam(q.parameters(), lr=0.001)

    for n_epi in tqdm(range(10000), desc="n_epi"):
        eps = max(0.3, 0.9 - 0.01 * (n_epi / 200))
        s = env.reset() #(x,y,f_x,f_y)
        done = False
        score = 0

        while not done:
            a = q.sample_action(torch.from_numpy(np.array(s)).float(), eps)
            next_s, reward, done = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, reward, next_s, done_mask))
            s = next_s
            score += reward
            if done:
                break

        if memory.len() > 2000:
            train(q, q_target, memory, optimizer)

        if n_epi % update_interval == 0:
            q_target.load_state_dict(q.state_dict())
            score = 0

    torch.save(q_target.state_dict(), "qnet")

    x, y, f_x, f_y = env.reset()
    q_target.eval()
    print(x, y, f_x, f_y)
    while True:
        action = q_target(torch.tensor([x, y, f_x, f_y]).float()).argmax().item()
        next_s, reward, done = env.step(action)
        print(action, next_s)
        x, y = next_s[0], next_s[1]
        if done:
            break

def train(q, q_target, memory, opt):
    for i in range(10):
        s_list, a_list, r_list, next_s_list, done_list = memory.sample(32)
        q_out = q(s_list)
        q_a = q_out.gather(1, a_list)
        max_q_prime = q_target(next_s_list).max(1)[0].unsqueeze(1)
        target = r_list + 0.98 * max_q_prime * done_list
        loss = F.smooth_l1_loss(q_a, target)

        opt.zero_grad()
        loss.backward()
        opt.step()


if __name__ == '__main__':
    main()