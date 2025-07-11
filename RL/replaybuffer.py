import random
from collections import deque, namedtuple

Transision = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'before_legal_mask', 'after_legal_mask'))

class ReplayBuffer():
    def __init__(self, capacity): # int capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, new_state, done, before_legal_mask, after_legal_mask):
        self.buffer.append(Transision(state, action, reward, new_state, done, before_legal_mask, after_legal_mask))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)