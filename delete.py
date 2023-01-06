import torch
import torch.nn as nn

obs_size = 100
fc_size = 200
lstm_state_size = 300

fc1 = nn.Linear(obs_size, fc_size)

print(fc1)

x = fc1.weight.new(1, lstm_state_size).zero_().squeeze(0)
print(x.shape)