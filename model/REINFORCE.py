import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear
from torch.distributions import Categorical
from model.gnn import GNN
import numpy as np
# torch.set_printoptions(precision=10)

class REINFORCE(nn.Module):
    def __init__(self, args):
        super(REINFORCE, self).__init__()
        self.args = args
        self.policy_num_layers = args.policy_num_layers
        self.hidden_dim = args.hidden_dim
        self.gnn = GNN(args)
        self.layers = torch.nn.ModuleList()
        if self.args.objective == 'makespan':
            self.layers.append(nn.Linear(self.hidden_dim * 2 + 1, self.hidden_dim))
        else:
            self.layers.append(nn.Linear(self.hidden_dim * 2 + 2, self.hidden_dim))
        for _ in range(self.policy_num_layers - 2):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.layers.append(nn.Linear(self.hidden_dim, 1))
        
        self.log_probs = []
        self.entropies = []
        self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, Linear):
                if self.args.act == 'leaky_relu':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                elif self.args.act == 'relu':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    raise "act error"
        
    def forward(self, avai_ops, data, op_unfinished, job_srpt, max_process_time, greedy=False, T=1.0):
        x_dict = self.gnn(data)

        if self.args.objective == 'makespan':
            score = torch.empty(size=(0, self.args.hidden_dim * 2 + 1)).to(self.args.device)
        else:
            score = torch.empty(size=(0, self.args.hidden_dim * 2 + 2)).to(self.args.device)

        for op_info in avai_ops:
            normalize_process_time = torch.tensor([op_info['process_time'] / max_process_time], dtype=torch.float32, device=self.args.device)
            if self.args.objective == 'makespan':
                score = torch.cat((score, torch.cat((x_dict['m'][op_info['m_id']],
                                                x_dict['op'][op_unfinished.index(op_info['node_id'])],
#                                                torch.tensor(op_info['process_time'] / max_process_time).to(torch.float32).to(self.args.device).unsqueeze(0)), dim=0).unsqueeze(0)), dim=0)
                                                normalize_process_time),dim=0).unsqueeze(0)), dim=0)

            else:

                score = torch.cat((score, torch.cat((x_dict['m'][op_info['m_id']],
                                                x_dict['op'][op_unfinished.index(op_info['node_id'])],
                                                job_srpt[op_info['job_id']].unsqueeze(0),
                                                normalize_process_time),dim=0).unsqueeze(0)), dim=0)

        for i in range(self.policy_num_layers - 1):
            if self.args.act == 'leaky_relu':
                score = F.leaky_relu(self.layers[i](score))
            elif self.args.act == 'relu':
                score = F.relu(self.layers[i](score))
            else:
                raise "act error"
        score = self.layers[self.policy_num_layers - 1](score)

        probs = F.softmax(score, dim=0).flatten()
        dist = Categorical(probs)
        idx = torch.argmax(score) if greedy else dist.sample()
        self.log_probs.append(dist.log_prob(idx))
        self.entropies.append(dist.entropy())
        return idx.item(), probs[torch.argmax(score)].item()
    
    def calculate_loss(self, tard, baseline):

        advantage = (-1.0) * (tard - baseline) / (baseline + 1)
        advantage = np.sign(advantage) * min(1.0, abs(advantage)) # clip

        policy_loss     = torch.stack(self.log_probs).mean() * advantage
        entropy_loss    = torch.stack(self.entropies).mean()
        loss            = - policy_loss - entropy_loss * self.args.entropy_coef

        return loss, policy_loss, entropy_loss

        # for log_prob, entropy in zip(self.log_probs, self.entropies):
        #     if baseline == 0:
        #         advantage = R * -1
        #     else:
        #         advantage = ((R - baseline) / baseline) * -1
        #     loss.append(-log_prob * advantage - self.args.entropy_coef * entropy)
 
    def clear_memory(self):
        del self.log_probs[:]
        del self.entropies[:]
        return
         
