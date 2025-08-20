import numpy as np
import copy
import os
import random
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from params import get_args
from env.env import JSP_Env
from model.REINFORCE import REINFORCE
from heuristic import *
from torch.utils.tensorboard import SummaryWriter
import json
import time

MAX = float(1e6)


def train():
    print("start Training")
    best_valid_makespan = MAX
    
    if args.train_arr == False:
        args.new_job_event = 0

    for episode in range(1, args.episode):

        if episode % 1000 == 0:
            torch.save(policy.state_dict(), "./weight/{}/{}".format(args.date, str(episode)))

        action_probs = []
        avai_ops = env.reset()
        while avai_ops is None:
            avai_ops = env.reset()

        if args.objective == 'tardy_rate':
            rule_result, _ = heuristic_tardy(copy.deepcopy(env), copy.deepcopy(avai_ops), args.rule)
        elif args.objective == 'makespan':
            rule_result = heuristic_makespan(copy.deepcopy(env), copy.deepcopy(avai_ops), args.rule)
        else:
            rule_result = heuristic_tardiness(copy.deepcopy(env), copy.deepcopy(avai_ops), args.rule)

        while True:
            data, op_unfinished, job_srpt = env.get_graph_data()
            action_idx, action_prob = policy(avai_ops, data, op_unfinished, job_srpt, env.jsp_instance.graph.max_process_time)
            avai_ops, done = env.step(avai_ops[action_idx])

            action_probs.append(action_prob)
            
            if done:

                optimizer.zero_grad()
                if args.objective == 'tardy_rate':
                    solver_result, _ = env.get_tardy_num_rate()
                elif args.objective == 'makespan':
                    solver_result = env.get_makespan()
                else:
                    solver_result = env.get_tardiness()
                loss, policy_loss, entropy_loss = policy.calculate_loss(solver_result, rule_result)
                loss.backward()

                if episode % 10 == 0:
                    writer.add_scalar("action prob", np.mean(action_probs),episode)
                    writer.add_scalar("loss", loss, episode)
                    writer.add_scalar("policy_loss", policy_loss, episode)
                    writer.add_scalar("entropy_loss", entropy_loss, episode)
                
                optimizer.step()
                scheduler.step()

                policy.clear_memory()
                improve = round(rule_result - solver_result, 1)
                print("Episode : {} \t\tJob : {} \t\tMachine : {} \t\tPolicy : {} \t\tImprove: {} \t\t {} : {}".format(
                    episode, env.jsp_instance.job_num, env.jsp_instance.machine_num, 
                    solver_result, improve, args.rule, rule_result))
                break

if __name__ == '__main__':
    args = get_args()
    print(args)

    os.makedirs('./result/{}/'.format(args.date), exist_ok=True)
    os.makedirs('./weight/{}/'.format(args.date), exist_ok=True)

    with open("./result/{}/args.json".format(args.date), "a") as outfile:
        json.dump(vars(args), outfile, indent=8)

    env = JSP_Env(args)
    policy = REINFORCE(args).to(args.device)

    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.99)
    writer = SummaryWriter(comment=args.date)
    # writer_valid = SummaryWriter(comment="{}_valid".format(args.date))

    train()
    
