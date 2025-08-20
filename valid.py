import os
import torch
from params import get_args
from env.env import JSP_Env
from model.REINFORCE import REINFORCE
import time
from torch.utils.tensorboard import SummaryWriter
from heuristic import *
MAX = float(1e6)

def number(file_name):
    return int(file_name.split('/')[-1][:-4])

def eval_(env, episode, best_ep, best_result, valid_sets=None):

    valid_dir = './datasets/DFJSP/Base_mk04/valid_seed_9569_newjob_Tarr=20_breakdown'
    if valid_sets is None:
        valid_sets = ['(15+20)x8']

    for _set in os.listdir(valid_dir):
        total_result = 0.
        for instance in sorted(os.listdir(os.path.join(valid_dir, _set)), key=number):

            file = os.path.join(os.path.join(valid_dir, _set), instance)

            avai_ops = env.load_instance(file)

            while True:
                data, op_unfinished, job_srpt= env.get_graph_data()
                action_idx, action_prob = policy(avai_ops, data, op_unfinished, job_srpt, env.jsp_instance.graph.max_process_time, greedy=True)
                avai_ops, done = env.step(avai_ops[action_idx])

                if done:
                    ed = time.time()
                    if args.objective == 'tardy_rate':
                        _, solver_result = env.get_tardy_num_rate()
                    elif args.objective == 'makespan':
                        solver_result = env.get_makespan()
                    else:
                        solver_result = env.get_tardiness()
                    total_result += solver_result
#tard = env.get_tardiness()
#total_tard += tard
                    print('date : {} \t instance : {}\t result : {}'.format(args.date, file, solver_result))
                    break
                    
        print('episode : {}\t result : {}'.format(episode, total_result / 100))
        if best_result > (total_result/100):
            best_result=total_result/100
            best_ep=episode
        with open("./result/{}/valid.txt".format(args.date), "a") as out:
            out.write('episode : {}\t result : {}\n'.format(episode, total_result / 100))
    with open("./result/{}/valid.txt".format(args.date), "a") as out:
        out.write('best episode : {}\t result : {}\n'.format(best_ep, best_result ))
    return best_ep, best_result

if __name__ == '__main__':
    args = get_args()
    print(args)
    with open("./result/{}/valid.txt".format(args.date), "a") as out:
        out.write('{}\n'.format(args))
    env = JSP_Env(args)
    policy = REINFORCE(args).to(args.device)
    os.makedirs('./result/{}/'.format(args.date), exist_ok=True)

    start_time = time.time()
    best_result = 1e6
    best_ep = 0
    for episode in range(90000, 201000, 1000):
        if episode == 'best':
            continue
        print(f'date : {args.date} episode : {episode}')
        if os.path.exists('./weight/{}/{}'.format(args.date, episode)) == False:
            break
        policy.load_state_dict(torch.load('./weight/{}/{}'.format(args.date, episode), map_location=args.device), False)
        with torch.no_grad():
            best_ep, best_result = eval_(env, episode, best_ep, best_result, args.valid_sets)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Took: ", execution_time)
