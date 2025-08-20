import numpy as np
import torch
from params import get_args
from env.env import JSP_Env
from env.utils.benchmark_generator import gen_EDDSPT_instance
from model.REINFORCE import REINFORCE
import torch.nn.functional as F
import time
import json
import os
from heuristic import *
import copy
import datetime

import csv
### for 11 fjs 
#config={
#        'ini_job_num' : 3,
#        'new_job_events' : 2,
#        'new_job_per_num' : 0,
#        'machine_num' : 3,
#        'process_time_range' : [1,5],
#        'job_arrival' : False,
#        'machine_breakdown' : True,
#        'job_arrival_time_dist' : 2,
#        'MTBF' : [3, 6],
#        'MTTR' : [1, 2],
#        'task_num' : 20,
#        'c': 1.2,
#        'seed': 9070,
#        'test_idx': datetime.datetime.now().strftime("%d_%H%M%S"),
#        'baseFile': 'dummy',
#}

config={
        'ini_job_num' : 3,
        'new_job_events' : 2,
        'new_job_per_num' : 3,
        'machine_num' : 3,
        'process_time_range' : [1,5],
        'job_arrival' : True,
        'machine_breakdown' : False,
        'job_arrival_time_dist' : 2,
        'MTBF' : [3, 6],
        'MTTR' : [1, 2],
        'task_num' : 20,
        'c': 1.2,
        'seed': 9070,
        'test_idx': datetime.datetime.now().strftime("%d_%H%M%S"),
        'baseFile': 'dummy',
}
random.seed(config['seed'])
np.random.seed(config['seed'])

def get_distribution(d):
    return np.min(d), np.max(d), np.mean(d), np.std(d)
            

def test(test_sets=None):

    
    gen_EDDSPT_instance(config)

    test_dir = './datasets/EDDSPT/'+config['test_idx']+'/' 
    test_set_name = config['test_idx']
#test_dir = test_dir+test_set_name[0]
    test_sets = os.listdir(test_dir)
#test_sets = os.listdir(test_dir+test_set_name[0])
    print(test_sets)

    for _set in test_sets:
        with open('./datasets/EDDSPT/{}/test_result.txt'.format(test_set_name),"a") as outfile:
            outfile.write(f'---------- {_set} ---------- \n')
#        with open('./datasets/EDDSPT/{}/test_result_detail.csv'.format(test_set_name),"a") as csvfile:
#            writer = csv.writer(csvfile)
#            writer.writerow(['instance', 'EDD_tard', 'SPT_tard', 'tard_diff'])

        for size in sorted(os.listdir(os.path.join(test_dir, _set))):
            size_set = os.path.join(test_dir, _set, size)
            max_diff=0
            max_instance=''
            EDD_result=0
            SPT_result=0
            for instance in sorted(os.listdir(size_set)):
                if instance == '@eaDir':
                    continue
                file = os.path.join(size_set, instance)

                avai_ops = env.load_instance(file)
                duedate = env.get_duedate()
                d_min, d_max, d_mean, d_std = get_distribution(duedate)
                processtime = env.get_processtime()
                p_min, p_max, p_mean, p_std = get_distribution(processtime)
#EDD_tard = heuristic_tardiness(copy.deepcopy(env), copy.deepcopy(avai_ops), 'EDD')
                SPT_tard = heuristic_tardiness(copy.deepcopy(env), copy.deepcopy(avai_ops), 'SPT')
                EDD_tard = heuristic_tardiness(env, avai_ops, 'EDD')
                duedate2 = env.get_duedate()
                d2_min, d2_max, d2_mean, d2_std = get_distribution(duedate2)
                processtime2 = env.get_processtime()
                p2_min, p2_max, p2_mean, p2_std = get_distribution(processtime2)
                tard_diff = EDD_tard - SPT_tard
                if tard_diff > max_diff:
                    max_diff = tard_diff
                    max_instance = instance
                    EDD_result=EDD_tard
                    SPT_result=SPT_tard


                with open('./datasets/EDDSPT/{}/test_result.txt'.format(test_set_name),"a") as outfile:
                    outfile.write(f'instance : {file:50} EDD_tard: {EDD_tard:10} SPT_tard: {SPT_tard:10} tard_diff: {tard_diff:10}\n')
                if tard_diff == 0:
                    continue
                with open('./datasets/EDDSPT/{}/test_result.txt'.format(test_set_name),"a") as outfile:
                    outfile.write('\ndue_date init : {:.2f}, {:.2f}, {:.2f}+-{:.2f}\n'.format(d2_min, d2_max, d2_mean, d2_std))
                    for d in duedate:
                        outfile.write('{:.2f}\t'.format(d))
                    outfile.write('\ndue_date after: {:.2f}, {:.2f}, {:.2f}+-{:.2f}\n'.format(d2_min, d2_max, d2_mean, d2_std))
                    for d in duedate2:
                        outfile.write('{:.2f}\t'.format(d))
                    outfile.write('\nprocess time init : {:.2f}, {:.2f}, {:.2f}+-{:.2f}\n'.format(p_min, p_max, p_mean, p_std))
                    for p in processtime:
                        outfile.write('{:.2f}\t'.format(p))
                    outfile.write('\nprocess time after: {:.2f}, {:.2f}, {:.2f}+-{:.2f}\n'.format(p2_min, p2_max, p2_mean, p2_std))
                    for p in processtime2:
                        outfile.write('{:.2f}\t'.format(p))
                    outfile.write('\n')
#               with open('./datasets/EDDSPT/{}/test_result_detail.csv'.format(test_set_name),"a") as csvfile:
#                    writer = csv.writer(csvfile)
#                    writer.writerow([file[-6:], EDD_tard, SPT_tard, tard_diff])


            with open('./datasets/EDDSPT/{}/test_result.txt'.format(test_set_name),"a") as outfile:
                outfile.write(f'instance : {file[:-6]:44} {max_instance} EDD tard : {EDD_result:10} SPT_tard : {SPT_result:10} diff : {max_diff}\nconfig: \n')
                for k in config.keys():
                    outfile.write(f'{k}: {config[k]}\n')

if __name__ == '__main__':
    args = get_args()
    # with open(f'./weight/5e2j_small_postpone/args.json', 'r') as f:
    #     args = json.load(f)

    print(args)
    env = JSP_Env(args)
    start_time = time.time()
#    time.sleep(10)
    test()
    end_time = time.time()
    execution_time = end_time - start_time
    print("Took: ", execution_time)
                    
