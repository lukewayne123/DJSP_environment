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
import argparse
from djsp_plotter import DJSP_Plotter

config={
        'ini_job_num' : 3,
        'new_job_events' : 2,
        'new_job_per_num' : 0,
        'machine_num' : 3,
        'process_time_range' : [1,5],
#        'job_arrival' : True,
        'job_arrival' : False,
        'machine_breakdown' : True,
        'job_arrival_time_dist' : 2,
        'MTBF' : [3, 6],
#        'MTBF' : [30, 50],
#        'MTBF' : [20, 40],
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
            

def test(file, idx):

    
    avai_ops = env.load_instance(file)
    EDD_tard = heuristic_tardiness(env, avai_ops, 'EDD')
    env.log('./datasets/EDDSPT/{}/EDD'.format(idx))
    EDDplotter = DJSP_Plotter(env.logger)
    html_out_file = os.path.join('./datasets/EDDSPT/{}/EDD_tard{}.html'.format(idx,EDD_tard))
    EDDplotter.plot_googlechart_timeline(html_out_file)

    env.reset()

    avai_ops = env.load_instance(file)
    SPT_tard = heuristic_tardiness(env, avai_ops, 'SPT')
    env.log('./datasets/EDDSPT/{}/SPT'.format(idx))
    SPTplotter = DJSP_Plotter(env.logger)
    html_out_file = os.path.join('./datasets/EDDSPT/{}/SPT_tard{}.html'.format(idx,SPT_tard))
    SPTplotter.plot_googlechart_timeline(html_out_file)



if __name__ == '__main__':
    args = get_args()
    # with open(f'./weight/5e2j_small_postpone/args.json', 'r') as f:
    #     args = json.load(f)

    print(args)
    env = JSP_Env(args)
    start_time = time.time()
#    time.sleep(10)
    test(args.test_file,args.test_dir)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Took: ", execution_time)
                    
