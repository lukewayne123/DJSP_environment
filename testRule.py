import numpy as np
import torch
from params import get_args
from env.env import JSP_Env
from model.REINFORCE import REINFORCE
import torch.nn.functional as F
import time
import json
import os
from heuristic import *
import copy

import csv

def test(test_sets=None):

    if args.instance_type == 'FJSP':

        test_dir = './datasets/DFJSP/'+args.test_dir+'/' 
#        test_dir = './datasets/DFJSP/edata/' 
#        test_dir = './datasets/DFJSP/rdata/' 
#        test_dir = './datasets/DFJSP/vdata/' 
        test_sets = os.listdir(test_dir)
        print(test_sets)

    else:
        test_dir = './datasets/DJSP'
        if test_sets is None:
            test_sets = ['(10+20)x10_DJSP']

    os.makedirs('./result/{}'.format(args.date), exist_ok=True)

    for _set in test_sets:
        with open('./result/{}/test_result.txt'.format(args.date),"a") as outfile:
            outfile.write(f'---------- {_set} ---------- \n')
        with open('./result/{}/test_result_detail.csv'.format(args.date),"a") as csvfile:
            writer = csv.writer(csvfile)
#            writer.writerow(['instance', 'sys_util', 'sys_util_idv', 'tard'])
            writer.writerow(['instance', 'tard', 'tardy', 'tardy_r', 'sys_util', 'sys_util_idv'])
        with open('./result/{}/test_result_avg.csv'.format(args.date),"a") as csvfile:
            writer = csv.writer(csvfile)
#            writer.writerow(['instance', 'sys_util', 'sys_util_idv', 'tard'])
            writer.writerow(['instance', 'tard', 'tardy', 'tardy_r', 'sys_util', 'sys_util_idv'])

        for size in sorted(os.listdir(os.path.join(test_dir, _set))):
            size_set = os.path.join(test_dir, _set, size)
            avg_tard = 0
            avg_sys_util = 0
            avg_sys_util_idv = 0
            avg_tardy_num = 0
            avg_tardy_rate = 0
            for instance in sorted(os.listdir(size_set)):
                if instance == '@eaDir':
                    continue
                best_tard = 1e6
                file = os.path.join(size_set, instance)

                avai_ops = env.load_instance(file)
                tard = heuristic_tardiness(env, avai_ops, args.rule)

                util_idv = env.get_individual_utilization()
                util_sys = env.get_system_utilization()

                tardy_num, tardy_rate = env.get_tardy_num_rate()

                process_time = env.get_total_process_time()
                makespan = env.get_makespan()
#print(process_time, makespan)
                print(file, util_idv, util_sys, tard, tardy_num, tardy_rate)

                with open('./result/{}/test_result.txt'.format(args.date),"a") as outfile:
                    outfile.write(f'instance : {file:50} util_sys : {util_sys:4} util_idv : {util_idv:4} tard: {tard:10} tardy_num: {tardy_num:10} tardy_rate: {tardy_rate:10}\n')
                with open('./result/{}/test_result_detail.csv'.format(args.date),"a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([file[-6:], tard, tardy_num, tardy_rate, util_sys, util_idv])

                avg_tard += tard
                avg_sys_util += util_sys
                avg_sys_util_idv += util_idv
                avg_tardy_num += tardy_num
                avg_tardy_rate += tardy_rate


            print(f'instance : {file[:-6]:44}, AVG tard : {avg_tard//10:10}')
            with open('./result/{}/test_result.txt'.format(args.date),"a") as outfile:
                outfile.write(f'instance : {file[:-6]:44} AVG tard : {avg_tard//10:10} \n')
            with open('./result/{}/test_result_avg.csv'.format(args.date),"a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([file[:-6], avg_tard/10.0, avg_tardy_num/10.0, avg_tardy_rate/10.0, avg_sys_util / 10.0, avg_sys_util_idv / 10.0])

if __name__ == '__main__':
    args = get_args()
    # with open(f'./weight/5e2j_small_postpone/args.json', 'r') as f:
    #     args = json.load(f)

    print(args)
    env = JSP_Env(args)
    start_time = time.time()
    test()
    end_time = time.time()
    execution_time = end_time - start_time
    print("Took: ", execution_time)
                    
