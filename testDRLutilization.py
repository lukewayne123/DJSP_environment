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
#        test_dir = './datasets/DFJSP/MK/'
#        test_dir = './datasets/DFJSP/MK_1207/'
#        test_dir = './datasets/DFJSP/MK_1210/' 
#        test_dir = './datasets/DFJSP/MK_1213U/' 
#        test_dir = './datasets/DFJSP/MK_1215/' 
#        test_dir = './datasets/DFJSP/MK10new/' 
        test_dir = './datasets/DFJSP/'+args.test_dir+'/' 
        test_sets = os.listdir(test_dir)
        print(test_sets)
#        test_dir = './datasets/DFJSP/Base_mk04'
#        if test_sets is None:
#            test_sets = [
#                            'seed_2011_newjob_Tarr=20_breakdown_Tbreak=[60, 80]',
#                            'seed_8039_newjob_Tarr=20_breakdown_Tbreak=[40, 60]',
#                            'seed_8914_newjob_Tarr=15_breakdown',
#                            'seed_6404_newjob_Tarr=20_breakdown',
#                            'seed_1468_newjob_Tarr=25_breakdown',
#                            'seed_1855_newjob_Tarr=30_breakdown',
#                        ]

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
#           writer.writerow(['instance', 'sys_util', 'sys_util_idv', 'tard'])
            writer.writerow(['instance', 'tard', 'tardy', 'tardy_r', 'sys_util', 'sys_util_idv'])
        for size in sorted(os.listdir(os.path.join(test_dir, _set))):
            size_set = os.path.join(test_dir, _set, size)
            avg_tard = 0
            avg_sys_util = 1.0
            avg_sys_util_idv = 1.0
            avg_tardy_num = 0
            avg_tardy_rate = 0
            for instance in sorted(os.listdir(size_set)):
                if instance == '@eaDir':
                    continue
                best_tard = 1e6
                best_util_sys = 0.0
                best_util_idv = 0.0
                best_tardy_num = 1e6
                best_tardy_rate = 1.0
                file = os.path.join(size_set, instance)

                if args.test_sample_times > 1 :
                    # apex
                    N = args.test_sample_times
                    alpha, epsilon = 7, 0.4
                    apex = np.array([epsilon ** (1 + i / (N - 1) * alpha) for i in range(N)])
                    apex /= sum(apex)
                    all_T = np.random.choice(sorted(np.random.uniform(0, 1.0, size=N)), N, p=apex)

                for cnt in range(args.test_sample_times):

                    avai_ops = env.load_instance(file)

                    while True:
                        data, op_unfinished, job_srpt= env.get_graph_data()
                        if cnt == 0:
                            action_idx, action_prob = policy(avai_ops, data, op_unfinished, job_srpt, env.jsp_instance.graph.max_process_time, greedy=True)
                        # else:
                            # action_idx, action_prob = policy(avai_ops, data, op_unfinished, job_srpt, env.jsp_instance.graph.max_process_time, greedy=True)
                        avai_ops, done = env.step(avai_ops[action_idx])
                        
                        if done:
                            if best_tard > env.get_tardiness():
                                best_tard = env.get_tardiness()
                            if best_tard == env.get_tardiness():
                                current_util_idv = env.get_individual_utilization()
                                current_util_sys = env.get_system_utilization()
                                current_tardy_num, current_tardy_rate = env.get_tardy_num_rate()
                                if current_util_idv > best_util_idv:
                                    best_util_idv = current_util_idv
                                if current_util_sys > best_util_sys:
                                    best_util_sys = current_util_sys
                                if current_tardy_num < best_tardy_num:
                                    best_tardy_num = current_tardy_num
                                if current_tardy_rate < best_tardy_rate:
                                    best_tardy_rate = current_tardy_rate
                                

                            print("instance : {}, tard : {}".format(file, env.get_tardiness()))
                            break

                with open('./result/{}/test_result.txt'.format(args.date),"a") as outfile:
#outfile.write(f'instance : {file:50} tard : {best_tard:10} \n')
                    outfile.write(f'instance : {file:50} util_sys : {best_util_sys:4} util_idv : {best_util_idv:4} tard: {best_tard:10} tardy_num: {best_tardy_num:10} tardy_rate: {best_tardy_rate:10}\n')
                with open('./result/{}/test_result_detail.csv'.format(args.date),"a") as csvfile:
                    writer = csv.writer(csvfile)
#writer.writerow([file[-6:], best_util_sys, best_util_idv, best_tard])
                    writer.writerow([file[-6:], best_tard, best_tardy_num, best_tardy_rate, best_util_sys, best_util_idv])

                avg_tard += best_tard
                avg_sys_util += best_util_sys
                avg_sys_util_idv += best_util_idv
                avg_tardy_num += best_tardy_num
                avg_tardy_rate += best_tardy_rate


            print(f'instance : {file[:-6]:44}, AVG tard : {avg_tard//10:10}')
            with open('./result/{}/test_result.txt'.format(args.date),"a") as outfile:
                outfile.write(f'instance : {file[:-6]:44} AVG tard : {avg_tard//10:10} \n')
            with open('./result/{}/test_result_avg.csv'.format(args.date),"a") as csvfile:
                writer = csv.writer(csvfile)
#writer.writerow([file[:-6], avg_sys_util / 10.0, avg_sys_util_idv / 10.0, avg_tard/10.0])
                writer.writerow([file[:-6], avg_tard/10.0, avg_tardy_num/10.0, avg_tardy_rate/10.0, avg_sys_util / 10.0, avg_sys_util_idv / 10.0])

if __name__ == '__main__':
    args = get_args()
    # with open(f'./weight/5e2j_small_postpone/args.json', 'r') as f:
    #     args = json.load(f)

    print(args)
    env = JSP_Env(args)
    policy = REINFORCE(args).to(args.device)
    
#policy.load_state_dict(torch.load('./weight/{}/200000'.format(args.date), map_location=args.device), False)
#policy.load_state_dict(torch.load('./weight/5e2j_small_postpone/193000', map_location=args.device), False)
#    policy.load_state_dict(torch.load('./weight/1009_ScNwAwB_t1/192000', map_location=args.device), False)
#    policy.load_state_dict(torch.load('./weight/1204_RSwAwB_t1/101000', map_location=args.device), False)
#    policy.load_state_dict(torch.load('./weight/1204_RSwoAwoB_t1/153000', map_location=args.device), False)
#    policy.load_state_dict(torch.load('./weight/1204_RSwAwoB_t1/123000', map_location=args.device), False)
#    policy.load_state_dict(torch.load('./weight/1204_ScNwAwB_t1check/110000', map_location=args.device), False)
#    policy.load_state_dict(torch.load('./weight/1204_ScNwoAwB_t1/132000', map_location=args.device), False)
#    policy.load_state_dict(torch.load('./weight/1204_ScNwoAwoB_t1/192000', map_location=args.device), False)
#    policy.load_state_dict(torch.load('./weight/1212_RSwAwBFIFO_tardy_m2cu1/116000', map_location=args.device), False)
#    policy.load_state_dict(torch.load('./weight/1212_RSwAwBSPT_tardy_m6cu1/134000', map_location=args.device), False)
#    policy.load_state_dict(torch.load('./weight/1212_RSwAwBEDD_tardy_m4cu1/143000', map_location=args.device), False)
#    policy.load_state_dict(torch.load('./weight/1217_RSwoAwoB_t1pos/161000', map_location=args.device), False)
#    policy.load_state_dict(torch.load('./weight/1217_RSwAwB_t1pos/184000', map_location=args.device), False)
#    policy.load_state_dict(torch.load('./weight/1217_ScNwoAwoB_t1pos/92000', map_location=args.device), False)
#    policy.load_state_dict(torch.load('./weight/1217_ScNwAwB_t1pos/165000', map_location=args.device), False)
###
# obj = tardy rate
###
#    policy.load_state_dict(torch.load('./weight/1212_RSwAwBEDD_tardy_m4cu1/150000', map_location=args.device), False)
#    policy.load_state_dict(torch.load('./weight/1212_RSwoAwoBEDD_tardy_m4cu1/114000', map_location=args.device), False)
#    policy.load_state_dict(torch.load('./weight/1212_RSwoAwoBSPT_tardy_m6cu1/129000', map_location=args.device), False)
    policy.load_state_dict(torch.load('./weight/1212_ScNwoAwoBEDD_tardy_m1cu0/116000', map_location=args.device), False)
#    policy.load_state_dict(torch.load('./weight/1212_ScNwAwBEDD_tardy_m2cu1/190000', map_location=args.device), False)
#    policy.load_state_dict(torch.load('./weight/1226_RSEDDwoAwoB_t1pos/195000', map_location=args.device), False)
#    policy.load_state_dict(torch.load('./weight/1226_RSwAwBEDD_tardy_pos_m4cu1/175000', map_location=args.device), False)
#    policy.load_state_dict(torch.load('./weight/1231_ScNEDDwoAwoB_t1pos/169000', map_location=args.device), False)
    start_time = time.time()
    with torch.no_grad():
        test()
    end_time = time.time()
    execution_time = end_time - start_time
    print("Took: ", execution_time)
                    
