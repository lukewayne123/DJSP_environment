import numpy as np
import random
import os
import time
# random.seed(6404)
# np.random.seed(6404)

from itertools import accumulate

import argparse

MAX=1e6

# FJSP
def gen_operations_FJSP(machine_num, op_process_time_range): 
    op = []
    
    op_num = random.randint(1, machine_num + 2)
#print(op_num, machine_num)
    # op_num = random.randint(2, 4)

    for op_id in range(op_num):
        random_size = random.randint(1, machine_num) # the number of usable machine for this operation
        # random_size = random.randint(1, 3) # the number of usable machine for this operation
        m_id = sorted(np.random.choice(machine_num, size=random_size, replace=False)) # the set of index of usable machine id with size random_size
        mach_ptime = []
        for id in m_id:
            process_time = np.random.randint(*op_process_time_range)
            mach_ptime.append((id, process_time))
        op.append({"id": op_id, "machine_and_processtime": mach_ptime})
# print(op)
    return op

# https://stackoverflow.com/questions/71593308/creating-random-number-using-exponential-distribution-in-a-specified-range
def gen_range_exp(scale, size, t_min, t_max):
    random_u = np.random.rand(size)
    u_min = np.exp(-t_min * scale)
    u_max = np.exp(-t_max * scale)

#u = u_min + (1-random_u/2) * (u_max - u_min)
#u = u_min + (1-random_u/t_max) * (u_max - u_min)
    u = u_min + random_u * (u_max - u_min)
    return t_min + (-np.log(u) / scale) * (t_max - t_min)

def scale_exp(t, t_min, t_max):
    if t[0] > t_min and t[-1] < t_max:
        return t
    elif t[0] > t_min and t[-1] > t_max:
        # resize
        return [ int(i * t_max / t[-1]) for i in t]
    else:
        # shift and resize
        return [int(t_min + i * (t_max-t_min) / t[-1]) for i in t]


def gen_instance_FJSP(config):

#dir_name = './({}+{})x{}_TEST'.format(config['ini_job_num'], config['new_job_events'] * config['new_job_per_num'], config['machine_num'])
#dir_name = './{}({}+{})x{}_seed{}'.format(config['baseFile'],config['ini_job_num'], config['new_job_events'] * config['new_job_per_num'], config['machine_num'], config['seed'])
#    dir_name = './{}_m{}_seed{}_C{}'.format(config['baseFile'], config['machine_num'], config['seed'], config['c'])
    time_range=[i for i in np.linspace(0,config['t_base'],config['times'])]
    time_range.append(config['t_base']*(1+1/(config['times']-1)))

    dirs = []
    for i in range(config['times']):
        dir_name = './{}_makespan_t{}_{}_C{}_m{}_seed{}'.format(config['baseFile'], time_range[i], time_range[i+1], config['c'], config['machine_num'], config['seed'])
        if config['job_arrival']:
            dir_name += '_arr{:.2f}'.format(config['job_arrival_time_dist'])
        if config['machine_breakdown']:
            dir_name += '_break{}_{}'.format(config['MTBF'][0], config['MTBF'][1])
        else:
            dir_name += 'noBreak'
        dir_name += '/({}+{})x{}'.format(config['ini_job_num'], config['new_job_events'] * config['new_job_per_num'], config['machine_num'])
        os.makedirs(dir_name, exist_ok=True)
        dirs.append(dir_name)

#    dir_name1 = './{}_makespan_t{}_{}_C{}_m{}_seed{}'.format(config['baseFile'], time_range[0], time_range[1], config['c'], config['machine_num'], config['seed'])
#    dir_name2 = './{}_makespan_t{}_{}_C{}_m{}_seed{}'.format(config['baseFile'], time_range[1], time_range[2], config['c'], config['machine_num'], config['seed'])
#    dir_name3 = './{}_makespan_t{}_{}_C{}_m{}_seed{}'.format(config['baseFile'], time_range[2], time_range[3], config['c'], config['machine_num'], config['seed'])
#    if config['job_arrival']:
#        dir_name1 += '_arr{:.2f}'.format(config['job_arrival_time_dist'])
#        dir_name2 += '_arr{:.2f}'.format(config['job_arrival_time_dist'])
#        dir_name3 += '_arr{:.2f}'.format(config['job_arrival_time_dist'])
#    if config['machine_breakdown']:
#        dir_name1 += '_break{}_{}'.format(config['MTBF'][0], config['MTBF'][1])
#        dir_name2 += '_break{}_{}'.format(config['MTBF'][0], config['MTBF'][1])
#        dir_name3 += '_break{}_{}'.format(config['MTBF'][0], config['MTBF'][1])
#    else:
#        dir_name1 += 'noBreak'
#        dir_name2 += 'noBreak'
#        dir_name3 += 'noBreak'
#    dir_name1 += '/({}+{})x{}'.format(config['ini_job_num'], config['new_job_events'] * config['new_job_per_num'], config['machine_num'])
#    dir_name2 += '/({}+{})x{}'.format(config['ini_job_num'], config['new_job_events'] * config['new_job_per_num'], config['machine_num'])
#    dir_name3 += '/({}+{})x{}'.format(config['ini_job_num'], config['new_job_events'] * config['new_job_per_num'], config['machine_num'])
#    os.makedirs(dir_name1, exist_ok=True)
#    os.makedirs(dir_name2, exist_ok=True)
#    os.makedirs(dir_name3, exist_ok=True)
#    dirs = [dir_name1, dir_name2, dir_name3]

    for case in range(config['task_num']):
        job_arrival_cases=[]
        if config['job_arrival'] is True:
            random_arrival_time = np.random.exponential(scale=config['job_arrival_time_dist'], size=(config['new_job_events']+1)) + 1
#random_arrival_time = gen_range_exp(config['job_arrival_time_dist'], config['new_job_events'], config['t_min'], config['t_max'])+1
#print(random_arrival_time)
            job_arrival_time = list(accumulate(map(int, random_arrival_time)))
            print(job_arrival_time)
#job_arrival_time = scale_exp(job_arrival_time, config['t_min'], config['t_max'])[:-1]
#job_arrival_time1 = scale_exp(job_arrival_time, time_range[0], time_range[1])[:-1]
#job_arrival_time2 = scale_exp(job_arrival_time, time_range[1], time_range[2])[:-1]
#job_arrival_time3 = scale_exp(job_arrival_time, time_range[2], time_range[3])[:-1]
            for i in range(config['times']):
                job_arrival_cases.append(scale_exp(job_arrival_time, time_range[i], time_range[i+1])[:-1])
#job_arrival_cases.append(scale_exp(job_arrival_time, time_range[0], time_range[1])[:-1])
#job_arrival_cases.append(scale_exp(job_arrival_time, time_range[1], time_range[2])[:-1])
#job_arrival_cases.append(scale_exp(job_arrival_time, time_range[2], time_range[3])[:-1])
            print(job_arrival_time)

        cnt = 0.0
        op_num = 0.0
        job_configs = []
        configs = []
        base_configs = []
        for i in range(config['ini_job_num'] + config['new_job_events'] * config['new_job_per_num']):
            job_conf = gen_operations_FJSP(config['machine_num'], config['process_time_range'])
            op_num += len(job_conf)
            job_configs.append(job_conf)
            for op in job_conf:
                cnt += len(op['machine_and_processtime'])

        if config['baseFile'] != 'dummy':
            with open('{}'.format(config['baseFilePath']), "r") as source:
                lines = source.readlines()[1:]
                for line in lines:
                    if line != '\n':
                        if line[-1] != '\n':
                            line+='\n'
                        base_configs.append(line)
#print(base_configs)
        for dir_name in dirs:

            with open('{}/{}.fjs'.format(dir_name, case), "w") as outfile:
                outfile.write('{} {} {}\n'.format(config['ini_job_num'], config['new_job_events'] * config['new_job_per_num'], config['machine_num']))
                base_idx = 0
                if config['baseFile'] != 'dummy':
                    for base_conf in base_configs:
                        outfile.write('{} {}'.format(0, base_conf))
                    base_idx = len(base_configs) 
                for idx, job_conf in enumerate(job_configs):
                    if (idx + base_idx) >= (config['ini_job_num'] + config['new_job_events'] * config['new_job_per_num']):
                        break
                    if (idx + base_idx) < config['ini_job_num']:
                        outfile.write('{} {} '.format(0, len(job_conf)))
                    else:
                        outfile.write('{} {} '.format(job_arrival_cases[dirs.index(dir_name)][(idx + base_idx - config['ini_job_num']) // config['new_job_per_num']], len(job_conf)))
    #print(job_arrival_time[(idx + base_idx - config['ini_job_num']) // config['new_job_per_num']])
                    for op in job_conf:
                        outfile.write('{} '.format(len(op['machine_and_processtime'])))
                        for pair in op['machine_and_processtime']:
                            outfile.write('{} {} '.format(pair[0] + 1, pair[1]))
                    outfile.write('\n')

        if config['machine_breakdown'] is True:
            machine_breakdown_events = list()
            machine_info = {}
            for idx in range(config['machine_num']):
                failure_dist = np.random.randint(*config['MTBF'])  # freq
                repair_dist = np.random.randint(*config['MTTR'])  # length
                breakdown_event = list()
                breakdown_time, repair_time = 0, 0
                for _ in range(100):
                    breakdown_time += repair_time + int(np.random.exponential(scale=failure_dist)) + 1
                    repair_time = int(np.random.exponential(scale=repair_dist)) + 1
                    breakdown_event.append((breakdown_time, repair_time))
                machine_breakdown_events.append(breakdown_event)
                machine_info['m{}'.format(idx)] = {'MTBF': failure_dist, 'MTTR': repair_dist}

            for dir_name in dirs:
                with open('{}/{}.fjs'.format(dir_name, case), "a") as outfile:
                    for idx in range(config['machine_num']):
                        outfile.write('m{} {} {} '.format(idx, machine_info['m{}'.format(idx)]['MTBF'], machine_info['m{}'.format(idx)]['MTTR']))
                    outfile.write('BREAKDOWN EVENTS:\n')
                    for breakdown_event in machine_breakdown_events:
                        outfile.write('{}\n'.format(breakdown_event))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for benchmark generator')
    parser.add_argument('--baseFile', type=str, default='dummy')
    parser.add_argument('--baseFilePath', type=str, default='dummy')
    parser.add_argument('--c', type=float, default=1.2)
    parser.add_argument('--seed', type=int, default=6404)
    parser.add_argument('--minMTBF', type=int, default=50)
    parser.add_argument('--maxMTBF', type=int, default=70)
    parser.add_argument('--t_min', type=int, default=0)
    parser.add_argument('--t_max', type=int, default=350)
    parser.add_argument('--t_base', type=int, default=700)
    parser.add_argument('--times', type=int, default=5)
    args = parser.parse_args()

    config = {
        'ini_job_num' : 15,
        'new_job_events' : 2,
#        'new_job_events' : 0,
        'new_job_per_num' : 5,
#        'new_job_per_num' : 0,
        'machine_num' : 8,
        'process_time_range' : [1,10],
        'job_arrival' : True,
#        'job_arrival' : False,
        'machine_breakdown' : True,
#        'machine_breakdown' : False,
#        'job_arrival_time_dist' : 25,
        'job_arrival_time_dist' : 20,
#        'job_arrival_time_dist' : 15,
#        'MTBF' : [110, 130],
#        'MTBF' : [70, 90],
        'MTBF' : [args.minMTBF, args.maxMTBF],
#        'MTBF' : [30, 50],
#        'MTBF' : [20, 40],
        'MTTR' : [10, 20],
#        'task_num' : 10,
        'task_num' : 10,
        'baseFile': args.baseFile,
        'baseFilePath': args.baseFilePath,
        'c': args.c,
        'seed': args.seed,
#        'seed': 6404 # MTBF 50, 70 1.25 > C > 1  
#        'seed': 2011 # C=1.2, MTBF 70, 90 1/U ~= 1.1875
#        'seed': 8039 # C=1.2, MTBF 40, 60 1/U ~= 1.375
#        'seed': 8920 # C=1.2, MTBF 100, 120 1/U ~= 1.125
#        'seed': 4823 # C=1.2, MTBF 40, 60 1/U ~= 1.5
#        'seed': 8672 # C=1.2, 1/U = 1 no machine break
#        'c': 1.4,
#        'seed': 1468 # MTBF 50, 70
#        'c': 1,
#        'seed': 8914 # C=1.2, MTBF 50, 70 1.25 > C > 1
#        'c': 0.6,
#'seed': 3605 # C=0.6, MTBF 50, 70 1.25 > C > 1 ==> or tools wrong with case 3 & 4
#        'seed': 5063 # C=0.6, MTBF 50, 70 1.25 > C > 1
#        'c': 2.4,
#        'seed': 7638 # MTBF 50, 70
#        'c': 5,
#        'c': 3,
#        'c': 7,
#        'seed': 3436
          't_min': args.t_min,
          't_max': args.t_max,
          't_base': args.t_base,
          'times': args.times,
    }
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    if config['baseFile'] != 'dummy':
        with open('{}'.format(config['baseFilePath']), "r") as source:
            lines = source.readlines()[0]
            if args.baseFile[-4:-2] == 'MK':
                nxm = lines.split("\t")[:2] # for mk
            else:
                nxm = lines.split("  ")[:2] # for la
#print(nxm)
            config['ini_job_num'] = int(nxm[0])
            config['machine_num'] = int(nxm[1])
            
            config['job_arrival_time_dist'] = 5.5*5*config['c'] / 2.0
            
            
    print(config)
    # basic
#gen_instance_FJSP(config)
#    for i in range(8):
#        config['new_job_events'] = 2*(i+3)
#        gen_instance_FJSP(config)
    config['new_job_events'] = 2*(3)
    gen_instance_FJSP(config)
#    config['new_job_events'] = 2*(5)
#    gen_instance_FJSP(config)
#    config['new_job_events'] = 2*(8)
#    gen_instance_FJSP(config)
#    config['new_job_events'] = 2*(10)
#    gen_instance_FJSP(config)
    # for arrival_time_dist in arrival_time_dists:
    #     for new_job_num in new_job_nums:
    #         gen_instance_FJSP_job_insert(ini_job_num, new_job_num, machine_num, arrival_time_dist)
   
