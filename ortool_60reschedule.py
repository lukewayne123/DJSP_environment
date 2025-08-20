import collections
import os
import json
from ortools.sat.python import cp_model
from pprint import pprint
import shutil
import copy
import time
import argparse

MAX = 1e6

class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0

    def on_solution_callback(self):
        """Called at each new solution."""
        print('Solution %i, time = %f s, objective = %i' %
              (self.__solution_count, self.WallTime(), self.ObjectiveValue()))
        self.__solution_count += 1


def load_instance(path, DDT):
    fjsp_data = []
    job_arrival_time = []
    job_due_date = []
    machine_breakdown_events = []
    total_alt_num = 0

    with open(path, 'r') as fobj:
        ini_job_num, new_job_num, machine_num = map(int, fobj.readline().split())

        for _ in range(ini_job_num + new_job_num):
            line = fobj.readline().split()
            arrival_time, op_num = map(int, line[:2])
            job_arrival_time.append(arrival_time)

            job_data = []
            expect_process_time = 0. + arrival_time

            start = 2

            for _ in range(op_num):
                alt_num = int(line[start])
                alt_data = list(map(int, line[start + 1: start + 1 + 2 * alt_num]))
                total_alt_num += alt_num

                expect_process_time += sum(alt_data[1::2]) / alt_num

                op_data = [(alt_data[i + 1], alt_data[i] - 1) for i in range(0, len(alt_data), 2)]
                job_data.append(op_data)

                start += 1 + 2 * alt_num
            assert op_num == len(job_data)

            fjsp_data.append(job_data)
            job_due_date.append(int(expect_process_time * DDT)) #0117 debug
#            job_due_date.append(int(expect_process_time))

        line = fobj.readline()

        for _ in range(machine_num):
            line = fobj.readline().replace('[', ' ').replace(']', ' ').replace('(', ' ').replace(')', ' ').replace(',', ' ').split()
            breakdown_event = [(int(t), int(d)) for t, d in zip(line[::2], line[1::2])]
            machine_breakdown_events.append(breakdown_event)
    print(job_due_date)
#    input()
    return machine_num, fjsp_data, job_arrival_time, job_due_date, machine_breakdown_events


def time_stamp(item):
    return item[1]

def st_time(op_info):
    return op_info['start_time']

def record_information(op_info, fjsp_data, current_op_id, start_time, model, machine_occupied):
    job_id, machine_id, alt  = op_info['job_id'], op_info['machine_id'], op_info['alternative']
    start, duration, end     = op_info['start_time'], op_info['process_time'], op_info['finish_time']

    fjsp_data[job_id].pop(0)
    op_id = current_op_id[job_id]
    current_op_id[job_id] += 1
    start_time[job_id].append(end)

    op_name = str(job_id) + '_' + str(op_id) + '_' + str(alt)
    op = model.NewIntervalVar(start, duration, end, op_name)
    machine_occupied[machine_id].append(op)

def is_finish(fjsp_data, start_time, event_holder):
    return all(not job for job in fjsp_data) and (not event_holder or max(st[-1] for st in start_time) <= event_holder[0][1])

def output_json(filename, schedule):
    infos = []
    for m_id in range(len(schedule)):
        for info in schedule[m_id]:
            if info.Name() == 'breakdown':
                breakdown = {
                    'Order':        None,
                    'job_id':       -1,
                    'op_id':        -1,
                    'machine_id':   m_id,
                    'start_time':   info.Proto().start.offset,
                    'process_time': info.Proto().size.offset,
                    'finish_time':  info.Proto().end.offset,
                    'alternative':  -1,
                }
                infos.append(breakdown)
            else:
                op = {
                    'Order':        None,
                    'job_id':       int(info.Name().split('_')[0]),
                    'op_id':        int(info.Name().split('_')[1]),
                    'machine_id':   m_id,
                    'start_time':   info.Proto().start.offset,
                    'process_time': info.Proto().size.offset,
                    'finish_time':  info.Proto().end.offset,
                    'alternative':  None,
                }
                infos.append(op)
            # print(op_info.Proto())
    with open(filename, 'w') as out:
        json.dump(infos, out, indent=4)

def solver(path, timelinedir, time_limit=60, num_thread=1, enable_sol_printer=False, DDT=1.5, objective='tardiness'):
    print(path)
    machine_num, fjsp_data, job_arrival_time, job_due_date, machine_breakdown_events= load_instance(path, DDT)

    fjsp_data_back_up = copy.deepcopy(fjsp_data)

    current_time, current_job_num = 0, 0
    total_job_num = len(fjsp_data)
    current_op_id = [0 for _ in range(total_job_num)]
    start_time = [[0] for _ in range(total_job_num)]

    event_holder = []
    machine_occupied = collections.defaultdict(list)

    # New job arrival event
    for arrival_time in job_arrival_time:
        if arrival_time:
            if not event_holder or arrival_time != event_holder[-1][1]:
                event_holder.append(('new job', arrival_time))
        else:
            current_job_num += 1

    # Machine breakdown event
    event_holder.extend(('breakdown', breakdown[0], breakdown[1], m_id) for m_id in range(machine_num) for breakdown in machine_breakdown_events[m_id])
    
    # Last event
    event_holder.append(('end', MAX))
    event_holder = sorted(event_holder, key=time_stamp)

    # initial solution
    model = cp_model.CpModel()
    _round = 0

    while True:
        print("="*50 + str(_round) + "="*50)
        # 1. Schedule
        env = {
            "model" : model,
            "machine_occupied" : machine_occupied,
            "machine_num" : machine_num,
            "fjsp_data" : fjsp_data[:current_job_num],
            "job_arrival_time" : job_arrival_time[:current_job_num],
            "job_due_date" : job_due_date[:current_job_num],
            "start_time" : start_time[:current_job_num],
            "current_time" : current_time
        }

        result = solve(env, time_limit, num_thread, enable_sol_printer)
        result = sorted(result, key=st_time)

        f_d = copy.deepcopy(fjsp_data)
        c_o_i = copy.deepcopy(current_op_id)
        s_t = copy.deepcopy(start_time)
        m = copy.deepcopy(model)
        m_cp = copy.deepcopy(machine_occupied)

        for op_info in result:
            if op_info['job_id'] == -1: #breakdown
                continue
            # Record information
            record_information(copy.deepcopy(op_info), f_d, c_o_i, s_t, m, m_cp)
#output_json('./timeline/Base_mk04/postpone/seed_2011_newjob_Tarr=20_breakdown_Tbreak=[60, 80]/60/{}/{}_{}.json'.format(path.split('/')[-2], path.split('/')[-1][:-4], _round), m_cp)
        f_data = path.split('/')
#output_json('./timeline/{}/{}/60/{}/{}_{}.json'.format(f_data[2], f_data[3], f_data[-2], f_data[-1][:-4], _round), m_cp)
        output_json('{}/{}_{}.json'.format(timelinedir, f_data[-1][:-4], _round), m_cp)

        # 2. Event occur time
        current_time = event_holder[0][1]

        # 3. Add event & Cancel schedule
        for op_info in result:
            if op_info['job_id'] == -1: #breakdown
                continue
            if op_info['start_time'] < current_time: # Try to keep it if start before event occur 
                # Record information
                record_information(op_info, fjsp_data, current_op_id, start_time, model, machine_occupied)

        # 4. Add event
        while len(event_holder) and event_holder[0][1] == current_time:
            dynamic_event = event_holder.pop(0)

            if dynamic_event[0] == 'new job':
                while current_job_num < total_job_num and job_arrival_time[current_job_num] == current_time:
                    current_job_num += 1
                    
            elif dynamic_event[0] == 'breakdown':
                # CPmodel update 
                breakdown_start = dynamic_event[1]
                breakdown_duration = dynamic_event[2]
                machine_id = dynamic_event[3]
                breakdown = model.NewIntervalVar(breakdown_start, breakdown_duration, breakdown_start + breakdown_duration, 'breakdown')

                while len(machine_occupied[machine_id]) and (machine_occupied[machine_id][-1].Proto().start.offset <= breakdown_start and breakdown_start < machine_occupied[machine_id][-1].Proto().end.offset):
                    recover_op = machine_occupied[machine_id].pop()
                    recover_job_id, recover_op_id = map(int, recover_op.Name().split('_')[:2])
                    current_op_id[recover_job_id] -= 1
                    fjsp_data[recover_job_id].insert(0, fjsp_data_back_up[recover_job_id][current_op_id[recover_job_id]])
                    start_time[recover_job_id].pop()

                machine_occupied[machine_id].append(breakdown)
                machine_breakdown_events[machine_id].pop(0)

        # output_json('./result/test/{}.json'.format(_round), machine_occupied)

        # 5. END?
        if is_finish(fjsp_data, start_time, event_holder):
            break
        _round += 1
    # output_json('./result/test/{}/{}_after.json'.format(_round), machine_occupied)
    if objective == 'tardiness':
        return machine_occupied, get_tard(machine_num, machine_occupied, job_due_date)
    else:
        return machine_occupied, get_tardy_num(machine_num, machine_occupied, job_due_date) / float(total_job_num)
    
def get_tard(machine_num, machine_occupied, job_due_date):
    job_ends = [0 for _ in range(len(job_due_date))]
    for m_id in range(machine_num):
        for op_info in machine_occupied[m_id]:
            if op_info.Name() == 'breakdown':
                continue
            job_ends[int(op_info.Name().split('_')[0])] = max(job_ends[int(op_info.Name().split('_')[0])], op_info.Proto().end.offset)
    return sum([max(0, job_ends[i] - job_due_date[i]) for i in range(len(job_due_date))])
    
#def get_tardy_num_rate(self):
def get_tardy_num(machine_num, machine_occupied, job_due_date):
    tardy_jobs = 0
    job_ends = [0 for _ in range(len(job_due_date))]
    for m_id in range(machine_num):
        for op_info in machine_occupied[m_id]:
            if op_info.Name() == 'breakdown':
                continue
            job_ends[int(op_info.Name().split('_')[0])] = max(job_ends[int(op_info.Name().split('_')[0])], op_info.Proto().end.offset)
    for i in range(len(job_due_date)):
        if (job_ends[i] - job_due_date[i]) > 0:
            tardy_jobs += 1
    print(tardy_jobs)
    return tardy_jobs 

def solve(env, time_limit=60, num_thread=1, enable_sol_printer=False, objective='tardiness'):

    """Solve a small flexible jobshop problem."""
    # machine_num, fjsp_data, job_arrival_time, job_due_date, machine_breakdown_events= load_instance(path)
    all_model, all_machine_occupied = env["model"], env["machine_occupied"]
    machine_num, fjsp_data = env["machine_num"], env["fjsp_data"]
    job_arrival_time, job_due_date = env["job_arrival_time"], env["job_due_date"]
    start_time, current_time = env['start_time'], env["current_time"]
    num_jobs = len(fjsp_data)
    all_jobs = range(num_jobs)
    all_machines = range(machine_num)

    # Model the flexible jobshop problem.
    model = copy.deepcopy(all_model)
    # print(max([st[-1] for st in start_time]))

    horizon = max(max([st[-1] for st in start_time]), current_time)
    for job in fjsp_data:
        for task in job:
            max_task_duration = 0
            for alternative in task:
                max_task_duration = max(max_task_duration, alternative[0])
            horizon += max_task_duration
    horizon = horizon * 10
    # print('Horizon = %i' % horizon)

    # Global storage of variables.
    intervals_per_machine = copy.deepcopy(all_machine_occupied)
    # intervals_per_machine = collections.defaultdict(list)
    starts = {}  # indexed by (job_id, task_id).
    presences = {}  # indexed by (job_id, task_id, alt_id).
    job_ends = []
    tards = []

    # Scan the jobs and create the relevant variables and intervals.
    for job_id in all_jobs:
        job = fjsp_data[job_id]
        num_tasks = len(job)
        previous_end = None
        for op_id in range(num_tasks):
            task = job[op_id]

            min_duration = task[0][0]
            max_duration = task[0][0]

            num_alternatives = len(task)
            all_alternatives = range(num_alternatives)

            for alt_id in range(1, num_alternatives):
                alt_duration = task[alt_id][0]
                min_duration = min(min_duration, alt_duration)
                max_duration = max(max_duration, alt_duration)

            # Create main interval for the task.
            suffix_name = '_j%i_t%i' % (job_id, op_id)
            start = model.NewIntVar(0, horizon, 'start' + suffix_name)
            duration = model.NewIntVar(min_duration, max_duration,
                                       'duration' + suffix_name)
            end = model.NewIntVar(0, horizon, 'end' + suffix_name)
            interval = model.NewIntervalVar(start, duration, end,
                                            'interval' + suffix_name)

            # Store the start for the solution.
            starts[(job_id, op_id)] = start

            # Add precedence with previous task in the same job.
            if previous_end is not None:
                model.Add(start >= previous_end)
            else:
                model.Add(start >= max(current_time, start_time[job_id][-1])) 
            previous_end = end

            # Create alternative intervals.
            if num_alternatives > 1:
                l_presences = []
                for alt_id in all_alternatives:
                    alt_suffix = '_j%i_t%i_a%i' % (job_id, op_id, alt_id)
                    l_presence = model.NewBoolVar('presence' + alt_suffix)
                    l_start = model.NewIntVar(0, horizon, 'start' + alt_suffix)
                    l_duration = task[alt_id][0]
                    l_end = model.NewIntVar(0, horizon, 'end' + alt_suffix)
                    l_interval = model.NewOptionalIntervalVar(
                        l_start, l_duration, l_end, l_presence,
                        'interval' + alt_suffix)
                    l_presences.append(l_presence)

                    # Link the master variables with the local ones.
                    model.Add(start == l_start).OnlyEnforceIf(l_presence)
                    model.Add(duration == l_duration).OnlyEnforceIf(l_presence)
                    model.Add(end == l_end).OnlyEnforceIf(l_presence)

                    # Add the local interval to the right machine.
                    intervals_per_machine[task[alt_id][1]].append(l_interval)

                    # Store the presences for the solution.
                    presences[(job_id, op_id, alt_id)] = l_presence

                # Select exactly one presence variable.
                model.Add(sum(l_presences) == 1)
            else:
                intervals_per_machine[task[0][1]].append(interval)
                presences[(job_id, op_id, 0)] = model.NewConstant(1)

        job_ends.append(previous_end)

    for machine_id in all_machines:
        intervals = intervals_per_machine[machine_id]
        if len(intervals) > 1:
            model.AddNoOverlap(intervals)

    # objective
    for job_id in all_jobs:
        if job_ends[job_id] is None:
            continue
        if objective == 'tardiness':
            tard = model.NewIntVar(0, horizon, 'tard_{}'.format(job_id))
            model.AddMaxEquality(tard, [0, job_ends[job_id] - job_due_date[job_id]])
        else:
            tard = model.NewBoolVar('tard_{}'.format(job_id))
            model.Add(job_ends[job_id] > job_due_date[job_id]).OnlyEnforceIf(tard)
            model.Add(job_ends[job_id] <= job_due_date[job_id]).OnlyEnforceIf(tard.Not())
        tards.append(tard)
    model.Minimize(sum(tards))

    # print(model)

    # Solve model.
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = num_thread
    if enable_sol_printer:
        solution_printer = SolutionPrinter()
        status = solver.Solve(model, solution_printer)
    else:
        status = solver.Solve(model)


    makespan = 0.

    # Print final solution.
    result = []
    for m_id in all_machines:
        op_info = {
            'Order':        None,
            'job_id':       -1,
            'op_id':        0,
            'machine_id':   m_id,
            'start_time':   0,
            'process_time': 0,
            'finish_time':  0,
            'alternative':  None,
        }
        result.append(op_info)

    for job_id in all_jobs:
        for op_id in range(len(fjsp_data[job_id])):
            print(job_id, op_id)
            start_value = solver.Value(starts[(job_id, op_id)])
            machine = -1
            duration = -1
            selected = -1
            for alt_id in range(len(fjsp_data[job_id][op_id])):
                if solver.Value(presences[(job_id, op_id, alt_id)]):
                    duration = fjsp_data[job_id][op_id][alt_id][0]
                    machine = fjsp_data[job_id][op_id][alt_id][1]
                    selected = alt_id
            op_info = {
                'Order':        None,
                'job_id':       job_id,
                'op_id':        op_id,
                'machine_id':   machine,
                'start_time':   start_value,
                'process_time': duration,
                'finish_time':  start_value + duration,
                'alternative':  fjsp_data[job_id][op_id],
            }
            makespan = max(makespan, start_value + duration)
            result.append(op_info)

    print('Solve status: %s' % solver.StatusName(status))
    print('Optimal objective value: %i' % solver.ObjectiveValue())
    print('Statistics')
    print('  - conflicts : %i' % solver.NumConflicts())
    print('  - branches  : %i' % solver.NumBranches())
    print('  - wall time : %f s' % solver.WallTime())
    return result

if __name__ == '__main__':
#    test_dir = './datasets/DFJSP/MK_1215/' 
#    test_dir = './datasets/DFJSP/MK_1215or/C/MK0' 
#    test_dir = './datasets/DFJSP/MK_1215or/C/MK04' 
#    test_dir = './datasets/DFJSP/MK_1215or/C/MK05' 
#    test_dir = './datasets/DFJSP/MK_1215or/C/MK06' 
#    test_dir = './datasets/DFJSP/MK_1215or/C/MK07' 
#    test_dir = './datasets/DFJSP/MK_1215or/C/MK08' 
#    test_dir = './datasets/DFJSP/MK_1215or/C/MK09' 
#    test_dir = './datasets/DFJSP/MK_1215or/C/MK10' 
#    test_dir = './datasets/DFJSP/MK_1215or/C/MK04' 
#    test_dir = './datasets/DFJSP/MK_1215or/U/MK05' 
#    test_dir = './datasets/DFJSP/MK_1215or/U/MK06' 
#    test_dir = './datasets/DFJSP/MK_1215or/U/MK07' 
#    test_dir = './datasets/DFJSP/MK_1215or/U/MK08' 
#    test_dir = './datasets/DFJSP/MK_1215or/U/MK09' 
#    test_dir = './datasets/DFJSP/MK_1215or/U/MK10' 
#    test_dir = './datasets/DFJSP/MK_1215or/C/C3' 
#    test_dir = './datasets/DFJSP/MK_1215or/C/C5' 
#    test_dir = './datasets/DFJSP/MK_1215or/C/C7' 
    # gen_type = ['Random', 'Base_mk04']
#gen_type = ['Base_mk04']
#    gen_type = ['MK_1215']
#    gen_type = ['MK_1215or/C/MK04']
#    gen_type = ['MK_1215or/C/MK05']
#    gen_type = ['MK_1215or/C/MK06']
#    gen_type = ['MK_1215or/C/MK07']
#    gen_type = ['MK_1215or/C/MK08']
#    gen_type = ['MK_1215or/C/MK09']
#    gen_type = ['MK_1215or/C/MK10']
#    gen_type = ['MK_1215or/C/MK04']
#    gen_type = ['MK_1215or/U/MK05']
#    gen_type = ['MK_1215or/U/MK06']
#    gen_type = ['MK_1215or/U/MK07']
#    gen_type = ['MK_1215or/U/MK08']
#    gen_type = ['MK_1215or/U/MK09']
#    gen_type = ['MK_1215or/U/MK10']
#    gen_type = ['MK_1215or/C/C3']
#    gen_type = ['MK_1215or/C/C5']
#    gen_type = ['MK_1215or/C/C7']
    # event_type = ['breakdown', 'newjob', 'newjob_breakdown']
#event_type = [
#                    'seed_2011_newjob_Tarr=20_breakdown_Tbreak=[60, 80]',
                    # 'seed_8039_newjob_Tarr=20_breakdown_Tbreak=[40, 60]',
                    # 'seed_8914_newjob_Tarr=15_breakdown',
                    # 'seed_6404_newjob_Tarr=20_breakdown',
                    # 'seed_1468_newjob_Tarr=25_breakdown',
                    # 'seed_1855_newjob_Tarr=30_breakdown',
#                ]
#    test_sets = os.listdir(test_dir)
    data_dir = 'datasets/DFJSP'
    parser = argparse.ArgumentParser(description='Arguments for Or postpone')
    parser.add_argument('--targetdir', type=str, default='asd')
    parser.add_argument('--logname', type=str, default='bak')
    parser.add_argument('--timelimit', type=int, default=60)
    parser.add_argument('--DDT', type=float, default=2.0)
    parser.add_argument('--objective', type=str, default='tardiness')
    args = parser.parse_args()
#    test_dir = './datasets/DFJSP/MK_1215or/'+args.targetdir 
    test_dir = './datasets/DFJSP/'+args.targetdir 
    gen_type = [args.targetdir]
    event_type = os.listdir(test_dir)
    
#    time_limit = 300
    time_limit = args.timelimit
    num_thread = 5
    for g_type in gen_type:
        for e_type in event_type:
            dir_name = os.path.join(g_type, e_type)
#        dir_name = os.path.join(test_dir, e_type)
            for dir_size in os.listdir(os.path.join(data_dir, dir_name)):
#                result_dir = os.path.join('./result', dir_name, str(time_limit), 'thread5')
                result_dir = './result/{}_{}_{}_DDT{}'.format(g_type, e_type, args.logname, args.DDT)
                timeline_dir = os.path.join('./timeline', dir_name, str(time_limit),  dir_size)
                os.makedirs(result_dir, exist_ok=True)
                os.makedirs(timeline_dir, exist_ok=True)
                for file in os.listdir(os.path.join(data_dir, dir_name, dir_size)):
                    fn, ext = os.path.splitext(file)
                    if ext != '.fjs':
                        continue
                    result, tard = solver(os.path.join(data_dir, dir_name, dir_size, file), timeline_dir, time_limit=time_limit, num_thread=num_thread, DDT=args.DDT, objective=args.objective)
                    with open(os.path.join(result_dir, 'or-tools_result{}.txt'.format(args.logname)), 'a') as outfile:
                        outfile.write('{} : {}\n'.format(os.path.join(dir_name, dir_size, file), tard))
                        output_json('{}.json'.format(os.path.join(timeline_dir, file)[:-4]), result)

