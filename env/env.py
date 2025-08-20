import gym
import copy
from env.utils.instance import JSP_Instance
from env.utils.mach_job_op import *
from env.utils.graph import Graph
from djsp_logger import DJSP_Logger

class JSP_Env(gym.Env):
    def __init__(self, args):
        self.args = args
        self.jsp_instance = JSP_Instance(args)
        self.logger = DJSP_Logger()

    def step(self, step_op):
        self.jsp_instance.assign(step_op)
        avai_ops = self.jsp_instance.current_avai_ops()
        return avai_ops, self.done()
    
    def reset(self):
        self.jsp_instance.reset()
        return self.jsp_instance.current_avai_ops()
       
    def done(self):
        return self.jsp_instance.done()
    
    def get_processtime(self):
#d={}
#idx=0
        d=[]
        for j in self.jsp_instance.jobs:
#d[f'j_{idx}']=[]
            for o in j.operations:
                d.append(o.expected_process_time)
#d[f'j_{idx}'].append(o.expected_process_time)
#idx+=1
        return d

    def get_duedate(self):
        return [j.due_date for j in self.jsp_instance.jobs]

    def get_tardiness(self):
        return int(sum([max(0, j.operations[-1].finish_time - j.due_date) for j in self.jsp_instance.jobs]))
    
    def get_tardy_num_rate(self):
        tardy_jobs = 0
        for j in self.jsp_instance.jobs:
            if (j.operations[-1].finish_time - j.due_date) > 0:
                tardy_jobs += 1

        return tardy_jobs, tardy_jobs / len(self.jsp_instance.jobs)
    
    def get_makespan(self):
        return max(m.avai_time() for m in self.jsp_instance.machines)   

    def get_individual_utilization(self):
        return self.jsp_instance.total_job_process_time / sum(m.avai_time() for m in self.jsp_instance.machines)


    def get_system_utilization(self):
        return self.jsp_instance.total_job_process_time / (max(m.avai_time() for m in self.jsp_instance.machines) * len(self.jsp_instance.machines))

    def get_total_process_time(self):
        total_process_time = 0

        job_process_time = {}

        for m in self.jsp_instance.machines:
            for op_his in m.processed_op_history:
              job_id = op_his["job_id"]
#    job_id = 'break'
#                continue
              if op_his["job_id"] != -1:
                  total_process_time += op_his["process_time"]
              if job_id in job_process_time:
                  job_process_time[job_id] += op_his["process_time"]
              else:
                  job_process_time[job_id] = op_his["process_time"]
#print(op_his["job_id"], op_his["process_time"])
#print("-"*8)
#print(self.jsp_instance.total_job_process_time)
#for key, value in sorted(self.jsp_instance.job_process_time.items(), key=lambda x: x[0]):
#   print("{} : {}".format(key, value))
#print(total_process_time - self.jsp_instance.total_repair_time, self.jsp_instance.total_repair_time)
#print(total_process_time, self.jsp_instance.total_repair_time)
#for key, value in sorted(job_process_time.items(), key=lambda x: x[0]):
#    print("{} : {}".format(key, value))
#print(self.jsp_instance.job_process_time)
        return total_process_time - self.jsp_instance.total_postpone_repair_time


    def get_graph_data(self):
        return self.jsp_instance.get_graph_data()
        
    def load_instance(self, filename, block_breakdown=False, start_offset=0.0):
        self.jsp_instance.load_instance(filename, block_breakdown, start_offset)
        return self.jsp_instance.current_avai_ops()

    def get_last_arrival(self):
        if len(self.jsp_instance.unarr_jobs) == 0:
            return 0
        else:
            return self.jsp_instance.unarr_jobs[-1].arrival_time

    def log(self, path):
        self.logger.reset()
        for m in self.jsp_instance.machines:
            for op_his in m.processed_op_history:

                op = Operation(None, op_his["job_id"], None, 0)
                op.op_id, op.selected_machine_id = op_his["op_id"], m.machine_id
                op.start_time, op.process_time, op.finish_time = op_his["start_time"], op_his["process_time"], op_his["start_time"] + op_his["process_time"]
                self.logger.add_op(op)

        self.logger.save(path)

