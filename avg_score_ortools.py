import os
import numpy as np
import csv
import argparse

def calculate_average_tardy_rate(dir_path, file_path):
    size_groups = {}
    
    size_group_dir = os.path.join(dir_path, file_path)
    
#tardy_rate_file_path = os.path.join(size_group_dir, '300', f'thread{threads_num}', 'or-tools_result.txt')
#    tardy_rate_file_path = os.path.join(`size_group_dir, '60', f'thread{threads_num}', targefile)
    targetfile = sorted(os.listdir(size_group_dir))[0]
    tardy_rate_file_path = os.path.join(size_group_dir, targetfile)
    print(tardy_rate_file_path)
    if not os.path.isfile(tardy_rate_file_path):
        return size_groups, -1
    
    with open(tardy_rate_file_path, 'r') as f:
        count = 0
        tardy_rate_values = []
        total_tardy_rate_values = []
        for line in f:
            
            if line.strip():
#                tardy_rate_value = int(line.split(':')[-1].strip())
                tardy_rate_value = float(line.split(':')[-1].strip())
                tardy_rate_values.append(tardy_rate_value)
                total_tardy_rate_values.append(tardy_rate_value)
                count += 1
    
            if count == 10:
                average_tardy_rate = np.mean(tardy_rate_values)
#print(line.split(':')[-2].split('/'))
                size = line.split(':')[-2].split('/')[-2]
                print(size)
                size_groups[size] = average_tardy_rate
                count = 0
                tardy_rate_values = []

    
    return size_groups, np.mean(total_tardy_rate_values)

parser = argparse.ArgumentParser(description='Arguments for Avg')
parser.add_argument('--targetdir', type=str, default='asd')
parser.add_argument('--outputfile', type=str, default='avg_result.csv')
parser.add_argument('--targetfile', type=str, default='or-tools_result.txt')
args = parser.parse_args()
#dir_path = "./MK_1215or/"+args.targetdir
cases = ['NoBreak', 'Break']

for case in cases:

    dir_path = os.path.join(args.targetdir, case)
    files_path = os.listdir(dir_path)
    
    for file_path in files_path:
            
        average_tardy_rate_per_size, avg_total_tardy_rate = calculate_average_tardy_rate(dir_path, file_path)
        
    #    output_file = os.path.join(avg_file_dir, f'thread{threads_num}', '60', file_path)
        output_file = os.path.join(args.targetdir, file_path[-7:]+args.outputfile)
    #    with open(output_file, 'w') as f:
    #    print(count, count//40)
    #    print(file_path)
        if avg_total_tardy_rate < 0:
            print("skip"+file_path)
            continue
        with open(output_file,"a") as csvfile:
            writer = csv.writer(csvfile)
            
    #        writer.writerow(['DDT={} instance'.format(DDTs[count//40])])
            writer.writerow(['instance'])
            for size_group, avg_tardy_rate in average_tardy_rate_per_size.items():
    #            f.write(f'Size Group: {size_group}, Average Tardy rate: {avg_tardy_rate}\n')
                  writer.writerow([file_path+size_group, avg_tardy_rate])
    #        f.write(f'Total Average Tardy rate: {avg_total_tardy_rate}\n')
    #    count+=1
