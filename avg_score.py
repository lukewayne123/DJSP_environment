import os
import numpy as np


def calculate_average_tardiness(dir_path, file_path, output_file):
    
    tardiness_file_path = os.path.join(dir_path, file_path, 'test_result.txt')
    
    with open(tardiness_file_path, 'r') as f:
        count = 0
        tardiness_values = []
        with open(output_file, 'w') as of:
            for line in f:

                if line[0] == '-':
                    of.write(line)
                    continue
                
                # if line.strip():
                #     tardiness_value = int(line.split(':')[-1].strip())
                #     tardiness_values.append(tardiness_value)
                count += 1
        
                if count == 11:
                    average_tardiness = int(line.split(':')[-1].strip())
                    # print(line.split(':')[-2].split('/'))
                    size = line.split(':')[-2].split('\\')[-1].split()[0][:-2]
                    of.write(f'Size Group: {size}, Average Tardness: {average_tardiness}\n')
                    count = 0
                    tardiness_values = []
                
            

dir_path = "./result/"

files_path = [
                '5e2j_small',
                '5e2j_small_postpone'
]


avg_file_dir = './result/avg_tardness'
os.makedirs(avg_file_dir, exist_ok=True)

for file_path in files_path:
    output_file = os.path.join(avg_file_dir, file_path)
    average_tardiness_per_size = calculate_average_tardiness(dir_path, file_path, output_file)

    
