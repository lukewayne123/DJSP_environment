# DJSP environment

## Installation

Setup the virtual environment.
```c
podman run -it --name={YOUR_NAME}   -v $PWD/DJSP_environment:/DJSP_environment pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
```

Install required packages in the environment.
```c
pip install torch-geometric==2.3.1 ortools==9.11.4210 opencv-python plotly matplotlib gym tensorboard pandas colorhash
```

### Train RS-d in reschedule case
```c
python train.py --objective {tardiness/tardy_rate/makespan} --delete_node True --train_arr True --train_break True --date {YOUR_MODEL_NAME} --logU True --DDT {DDT} --device {YOUR_DEVICE}
```
For example, train RS-d for tardy_rate criteria
```c
python train.py --objective tardy_rate  --delete_node True --train_arr True --train_break True --date {Makespan_MODEL_NAME} --logU True --DDT {DDT} --device {YOUR_DEVICE} --rule EDD
```
Suggest baseline rule 
| Objective  | Rule |
| ------------- |:-------------:|
| tardy_rate      | EDD     |
| makespan     | MWKR    |
| tardiness      | EDD_SPT_rng     |

## Run RS-d validation in reschedule case
```c
python valid.py --objective {tardiness/tardy_rate/makespan} --delete_node True --train_arr True --train_break True --date {YOUR_MODEL_NAME} --logU True --DDT {DDT} --device {YOUR_DEVICE}
```
The default validation directory is `\datasets\DFJSP\Base_mk04\valid_seed_9569_newjob_Tarr=20_breakdown\(15+20)x8`

### Train ScN-d in reschedule case
```c
python train.py --objective {tardiness/tardy_rate/makespan} --train_arr True --train_break True --date {YOUR_MODEL_NAME} --logU True --DDT {DDT} --device {YOUR_DEVICE}
```
For example, train ScN-d for makespan criteria
```c
python train.py --objective makespan --train_arr True --train_break True --date {Makespan_MODEL_NAME} --logU True --DDT {DDT} --device {YOUR_DEVICE} --rule MWKR
```

## Run ScN-d validation in reschedule case
```c
python valid.py --objective {tardiness/tardy_rate/makespan} --train_arr True --train_break True --date {YOUR_MODEL_NAME} --logU True --DDT {DDT} --device {YOUR_DEVICE}
```

### Test 
Run the corresponding shell script for edata, rdata and vdata
```c
sh {TARGET_SCRIPT}.sh
```
For example, test RS-d
```c
sh testRS-dbatch.sh {YOUR_MODEL_NAME} {TEST_DATASET} {DDT}
```
For example, test Rule EDD
```c
sh testLaRule.sh EDD {TEST_DATASET} {LOG_DIRECTORY} {DDT}
```

### Run OR-Tools in reschedule case
```c
python ortool_60reschedule.py --targetdir {FILE_PATH} --DDT {2.0/3.0/10.0}
```


## Important hyperparameters list
| Parameters  | Description | Default value |
| ------------- |-------------| -------------|
| device      | Your CPU/GPU device     | cuda:0 | 
| logU     | Log information for debug    | False |
| ini_job_num      | Initial job number for instance | 15 |
| machine_num      | Machine number for instance | 8 |
| max_process_time | Maximum process time for operation | 10 |
| delete_node | Flag for RS-d (True) or ScN-d (False) | False |
| train_arr | Flag for job arrival events | False |
| train_break |  Flag for machine breakdown events | False |
| DDT | Due-date tardiness value | 3.0 |
| new_job_event | The number of new job arrival event | 5 |
| new_job_per_num | The number of arrival job in each new job arrival event | 2 |
| arrival_time_dist | Arrival time distribution between two job | 20 |
| MTBF | Mean time between failure, T~exp(M_mtbf) | [50, 70] |
| MTTR | Mean time to repair, T~exp(M_mttr) | [10, 20] |
| breakdown_handler | Flag for reschedule / postpone mechanism | reschedule |
| objective | Tardiness/Tardy_rate/Makespan criteria | tardiness |
| hidden_dim | Hidden dimension for model   | 256 |
| GNN_num_layers | Number of layers for GNN | 3 |
| policy_num_layers | Number of layers for policy network | 2 |