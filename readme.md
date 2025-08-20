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
## Run training
Run RS training procedure. (There are some parameters for ablation studying.
```c
python train.py --date=train --instance_type=FJSP --data_size=10 --delete_node=true
```

## Reproduced the result in paper
Follow the example to run a FJSP testing 
```c
python3 test.py --date=test --instance_type=FJSP --delete_node=true --test_dir='./datasets/FJSP/Brandimarte_Data' --load_weight='./weight/RS_FJSP/best'
```
Follow the example to run a FJSP testing (RS+op)
```c
python3 test.py --date=test --instance_type=FJSP --test_dir='./datasets/FJSP/Brandimarte_Data' --load_weight='./weight/RS+op_FJSP/best'
```

## Run OR-Tools reschedule case
```c
python ortool_60reschedule.py --targetdir {FILE_PATH} --DDT {2.0/3.0/10.0}
```

### Run ScN reschedule case
For makespan cre
```c
python ScN-dmakespan.py --train_arr True --train_break True --device cuda:0 --date LAe_RSwAwB_makespan_DDT3_0611 --logU True --test_dir edataMakespan --DDT 3.0
```
```c
python tardy_train.py --rule EDD --train_arr True --train_break Ture --DDT 3.0 --logU True --date 0203_ScNwAwBEDDddt3_tardy --device cuda:0        
```
### Similarly, for JSP
Follow the example to run a JSP testing (RS)
```c
python3 test.py --date=test --instance_type=JSP --delete_node=true --test_dir='./datasets/JSP/public_benchmark/ta' --load_weight='./weight/RS_JSP/best'
```
Follow the example to run a JSP testing (RS+op)
```c
python3 test.py --date=test --instance_type=JSP --test_dir='./datasets/JSP/public_benchmark/ta' --load_weight='./weight/RS+op_JSP/best'
```

## Hyperparameters list
```c
    python3 train.py \
    --device='cuda' \
    --instance_type='FJSP' \
    --data_size=10 \
    --max_process_time=100 \
    --delete_node=False \
    --entropy_coef=1e-2 \
    --episode=300001 \
    --lr=1e-4 \
    --step_size=1000 \
    --hidden_dim=256 \
    --GNN_num_layers=3 \
    --policy_num_layers=2 \
    --date='Dummy' \
    --detail=None \
    --test_dir='./datasets/FJSP/Brandimarte_Data' \
    --load_weight='./weight/RS_FJSP/best'
```
