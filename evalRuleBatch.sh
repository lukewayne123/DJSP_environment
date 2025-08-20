python testRule.py --logU True --rule SPT --date MK_SPT_reschedule1210c7 >> SPT_reschedule1210c7.txt
python testRule.py --logU True --rule SPT --date MK_SPT_postpone1210c7 --breakdown_handler postpone >> SPT_postpone1210c7.txt
python testRule.py --logU True --rule EDD --date MK_EDD_reschedule1210c7 >> EDD_reschedule1210c7.txt
python testRule.py --logU True --rule EDD --date MK_EDD_postpone1210c7 --breakdown_handler postpone >> EDD_postpone1210c7.txt
python testRule.py --logU True --rule FIFO --date MK_FIFO_reschedule1210c7 >> FIFO_reschedule1210c7.txt
python testRule.py --logU True --rule FIFO --date MK_FIFO_postpone1210c7 --breakdown_handler postpone >> FIFO_postpone1210c7.txt
#python testRule.py --logU True --rule $1 --date MK_$1_reschedule1207 >> $1_reschedule1207.txt
#python testRule.py --logU True --rule $1 --date MK_$1_postpone1207_check --breakdown_handler postpone >> $1_postpone1207check.txt
#python testRule.py --logU True --rule $1 --date MK_$1_postpone_check --breakdown_handler postpone >> $1_postponecheck.txt
