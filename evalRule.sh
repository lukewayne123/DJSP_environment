python testRule.py --logU True --rule $1 --date MK_$1_reschedule1213c3 >> $1_reschedule1213c3.txt
python testRule.py --logU True --rule $1 --date MK_$1_postpone1213c3 --breakdown_handler postpone >> $1_postpone1213c3.txt
#python testRule.py --logU True --rule $1 --date MK_$1_reschedule1210 >> $1_reschedule1210.txt
#python testRule.py --logU True --rule $1 --date MK_$1_postpone1210 --breakdown_handler postpone >> $1_postpone1210.txt
#python testRule.py --logU True --rule $1 --date MK_$1_reschedule1207 >> $1_reschedule1207.txt
#python testRule.py --logU True --rule $1 --date MK_$1_postpone1207_check --breakdown_handler postpone >> $1_postpone1207check.txt
#python testRule.py --logU True --rule $1 --date MK_$1_postpone_check --breakdown_handler postpone >> $1_postponecheck.txt
