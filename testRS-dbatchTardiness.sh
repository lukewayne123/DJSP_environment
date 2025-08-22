python RS-dutilizationTardiness.py --delete_node True --train_arr True --train_break True --device cuda:0 --date $1 --logU True --test_dir edata$2 --DDT $3
python RS-dutilizationTardiness.py --delete_node True --train_arr True --train_break True --device cuda:0 --date $1 --logU True --test_dir rdata$2 --DDT $3
python RS-dutilizationTardiness.py --delete_node True --train_arr True --train_break True --device cuda:0 --date $1 --logU True --test_dir vdata$2 --DDT $3
