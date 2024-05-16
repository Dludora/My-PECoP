#!/bin/bash

# Add your commands here

#!/bin/bash 
# for i in {2..6}
# do
#     python train.py --benchmark Seven --subset train --exp_name rank8_3_12345 --class_idx $i --gpu 0
#     sleep 1
# done

python train.py --benchmark MTL_Pace --subset train --exp_name pace_rank8_3_12345 --gpu 2,3