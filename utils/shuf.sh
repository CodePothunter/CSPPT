#!/bin/bash

input=$1
output=$1
random_seed=$2
awk '{if($0==""){print line;line="";}else{if(line!="")line=line";"$0;else line=$0;}}END{if(line!="")print line;}' $input > $input.tmp
#shuf --random-source=.randfile -o $input.tmp $input.tmp
python utils/shuffle_list.py --seed $random_seed $input.tmp > $input.tmp2
awk -F ";" '{for(i=1;i<=NF;i++){print $i;}print "";}' $input.tmp2 > $output
