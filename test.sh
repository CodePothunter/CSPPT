#!/bin/bash

#Just a test! refer:/slfs6/users/sz128/workspace/SLU_torch/SLU_torch_mb_fast

mb=10        # minibatch for training in parallel
lr=0.05      # learning rate, e.g. 0.0627142536696559
es=100       # word embedding size
proto="100"       # hidden layer prototype, e.g. 100-200-300
wl=0         # left context window of the current word
wr=4         # right context window of the current word
me=50     # max epoch

echo "$0 $@"
. CSPPT/utils/parse_options.sh || exit 1
#xx=$1

expdir=CSPPT/test #_STDbias
[ ! -d $expdir ] && mkdir -p $expdir

# echo Please input the text:
# python test/convert2list.py 2> $expdir/error_test.txt
#cp test/tmp/input.list $expdir/input_test

    # -vocab $datadir/atis/train -outlabel $datadir/atis/idx2la -print_vocab $expdir/vocab \ 
    
/home/slhome/qzx02/torch/install/bin/luajit CSPPT/main.lua -test $expdir/tmp/input.list \
  -vocab $expdir/vocab -outlabel CSPPT/idx2la -print_vocab $expdir/tmp/vocab \
  -test_only 1\
  -rnn_type lstm \
  -deviceId 1\
  -read_model $expdir/models/formal_cpu.rnn \
  -max_epoch $me \
  -hidden_prototype $proto -emb_size $es \
  -word_win_left $wl -word_win_right $wr \
  -batch_size $mb -bptt 9 \
  -alpha_decay 0.6 -alpha $lr \
  -init_weight 0.2 -random_seed 345 \
  # 1> $expdir/log_test.txt 2> $expdir/error_test.txt
  #| tee $expdir/log_test.txt 

# echo Your result:
# python test/output.py
