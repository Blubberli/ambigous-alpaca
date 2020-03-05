#!/bin/bash

CONFIG=/home/neele/PycharmProjects/ambigous-alpaca/scripts/configs/config3_gerco_lstm.json
TRAIN=/home/neele/PycharmProjects/ambigous-alpaca/scripts/train.py

for i in `seq 2 2 8`;
 do echo "dropout is now 0.$i";
  json -I -f $CONFIG -e "this.model.dropout='0.$i'";
  python $TRAIN $CONFIG;
  done