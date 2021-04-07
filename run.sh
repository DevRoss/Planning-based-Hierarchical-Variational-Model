#!/bin/bash

CUR_DIR="$( cd `dirname $0`; pwd )"
cd ${CUR_DIR}
alias python='/home/rossliang/miniconda/bin/python'

LOG=${OUTPUT}/train.log
export PYTHONUNBUFFERED="True"
export PYTHONPATH="${PYTHONPATH}:${CUR_DIR}"

python main.py --train True --restore True --cuda_visible_devices 0,1
