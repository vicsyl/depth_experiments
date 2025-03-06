#!/bin/bash

ts=`date +%Y_%m_%d_%H_%M_%S_%N`
log_file=./output/logs/ransac_${ts}.txt

python -u pose_estimation_from_sfm.py 2>&1 1>${log_file} &

echo "tail -f follows, you can kill it"

sleep 2
tail -f ${log_file}
