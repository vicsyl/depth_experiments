#!/bin/bash

# NOTE this is PROBABLY how it was run last time
python -u depths_from_sfm.py --scenes all 2>&1 1>scripts/out_sfm.txt &

echo "tail -f follows, you can kill it"

sleep 2

tail -f out_err.txt

