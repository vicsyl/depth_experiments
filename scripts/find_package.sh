#!/bin/bash

pack="pycolmap"

for env in $(conda env list |  grep -o '^\S*'); do
  echo "Checking $env"
  conda list -n $env $pack &>/dev/null | grep $pack
done