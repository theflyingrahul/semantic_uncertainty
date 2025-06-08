#!/bin/bash

for j in `seq 29775 29803` ; do
    scancel $j
    echo $j
done