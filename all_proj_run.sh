#!/bin/bash

##source ~/start-pyenv
##cd ~/PhD-ASP-Methods/
source ./env/bin/activate
cd ./scripts/
chmod +x da_proj.r

# params

metapath=../metadata/tsp.csv
outpath=../Results/tsp_all/
lab=tsp

# Preprocess
python proj_set.py $metapath $outpath $lab 1 -feat all

# Run DA
if test -f "${outpath}status.txt"; then
    rm -r "${outpath}status.txt"
    ./da_par.r $outpath 1
else
    echo "previous step failed"
fi

# make projections
if test -f "${outpath}status.txt"; then
    rm -r "${outpath}status.txt"
    python proj_set.py $metapath $outpath $lab 2 
else
    echo "previous step failed"
fi

# make sparse projections
if test -f "${outpath}status.txt"; then
    rm -r "${outpath}status.txt"
    ./sparse_par.r $outpath
else
    echo "previous step failed"
fi
# make and eval predictions
if test -f "${outpath}status.txt"; then
    rm -r "${outpath}status.txt"
    # python proj_set.py $metapath $outpath $lab 3 
    python proj_set.py $metapath $outpath $lab 5 
else
    echo "previous step failed"
fi





