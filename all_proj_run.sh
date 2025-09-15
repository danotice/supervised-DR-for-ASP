#!/bin/bash

##source ~/start-pyenv
# source ./env/bin/activate
# cd ./scripts/
chmod +x *.r

# params

metapath=../metadata/tsp.csv
outpath=../results/tsp_all/

# Preprocess
python proj_set.py $metapath $outpath 1 -feat all

# Run DA
if test -f "${outpath}status.txt"; then
    rm -r "${outpath}status.txt"
    ./da_par.r $outpath 1 #cv
else
    echo "failed to preprocess"
fi

# make projections
if test -f "${outpath}status.txt"; then
    rm -r "${outpath}status.txt"
    python proj_set.py $metapath $outpath 2 
else
    echo "previous step - DA"
fi

## make sparse projections
if test -f "${outpath}status.txt"; then
    rm -r "${outpath}status.txt"
    ./sparse_par.r $outpath 1 #cv
else
    echo "previous step failed - projections"
fi

# add sparse projections
if test -f "${outpath}status.txt"; then
    rm -r "${outpath}status.txt"
    python proj_set.py $metapath $outpath 6 -rproj sda_proj.csv 
    python proj_set.py $metapath $outpath 6 -rproj spls_proj.csv
else
    echo "previous step failed - sparse proj"
fi

## make and eval predictions
if test -f "${outpath}status.txt"; then
    rm -r "${outpath}status.txt"
    # python proj_set.py $metapath $outpath 3 
    python proj_set.py $metapath $outpath 5 -params default
else
    echo "previous step failed - add sparse proj"
fi





