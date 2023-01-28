#!/bin/bash

name=('triplets_single_prot_baseline' 'triplets_single_prot' 'triplets_multi_prot')
path=('triplets_dataset_single_prot_baseline' 'triplets_dataset_single_prot' 'triplets_dataset_multi_prot')
baseline=('MLP' 'DotProd')
version=('' '_e2e')

grouped_stratification=false # Specify here if you want grouped stratification. Possible values: <true, false>
i=0 # Specify here which task to run between those in name. Possible values <0, 1, 2>
v=0 # Specify here which model to run between those in version. Possible values in <0, 1>
b=1 # Specify here which baseline to run between those in baseline. Possible values in <0, 1>

# Check values
if (($i>2)) || (($v>1)) || (($b>1)) || (($i<0)) || (($v<0))|| (($b<0)); then
        echo "Values i and/or v and/or b out of bounds"
        exit 1
fi

if $grouped_stratification && (($i < 2)); then
        echo "CANNOT perform grouped stratification in single protein setting"
        exit 1
fi

# Apply group_strat label
if $grouped_stratification && (($i > 1)); then
        fname=${name[$i]}${version[$v]}_group_strat
else
        fname=${name[$i]}${version[$v]}
fi

fpath=dataset/${path[$i]}${version[$v]}.p
k=5
csv=${fname}_mod-sel_res.csv

# Jemalloc
export MALLOC_CONF=oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000
export LD_PRELOAD=~/miniconda3/lib/libjemalloc.so

export LD_LIBRARY_PATH=~/miniconda3/envs/covid-cpu/lib

# Parallelization settings
#export KMP_SETTING=granularity=fine,compact,1,0                                       ## affinity 1
#export KMP_SETTING=KMP_AFFINITY=noverbose,warnings,respect,granularity=core,none      ## affinity 2
export OMP_NUM_THREADS=2
export KMP_BLOCKTIME=50

echo MALLOC_CONF=$MALLOC_CONF
echo LD_PRELOAD=$LD_PRELOAD
echo OMP_THREADS=$OMP_NUM_THREADS
echo KMP_BLOCKTIME=$KMP_BLOCKTIME
echo KMP_SETTING=$KMP_SETTING

to_execute="nohup python3 -u main.py --dataset-path $fpath --name $fname --k $k --csv $csv"
if  [[ $fname =~ "_e2e" ]]; then
        to_execute=$to_execute" --e2e"
fi
if  grouped_stratification; then
        to_execute=$to_execute" --tag"
fi
if  [[ $fname =~ "baseline" ]]; then
        to_execute=$to_execute" --baseline "${baseline[$b]}
fi
to_execute=$to_execute" >out 2>err &"

echo "START: "$to_execute
$to_execute
