#!/bin/bash

#SBATCH --job-name=single-prot
#SBATCH --nodes=3
#SBATCH --tasks-per-node=1
#SBATCH --output=slurm_%A.out
#SBATCH --error=slurm_%A.err

worker_num=2 # Must be one less that the total number of nodes

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

node1=${nodes_array[0]}

ip_prefix=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address) # Making address
suffix=':6379'
ip_head=$ip_prefix$suffix
redis_password=$(uuidgen)

export ip_head # Exporting for latter access by trainer.py
export redis_password

ulimit -n 65536 # increase limits to avoid to exceed redis connections

srun --nodes=1 --ntasks=1 -w $node1 ray start --block --head --port=6379 --redis-password=$redis_password & # Starting the head
sleep 5
# Make sure the head successfully starts before any worker does, otherwise
# the worker will not be able to connect to redis. In case of longer delay,
# adjust the sleeptime above to ensure proper order.

for ((  i=1; i<=$worker_num; i++ ))
do
  node2=${nodes_array[$i]}
  srun --nodes=1 --ntasks=1 -w $node2 ray start --block --address=$ip_head --redis-password=$redis_password & # Starting the workers
  # Flag --block will keep ray process alive on each compute node.
  sleep 5
done


name=('triplets_single_prot_baseline' 'triplets_single_prot' 'triplets_multi_prot')
path=('triplets_dataset_single_prot_baseline' 'triplets_dataset_single_prot' 'triplets_dataset_multi_prot')
baseline=('MLP' 'DotProd')
version=('' '_e2e')

grouped_stratification=false # Specify here if you want grouped stratification. Possible values: <true, false>
i=0 # Specify here which taask to run between those in name. Possible values <0, 1, 2>
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
export LD_PRELOAD=~/miniconda3/envs/covid-cpu/lib/libjemalloc.so

export LD_LIBRARY_PATH=~/miniconda3/envs/covid-cpu/lib

# Parallelization settings
export KMP_SETTING=granularity=fine,compact,1,0                                       ## affinity 1
#export KMP_SETTING=KMP_AFFINITY=noverbose,warnings,respect,granularity=core,none      ## affinity 2
export OMP_NUM_THREADS=2
export KMP_BLOCKTIME=50

echo MALLOC_CONF=$MALLOC_CONF
echo LD_PRELOAD=$LD_PRELOAD
echo OMP_THREADS=$OMP_NUM_THREADS
echo KMP_BLOCKTIME=$KMP_BLOCKTIME
echo KMP_SETTING=$KMP_SETTING

to_execute="python3 -u main.py --dataset-path $fpath --name $fname --k $k --csv $csv"
if  [[ $fname =~ "_e2e" ]]; then
        to_execute=$to_execute" --e2e"
fi
if  grouped_stratification; then
        to_execute=$to_execute" --tag"
fi
if  [[ $fname =~ "baseline" ]]; then
        to_execute=$to_execute" --baseline "${baseline[$b]}
fi
echo "START: "$to_execute
$to_execute
