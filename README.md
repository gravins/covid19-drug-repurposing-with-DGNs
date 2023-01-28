# Deep Graph Networks for Drug Repurposing with Multi-Protein Targets (with COVID-19 use case)
Official code repository for our paper ***Deep Graph Networks for Drug Repurposing with Multi-Protein Targets***.

If you find our work useful for your research, please consider citing the following paper:

	@article{gravina2023DrugRep,
		 author = {Bacciu, Davide and Errica, Federico and Gravina, Alessio and Madeddu, Lorenzo and Podda, Marco and Stilo, Giovanni},
		 title = {Deep Graph Networks for Drug Repurposing with Multi-Protein Targets},
		 journal = {IEEE Transactions on Emerging Topics in Computing},
		 year = {2023},
		 volume={}
		 number={},
		 pages={1-14},
		 doi={10.1109/TETC.2023.3238963}
	}


## Installation
We provide a script to install the environment. You will need the conda package manager, which can be installed from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

To install the required packages (tested on a linux terminal):

- clone the repository
    - `git clone https://github.com/gravins/covid19-drug-repurposing-with-DGNs.git`

- cd into the cloned directory
    - `cd covid19-drug-repurposing-with-DGNs`

- run the install script
    - `./requirements/install_cpu.sh`

The script will create a virtual environment named `covid-cpu`, with all the required packages needed to run our code.

## Data download
The zipped dataset folder can be downloaded from [here](https://www.dropbox.com/s/685d7h2q8facao3/dataset.zip?dl=0).

## Run the experiment
_Note: To run the experiment is fundamental to define the task and model in the file run.sh or run_cluster.sh files._

- if you run the experiment on a standard machine:
	- `./run.sh`

- if you run the experiment on cluster managed by slurm:
	- `./run_cluster.sh`

- if you want to run the GraphDTA or DeepDTA baseline then
	- `cd DeepPurpose_baselines`
	- `python3 -u main.py`
	- For more details launch ```python3 main.py --help```

## Troubleshooting
If you get errors like `/lib64/libstdc++.so.6: version 'GLIBCXX_3.4.21' not found`:

- `conda install -c conda-forge c-compiler cxx-compiler` 
- `echo $LD_LIBRARY_PATH` should contain `:[path to your anaconda or miniconda folder name]/lib`
