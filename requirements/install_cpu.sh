export myname=covid-cpu
conda create --name $myname python=3.7 requests networkx
conda activate covid-cpu
conda install rdkit -c rdkit
conda install -c conda-forge jemalloc


~/miniconda3/envs/$myname/bin/pip install torch==1.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
~/miniconda3/envs/$myname/bin/pip install -r requirements_cpu.txt --find-links https://pytorch-geometric.com/whl/torch-1.4.0.html

