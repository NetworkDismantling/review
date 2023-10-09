# GDM: Graph Dismantling with Machine learning

This repository contains the scripts and data from the "Machine learning dismantling and early-warning signals of
disintegration in complex systems" paper by M. Grassia, M. De Domenico and G. Mangioni, available
at [Nature Communications](https://rdcu.be/cwqp3).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5105912.svg)](https://doi.org/10.5281/zenodo.5105912)

## How to cite the paper

> Grassia, M., De Domenico, M. & Mangioni, G. Machine learning dismantling and early-warning signals of disintegration
> in complex systems. Nat Commun 12, 5190 (2021). https://doi.org/10.1038/s41467-021-25485-8

BibTex:

```bib
@article{grassia2021machine,
  author   = {Grassia, Marco
and De Domenico, Manlio
and Mangioni, Giuseppe},
  title    = {Machine learning dismantling and early-warning signals of disintegration in complex systems},
  journal  = {Nature Communications},
  year     = {2021},
  month    = {Aug},
  day      = {31},
  volume   = {12},
  number   = {1},
  pages    = {5190},
  abstract = {From physics to engineering, biology and social science, natural and artificial systems are characterized by interconnected topologies whose features -- e.g., heterogeneous connectivity, mesoscale organization, hierarchy -- affect their robustness to external perturbations, such as targeted attacks to their units. Identifying the minimal set of units to attack to disintegrate a complex network, i.e. network dismantling, is a computationally challenging (NP-hard) problem which is usually attacked with heuristics. Here, we show that a machine trained to dismantle relatively small systems is able to identify higher-order topological patterns, allowing to disintegrate large-scale social, infrastructural and technological networks more efficiently than human-based heuristics. Remarkably, the machine assesses the probability that next attacks will disintegrate the system, providing a quantitative method to quantify systemic risk and detect early-warning signals of system's collapse. This demonstrates that machine-assisted analysis can be effectively used for policy and decision-making to better quantify the fragility of complex systems and their response to shocks.},
  issn     = {2041-1723},
  doi      = {10.1038/s41467-021-25485-8},
  url      = {https://doi.org/10.1038/s41467-021-25485-8}
}
```

RIS:

```ris
TY  - JOUR
AU  - Grassia, Marco
AU  - De Domenico, Manlio
AU  - Mangioni, Giuseppe
PY  - 2021
DA  - 2021/08/31
TI  - Machine learning dismantling and early-warning signals of disintegration in complex systems
JO  - Nature Communications
SP  - 5190
VL  - 12
IS  - 1
AB  - From physics to engineering, biology and social science, natural and artificial systems are characterized by interconnected topologies whose features – e.g., heterogeneous connectivity, mesoscale organization, hierarchy – affect their robustness to external perturbations, such as targeted attacks to their units. Identifying the minimal set of units to attack to disintegrate a complex network, i.e. network dismantling, is a computationally challenging (NP-hard) problem which is usually attacked with heuristics. Here, we show that a machine trained to dismantle relatively small systems is able to identify higher-order topological patterns, allowing to disintegrate large-scale social, infrastructural and technological networks more efficiently than human-based heuristics. Remarkably, the machine assesses the probability that next attacks will disintegrate the system, providing a quantitative method to quantify systemic risk and detect early-warning signals of system’s collapse. This demonstrates that machine-assisted analysis can be effectively used for policy and decision-making to better quantify the fragility of complex systems and their response to shocks.
SN  - 2041-1723
UR  - https://doi.org/10.1038/s41467-021-25485-8
DO  - 10.1038/s41467-021-25485-8
ID  - Grassia2021
ER  - 
```

## How to cite this repository

> Marco Grassia, Manlio De Domenico, & Giuseppe Mangioni. (2021, July 19). Machine learning dismantling and
> early-warning signals of disintegration in complex systems (Version 1.0). Nature Communications.
> Zenodo. http://doi.org/10.5281/zenodo.5105912

```
@software{marco_grassia_2021_5105912,
  author       = {Grassia, Marco and
                  De Domenico, Manlio and
                  Mangioni, Giuseppe},
  title        = {{Machine learning dismantling and early-warning 
                   signals of disintegration in complex systems}},
  month        = jul,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.5105912},
  url          = {https://doi.org/10.5281/zenodo.5105912}
}
```

## Setup

> Expected time for this step: 30m

Our test environment is hosted on a machine running Ubuntu 16.04 and equipped with a nVidia K80 (two cores with 12GB
VRAM each).
We use nVidia driver version 410.104 and CUDA version 10.0.

While suggested to improve prediction performance, a GPU is not needed as predicting values on a CPU is still feasible
in very reasonable time (e.g., in our environment _hyves_ network takes just 3m on CPU vs 1m on the GPU).

After driver and CUDA installation, run the following to check wether they are installed correctly.

```
# GPU drivers and CUDA version:
nvidia-smi
```

The python dependencies for this package can be installed with [
_conda_](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) using:

```
cd GDM
conda env create -f environment.yml
```

Unfortunately, _conda_ wouldn't let us recreate our exact environment due to some weird behavior with "conflicting
packages".
If you wish to check our full environment, check the environment_full.yml file.

The command above should create a new _conda_ environment ("gdm") that can be activated with

```bash
conda activate gdm
```

After activating the environment, check whether everything is installed correctly as follows:

```bash
# Check if pytorch is installed
python -c "import torch; print(torch.__version__)"

# Check if cuda is recognized by pytorch
python -c "import torch; print(torch.cuda.is_available())"

# Check which CUDA version pytorch is using
python -c "import torch; print(torch.version.cuda)"
``` 

We refer to PyTorch
Geometric's [full documentation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for any
issue during the installation phase or for OS specific steps.

**Please note that newer versions of PyTorch Geometric
introduce [changes](https://github.com/rusty1s/pytorch_geometric/commit/6fe33a2ab50f152e0756554e99c75a501c910ce0#diff-82ea677a2863fbf09ce4b99505c168e4cd69829d4e17e559d6737eb36d94f234L90,)
to the GAT layers that break compatibility with our models, even though no error is shown.**
More information can be found [here](https://github.com/rusty1s/pytorch_geometric/issues/1755#issuecomment-784990887).
***To reproduce our results, please stick with version v1.1.2.***

Before installing our package, ensure that you have g++ installed in order to compile our external dismantler. In our
environment, we tested using g++-9.

Lastly, install our package with pip:

```bash
cd GDM
pip install -e . -vv
```

The dismantler build command is run during this step, so no further action should be required.
If you get any compilation error message, you can manually check if the dependencies for
the [Tessil's robin-map](https://github.com/Tessil/robin-map.git) are met by running the official test scripts as
follows.

```bash
git clone https://github.com/Tessil/robin-map.git
cd robin-map
git reset --hard 5c6b2e5c55239999f989e996bcede0e1f46056f7
cd tests
mkdir build
cd build
cmake ..
cmake --build .
./tsl_robin_map_tests
```

The robin-map is a data-structure our dismantler depends on and we redistribute it with our code. If you wish to check
them out, keep in mind that in our tests we use revision _5c6b2e5c55239999f989e996bcede0e1f46056f7_.
For any issue with the robin-map we refer to the official GitHub page linked above.

To manually compile the dismantler (e.g., build failed during the pip install)

```bash
cd network_dismantling/common/external_dismantlers/
make clean
make
```

## Uncompress the dataset

Due to GitHub file size limitations, we had to compress the dataset and split the files into smaller ones.
You should be able to extract the files opening the

> dataset.tar.gz.aa

file with your compression tool GUI (like Ark).

If that doesn't work, the following command should do the trick:

```bash
cd dataset
cat dataset.tar.gz.* | tar xzvf -
```

We provide the synthetic data used for training and for testing.
Regarding the real-world test networks, we provide a sub-set due to the various licensing of the original data.
While most of these networks are publicly available[^1], feel free to email us to get the full dataset.

## Uncompress the pre-trained models

We provide the pre-trained models in the
> out.tar.gz
> file, which also includes the DataFrames containing the results of the other algorithms (_heuristics.csv_) and the
> maching of the best model for each network (by name).

As above, you can use the GUI of your compression tool (like Ark) or run the following command:

```bash
cd out
cat out.tar.gz.* | tar xzvf -
```

## Using the scripts

Every runnable script in this package comes with an _ArgParse_ interface that provides a help command (--help flag)
For instance:

```bash
python network_dismantling/machine_learning/pytorch/reproduce_results.py --help
```

### Reproducing our results

After the setup detailed above, you can reproduce the results in the paper using the scripts detailed in this section.
The base script (reproduce_results.py) and its variations will run the experiments and store the run info (i.e., the
resulting AUC, the removals list that include the removed node, the resulting LCC and SLCC size for each removal, etc.)
into a new file that can be used for plotting.

#### Dismantling results

```bash
# If you wish to reproduce the results on medium/small networks
python -u network_dismantling/machine_learning/pytorch/reproduce_results.py -lt dataset/test/dataset/

# If you wish to reproduce the results on large networks
python -u network_dismantling/machine_learning/pytorch/reproduce_results.py -lt dataset/test_large/dataset/

# If you wish to reproduce the results on synthetic networks
python -u network_dismantling/machine_learning/pytorch/reproduce_results.py -lt dataset/test_synth/dataset/ --file out/df/synth_train_NEW/t_0.18/T_0.1/GAT_Model.SYNTH.csv
```

If the run file is not found, you can specify its location with the --file flag. Keep in mind that the script requires
the same folder structure that comes with the data to get some information about the models to use.

You can specify how many jobs to run in parallel with the --jobs (-j) flag and how many processes should access each of
your GPUs simultaneously with the --simultaneous_access (-sa) flag. The GPU is used only to predict the dismantling
order and will be freed as soon as the predictions are computed, so it makes perfect sense to limit the simultaneous
access to avoid Out of Memory (OOM) issues while pushing on parallelism.

The expected run-time really depends on your hardware configuration and on the number of jobs you use. For all the
small/medium size networks, however, it should be quite fast (a few minutes at most). The large networks can take more
time (check the table in the paper to get an idea) and also keep in mind that, due to their size, parallelism and
simultaneous access should be reduced to avoid OOM issues.

The reinsertion phase (i.e., GDM+R) can be performed as described below in [Reinsert nodes](####Reinsert-nodes)

#### Reinsert nodes

Given a .csv run file, you can reintroduce the nodes (i.e., get the GDM+R results) with the following:

```bash
python -u network_dismantling/machine_learning/pytorch/reinsert.py -lt <NETWORKS_FOLDER> [-f <CSV_RUN_FILE>]

# Examples:
# The default -f (--file) value is "out/df/synth_train_NEW/t_0.18/T_0.1/GAT_Model_REPRODUCE.csv", i.e., the reproduced results obtained with the steps above.

# If you wish to reintroduce nodes on medium/small networks
python -u network_dismantling/machine_learning/pytorch/reinsert.py -lt dataset/test/dataset/

# If you wish to reintroduce nodes on large networks
python -u network_dismantling/machine_learning/pytorch/reinsert.py -lt dataset/test_large/dataset/

# If you wish to reintroduce nodes on synthetic networks
python -u network_dismantling/machine_learning/pytorch/reinsert.py -lt dataset/test_synth/dataset/ --file out/df/synth_train_NEW/t_0.18/T_0.1/GAT_Model_REPRODUCE.SYNTH.csv
```

This step can take quite some time for large networks. We have not optimized or changed the original code from
Braunstein et al. to avoid any bias.

#### EarlyWarning

You can prepare the EarlyWarning signal for plotting with the following command:

```bash
python -u network_dismantling/machine_learning/pytorch/reproduce_results_ew.py -f out/df/synth_train_NEW/t_0.18/EW/GAT_Model.csv -lt ../dataset/test_grid_ew/dataset/
```

This script will run the dismantling from scratch and store the predictions (p_n) for every node in the output file.

### Running a grid search

If you prefer running a grid search (and/or training the models), you can use the _grid.py_ script.
This script will perform a grid search on all the combinations of input parameters.

If the model is found (i.e., it is already trained and stored in the folders), it won't be trained again.
Similarly, if a dismantling is found (same network and parameters), the dismantling won't be run again.
On our (shared) hardware, training a model takes about ~5m.

Usage:

```bash
cd GDM 
python -u network_dismantling/machine_learning/pytorch/grid.py -lm <TRAIN_NETWORKS_LOCATION> -lt <TEST_NETWORKS_LOCATION> [-Ft "<GLOB_STYLE_FILTER>"] -e <NUM_EPOCHS> -wd <WEIGHT_DECAY> -r <LEARNING_RATE> -CL <CONVOLUTIONAL_LAYER_SIZES> -H <HEADS> -FCL <MPL_LAYERS> --features <NODE_FEATURES> -mf <MIN_NODE_FEATURES_NUM> -MF <MAX_NUM_FEATURES_NUM> -t <LEARNING_TARGET> -T <DISMANTLING_TARGET_SIZE> -j <MAX_JOBS> --simultaneous_access <MAX_JOBS_SIMOULTANEOUSLY_ON_GPU> [--force_cpu]

# Example:
python -u network_dismantling/machine_learning/pytorch/grid.py -lm dataset/synth_train_NEW/dataset/ -lt dataset/test/dataset/ -Ft "*" -e 50 -wd 1e-5 -r 0.003 -CL 40 30 20 10 -CL 30 20 -CL 40 30 -H 1 1 1 1 -H 5 5 -H 10 10 -FCL 100 100 --features degree clustering_coefficient kcore chi_degree -mf 4 -MF 4 -t "t_0.18" -T 0.10 -j 8 --simultaneous_access 4
```

The dismantling runs will be stored in "out/df/synth_train_NEW/t_0.18/T_0.1/GAT_Model_GRID.csv" while the prediction and
dismantling time will be saved in "out/df/synth_train_NEW/t_0.18/T_0.1/GAT_Model_GRID.time.csv".

### Dataset generation

We provide the networks we use in our experiments in the repository under the dataset/ folder[^1].

However, if you wish to test with your own networks, you can convert them from the edgelist (.el) to the GraphML (
.graphml) file-format via the command:

```bash
python -u network_dismantling/common/any2graphml.py -i <INPUT_FOLDER> -o <OUTPUT_FOLDER> -e <INPUT_FORMAT>

# Example
python -u network_dismantling/common/any2graphml.py -i dataset/MY_FOLDER/ -o dataset/MY_FOLDER/dataset/ -e el
```

After converting (or, if you already have networks in such format) you can add the node features and other info required
via:

```bash
python -u network_dismantling/machine_learning/pytorch/dataset_generator.py -d <GRAPHML_FOLDER> -uf -j <NUM_JOBS> [-F <GLOB_STYLE_FILTER>]

# Example
python -u network_dismantling/machine_learning/pytorch/dataset_generator.py -d dataset/MY_FOLDER/dataset/ -uf -j 4 -F "*"
```

[^1]: For more information about the data sources, citations, licenses, etc. see the [Read-me](dataset/README.md) in the
dataset folder.

### Plotting

#### Barplots

The barplot script will produce the .pdf output along with a table in .csv and .tex format.

It is worth mentioning that the -P flag (--pivot) will pivot the table.

The following examples will point to your locally reproduced results (i.e., the "
out/df/synth_train_NEW/t_0.18/T_0.1/GAT_Model_REPRODUCE.csv" file).
However, if you don't wish to use that but the data in the repository, you can replace the --file flag with
> --file out/df/synth_train_NEW/t_0.18/T_0.1/GAT_Model.csv

##### Real-world networks

For small and medium size networks:

```bash
# No reinsertion
python -u network_dismantling/machine_learning/pytorch/table_output.py -f out/df/synth_train_NEW/t_0.18/T_0.1/GAT_Model_REPRODUCE.csv -qp 'heuristic in ["EGND", "GND", "MS", "EI_s1"] or (heuristic=="degree" and static==False)' -po out/plot/synth_train_NEW/t_0.18/T_0.1/barplots/no_reinsertion/ -P

# With reinsertion
python -u network_dismantling/machine_learning/pytorch/table_output.py -f out/df/synth_train_NEW/t_0.18/T_0.1/GAT_Model_REPRODUCE.csv -qp 'heuristic in ["CoreHD", "GNDR", "MSR"] or (heuristic=="collective_influence_ell2" and static==False)' -q 'network in @rw_test_networks' -po out/plot/synth_train_NEW/t_0.18/T_0.1/barplots/reinsertion/ -fr -P

# Full table
python -u network_dismantling/machine_learning/pytorch/table_output.py -f out/df/synth_train_NEW/t_0.18/T_0.1/GAT_Model_REPRODUCE.csv -qp '(static==True and heuristic!="random" and ~heuristic.str.contains("_ell")) or (static!=True and static!=False) or ((heuristic=="degree" or heuristic.str.contains("_ell2")) and static!=True)' -po out/plot/synth_train_NEW/t_0.18/T_0.1/barplots/full_table/ -fr
```

For large networks:

```bash
# Full table
python -u network_dismantling/machine_learning/pytorch/table_output.py -f out/df/synth_train_NEW/t_0.18/T_0.1/GAT_Model_REPRODUCE.csv -qp 'heuristic in ["GND", "MS", "GNDR", "MSR", "CoreHD"]' -po out/plot/synth_train_NEW/t_0.18/T_0.1/barplots/large/ -fr -L -P
```

Again, if you wish to un-pivot the table, you can remove the -P flag.

##### Synthetic networks

We have a script to generate the synthetic network barplot and table. The usage is nearly identical as the
barplot_output.py described above.
To reproduce the plots in the paper:

```bash
# Barplot and table
python -u network_dismantling/machine_learning/pytorch/table_output_synth.py -f out/df/synth_train_NEW/t_0.18/T_0.1/GAT_Model_REPRODUCE.SYNTH.csv -s "r_auc" -po out/plot/synth_train_NEW/t_0.18/T_0.1/barplots/synth/  -qp 'heuristic!="random"' -fr -P

## Once again, if you don't wish to use your locally reproduced results, you can point to the run file that comes with the data. E.g.:
python -u network_dismantling/machine_learning/pytorch/table_output_synth.py -f out/df/synth_train_NEW/t_0.18/T_0.1/GAT_Model.SYNTH.csv -s "r_auc" -po out/plot/synth_train_NEW/t_0.18/T_0.1/barplots/synth/  -qp 'heuristic!="random"' -fr -P
```

If you wish to un-pivot the table, you can remove the -P flag.

#### Dismantling curves

You can plot the dismantling curves using the grid_output.py script.

The following examples will point to your locally reproduced results (i.e., the "
out/df/synth_train_NEW/t_0.18/T_0.1/GAT_Model_REPRODUCE.csv" file).
However, if you don't wish to use that but the data in the repository, you can replace the --file flag with
> --file out/df/synth_train_NEW/t_0.18/T_0.1/GAT_Model.csv

```bash
# Plot small/medium size networks
python network_dismantling/machine_learning/pytorch/grid_output.py -f out/df/synth_train_NEW/t_0.18/T_0.1/GAT_Model_REPRODUCE.csv -sf 1 -p -po out/plot/synth_train_NEW/t_0.18/T_0.1/dismantling/ -q 'network in @rw_test_networks' -qp '(static!=True and static!=False and ~heuristic.str.contains("_s2")) or (static==False and heuristic in ["degree", "collective_influence_ell2"])' -fr

# Plot large networks
python network_dismantling/machine_learning/pytorch/grid_output.py -f out/df/synth_train_NEW/t_0.18/T_0.1/GAT_Model_REPRODUCE.csv -sf 1 -p -po out/plot/synth_train_NEW/t_0.18/T_0.1/dismantling/ -q 'network in @rw_large_test_networks' -qp '(heuristic in ["GND", "GNDR", "MS", "MSR", "CoreHD"])' -fr
```

#### EarlyWarning

Plotting the Early Warning signal curve requires another script.
After reproducing the results as detailed in the previous section, you can run:

```bash
python network_dismantling/machine_learning/pytorch/grid_output_pi.py -f out/df/synth_train_NEW/t_0.18/EW/GAT_Model_REPRODUCE.csv -p -po out/plot/synth_train_NEW/t_0.18/T_0.1/EarlyWarning/ -qp 'heuristic in ["GNDR", "MSR", "degree", "CoreHD"]'
```

This will create a sub-folder for each network in the output folder (
out/plot/synth_train_NEW/t_0.18/T_0.1/EarlyWarning/). The sub-folders contain, for each heuristic attack simulation, a
plot (PDF) file, plus a .csv file containing the values of LCC, SLCC and EW ("ew" column) after each removal.

## Issues

If you find any issue with the code after reading this file carefully, please feel free to submit an Issue in the GitHub
repository.
Please provide information about your system, your environment, the traceback of the exception (if any), the script
where the issue appeared, the input parameters and any other useful information.
