# WARNING

This repository is being refactored and improved to make it as user-friendly as possible.
Some features may be missing or may not work as expected.
We are working on it, and we will update this warning as soon as we are done.

If you have any issues, please contact the authors or open an issue on GitHub.
We apologize for the inconvenience.

# Robustness and resilience of complex networks

This repository contains the scripts and data from the "Robustness and resilience of complex networks" paper by Oriol
Artime, Marco Grassia, Manlio De Domenico, James P. Gleeson, Hernán A. Makse, Giuseppe Mangioni, Matjaž Perc and Filippo
Radicchi.
The manuscript is currently under review.


## Citations

### How to cite the paper

The paper is currently under review.
We will update this section as soon as it is published.

>

BibTex:

```bibtex

```

RIS:

```ris

```

Also, please cite the original papers of the algorithms you use (see the [Algorithms](#algorithms) section),
the datasets you use (see the [Dataset](#dataset) section), and the libraries you use (see the [Setup](#setup)

### How to cite this repository

>

```bib

```

## License

The code implemented for the review (that is, the wrappers, the dismantlers and the output scripts) available in this
repository is released under the GPLv3 License.
See the LICENSE file for more information.
However, **please note that the code of the specific integrated algorithms is redistributed from their original
repositories and may be
subject to different licenses**.
Thus, the GPLv3 License may not apply to them and to the code in their folders.
If you wish to use them, please refer to their original LICENSE files (in the respective folders), and to the original
authors.

## Algorithms

This library integrates and provides a common interface to the following algorithms:

- Brute Force
- [Collective Influence (CI)](https://github.com/makselab/Collective-Influence):  [reference](CITATIONS.md#collective-influence-ci)
- CoreGDM (almost ready, needs some fixes): [reference](CITATIONS.md#coregdm)
- CoreHD: [reference](CITATIONS.md#corehd)
- [Ensemble GND (EGND)](https://github.com/renxiaolong/2019-Ensemble-approach-for-generalized-network-dismantling): [reference](CITATIONS.md#ensemble-gnd-egnd)
- [Explosive Immunization (EI)](https://github.com/pclus/explosive-immunization): [reference](CITATIONS.md#explosive-immunization-ei)
- [FINDER](https://github.com/FFrankyy/FINDER): [reference](CITATIONS.md#finder)
- Generalized Network Dismantling (GND): [reference](CITATIONS.md#generalized-network-dismantling-gnd)
- [Graph Dismantling Machine (GDM)](https://github.com/renxiaolong/Generalized-Network-Dismantling): [reference](CITATIONS.md#graph-dismantling-machine-gdm)
- [Min-Sum](https://github.com/abraunst/decycler/): [reference](CITATIONS.md#min-sum-ms--greedy-reinsertion-algorithm)
- Node heuristics (to be integrated in the main dismantler, need to use the separate script for now)
    - Degree
    - Betweenness
    - Eigenvector centrality
    - PageRank

The package also includes the greedy reinsertion algorithm proposed for the reinsertion phase
of [Min-Sum](https://github.com/abraunst/decycler/).

**If you use any of these algorithms, please also cite the original papers.
See the [Citations file](CITATIONS.md) for more information.**

## Setup
> Expected time for this step: 30m - 1h

This package is written in Python (tested with version 3.9), and uses graph-tool[1] for the graph processing.
The bundled scripts should take cake of building and setting up the included algorithms.
However, the users may need to manually install some dependencies, see [Requirements](#requirements) for more
information.

Regarding the Python environment, we recommend you to use Conda environments and avoid installing any package in your
main Python environment, to avoid conflicts or issues with your operating system.
This step is crucial, since some of the algorithms depend (directly and/or indirectly) on different libraries version (
e.g., pytorch and tensorflow with different library version requirements).

### Requirements

This package includes many C/C++ algorithms, which require a C++ compiler and some libraries.
Make sure you have the following libraries installed on your system:

- GCC (tested with version 12.x)
- Build essentials (e.g., make, cmake, etc.)
- OpenMP
- Boost
- CMake

In an Ubuntu system, you can install them with the following command:

```bash
sudo apt install build-essential gcc-12 g++-12 cmake libboost-all-dev libomp-dev
```

While in a Fedora system, you can use the following command:

```bash
sudo dnf install development-tools gcc gcc-c++ make cmake boost-devel libomp-devel
```

To install Conda, please refer to
the [miniconda official documentation](https://docs.conda.io/en/latest/miniconda.html), or to the
full [Anaconda distribution](https://www.anaconda.com/).

To make the setup phase easier, we provide the Conda[2] environments we used to run our experiments.
In particular, we include two Conda[2] environments, one for FINDER and one for all the other algorithms.
See the [Bundled environments](#bundled-environments) section for more information.
**Unfortunately, such environments are not guaranteed to work on all systems (e.g., Windows) or to be optimized for your
host machine.
We strongly suggest to build your own environment using the instructions in
the [Building your own environment](#building-your-own-environment) section.**

### Bundled environments

To set up the provided environments:

1. Install conda-pack:

    ```bash
    conda install conda-pack
    ```

2. Unpack the environments using conda-pack as follows:
   - Generic and GDM's environment
      ```bash
      cd review
      
      conda-unpack -n gdm -f environment_full.tar.gz
      ```
    
      Finder environment:
    
      ```bash
      cd review
      
      conda-unpack -n finder -f environments/finder.tar.gz  
      ```

3. Follow the instructions in section [Common](#common)

### Building your own environment

In this section, we will show how to build the Conda environments with the required dependencies for the various
algorithms.
Remember to switch to the right environment before running the algorithms.
If you have any issues, please refer to the official documentation of the specific algorithms.
You can find them in their sub-folders.

#### All algorithms except FINDER and GDM

```bash
conda create -n dismantling python=3.9 graph-tool seaborn numpy scipy matplotlib mkl-service -c anaconda -c conda-forge

conda activate dismantling
```

#### GDM

In progress.

TODO: Update the old environment and convert the pre-trained PyG 1.2.1 models to the newer PyG GAT models.

```bash
conda env create -f environment.yml

conda activate gdm

```

[//]: # (conda create -n gdm python=3.9 graph-tool pytorch torchvision torchaudio cudatoolkit pyg seaborn numpy scipy matplotlib dill jupyter mkl-service -c pyg -c pytorch -c nvidia -c anaconda -c conda-forge)

#### FINDER

```bash
# If you have a CUDA compatible GPU
conda create --name finder cython==0.29.13 networkx==2.3 numpy==1.17.3 pandas==0.25.2 scipy==1.3.1 tensorflow-cpu==1.14.0 tqdm==4.36.1 dill graph-tool -c conda-forge

# Otherwise, the models will be run on the CPU
conda create --name finder cython==0.29.13 networkx==2.3 numpy==1.17.3 pandas==0.25.2 scipy==1.3.1 tensorflow==1.14.0 tqdm==4.36.1 dill graph-tool -c conda-forge
```

### Common

After creating (or unpacking) the environment, you can install the package as follows:

```bash
conda activate <ENVIRONMENT_NAME>

cd review

pip install -e .
```

FINDER also requires the following additional steps, due to the use of Cython:
Please make sure to use the GCC-8 compiler, as the code is not compatible with GCC-9 right now.

```bash
# Ubuntu
sudo apt install gcc-8 g++-8

# Fedora
sudo dnf install gcc-8 g++-8
```

```bash
cd network_dismantling/FINDER/

export CC=/usr/bin/gcc-8
export GCC=/usr/bin/gcc-8

python setup.py clean
python setup.py build
````

## Dataset

### Data used in the paper

This repository contains the datasets used in the experiments.
We refer the Reader to the [Read-me](dataset/README.md) in the dataset folder for more information about the data
sources, citations, licenses, etc.

### New data

If you wish to use this code to dismantle your own networks, you can do so as follows:

1. First, convert your networks to the GT/GraphML formats.

    ```bash
    python -u network_dismantling/converter.py -i <INPUT_FOLDER> -o <OUTPUT_FOLDER> -e <INPUT_FORMAT>
    
    # Example
    python -u network_dismantling/converter.py -i dataset/MY_FOLDER/ -o dataset/MY_FOLDER/dataset/ -e el
    ```

   Arguments:
    - -h, --help show this help message and exit
    - -i, --input INPUT Location of the input networks (directory)
    - -o, --output OUTPUT Location of the output directory
    - -nw, --no_weights Discard weights
    - -ei, --input_ext [EXT(s)]   Input extension(s) without dot
    - -eo, --output_ext EXT Output file extension without dot (e.g., gt, graphml, ...)

3. After converting (or, if you already have networks in such formats), you can run the dismantler script as detailed in
   section [Running the dismantler](#running-the-dismantler).

## Running the dismantler

To run the dismantler, you first need to set up the environment as detailed in section [Setup](#setup).
After that, you should activate the right environment, depending on the algorithm you want to run.
For example, to run the GDM algorithm, you should run:

```bash
conda activate gdm
```

Or, to run the FINDER algorithm, you should run:

```bash
conda activate finder
```

A similar command should be used for the other algorithms.

```bash
conda activate dismantling
```

Please note that a script that takes care of switching to the right environment is provided in currently in the works.

Then, you can run the dismantler script as follows:
[//]: # (3. you can add the node features and other info required via:)

```bash
cd review

python network_dismantling/dismantler.py -l <DATASET_FOLDER> -H <DISMANTLING_ALGORITHM(s)> -t <DISMANTLING_THRESHOLD> [-i <INPUT_DATAFRAME(s)] [-o <OUTPUT_DATAFRAME>] [-F <DATA_FILTER] 
```

The script will run all the algorithms and save the results in the _out_ folder.
In particular, by default, the results will be saved in the _out/df/heuristics.csv_ DataFrame file.
The file will contain the results of all the algorithms, and will be used by the plotting scripts.

The parameters are as follows:

- --location, -l: the folder containing the datasets in the GT/GraphML format.
- --heuristics, -H: the dismantling algorithm(s) to run. See the [Algorithms](#algorithms) section for more
  information.
- --threshold, -t: the dismantling threshold to use for the dismantling algorithm(s): $t \in [0, 1]$, represents
  the target fraction of the Largest Connected Component Size.
- --output, -o: the output DataFrame file to use.
- --input, -i: the input DataFrame file(s) to read to avoid duplicate runs. The script will try to use the same
  DataFrame used for the output.
- --filter, -F: the filter to use to select the networks to run the algorithms on. The filter is a string that
  will be used to match the network names. The default is "*", which will run the algorithms on all the networks
  in the folder.
- --jobs, -j: number of parallel jobs to run. The default is 1. NOT YET IMPLEMENTED IN THIS VERSION.

Example usage:

```bash
# Run the dismantler on all the networks in the dataset/test_review folder, using the GND, the MS and the CI l2 algorithms with threshold 0.1
python network_dismantling/dismantler.py -l dataset/test_review/ -t 0.1 -H GND MS CollectiveInfluenceL2 -o out/df/heuristics.csv

# As above, but only on the networks whose name contains the string "corruption"
python network_dismantling/dismantler.py -l dataset/test_review/ -t 0.1 -H GND MS CollectiveInfluenceL2 -o out/df/heuristics.csv -F "*corruption*"

# As above, but also scans the dataset/test_review_lfr folder for networks (without filtering)
python network_dismantling/dismantler.py -l dataset/test_review/ dataset/test_review_lfr/ -t 0.1 -H CoreHD GND -o out/df/heuristics.csv 
```

## Result visualization

### Plotting

We provide a script to plot the results of the dismantling algorithms.
In particular, the _plot.py_ script will produce a plot for each network, showing the dismantling curve of the
algorithms.

```bash
python plot.py -f <DATAFRAME_FILE(s)> [-q <QUERY>] [-sf <MAX>] [-o <OUTPUT>] [-s <COLUMN>] [-sa]
```

Flags:

- -h, --help Shows the help message and exit
- -f, --file FILE(s)            Output DataFrame file(s) location
- -q, --query QUERY Provides a way to query the DataFrame.
  See [pandas.DataFrame.query](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html)
  for the syntax.
- -sf, --show_first MAX Shows at most the first MAX dismantling curves (the best)
- -o, --output OUTPUT Output plot location
- -s, --sort_column COLUMN DataFrame column used to sort the entries
- -sa, --sort_descending Sorting the entries in descending order

Example usage:

```bash
# Plot the results of the dismantling algorithms store in the out/df/heuristics.csv file, uses the out/plot/ folder as root for the output plots
python plot.py -f out/df/heuristics.csv -o out/plot/

# Same as above, but shows only the first 5 dismantling curves (the best) according to the roc_auc score (sorted ascending)
python plot.py -f out/df/heuristics.csv -o out/plot/ -sf 5 -s roc_auc -sa

# Same as above, but only for the networks whose name is "corruption"
python plot.py -f out/df/heuristics.csv -o out/plot/ -sf 5 -s roc_auc -sa -q 'network=="corruption"'

# Same as above, but only for the networks whose name contains the string "corruption"
python plot.py -f out/df/heuristics.csv -o out/plot/ -sf 5 -s roc_auc -sa -q 'network.str.contains("corruption")'
```

### Tables

The result tables are generated by the _table\_output.py_ script.
The script will generate a .csv file containing the results of the algorithms in tabular form, along with the same data
in .tex format.

The script can be run as follows:

```bash
python table_output.py -f <DATAFRAME_FILE(s)> [-q <QUERY>] [-o <OUTPUT>] [-s <COLUMN>] [-sa]
```

Flags:

- -h, --help Shows the help message and exit
- -f, --file FILE(s)            Output DataFrame file(s) location
- -q, --query QUERY Provides a way to query the DataFrame.
  See [pandas.DataFrame.query](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html)
  for the syntax.
- -o, --output OUTPUT Output plot location
- -s, --sort_column COLUMN Column values to display in the table
- -sa, --sort_descending Sorting descending
- -i, --index INDEX DataFrame column used as table rows
- -c, --columns COLUMNS DataFrame columns used as table columns
- -P, --pivot Transpose x and y axes (convenience flag for leaving -i and -c as defaults)
- -rn, --row_nan Drop any row with NaN values
- -cn, --col_nan Drop any column with NaN values

```bash
# Generate the table for the results of the dismantling algorithms store in the out/df/heuristics.csv file, uses the out/table/ folder as root for the output tables
python table_output.py -f out/df/heuristics.csv -o out/table/

# Same as above, but shows the auc values and pivots the table
python table_output.py -f out/df/heuristics.csv -o out/table/ -s roc_auc -sa -P

# Same as above, but only for the networks used in the review
python table_output.py -f out/df/heuristics.csv -o out/table/ -s roc_auc -sa -P -q 'network in @review_networks'
```

We have similar scripts for the synthetic networks, which use the same syntax.
Specifically, the _table\_output_lfr.py_ and the _table\_output_synth.py_ scripts will generate the tables for the LFR
and for the other synthetic networks, respectively.

All the table scripts will produce a .csv output along with a table in .tex format.

It is worth mentioning that the -P flag (--pivot) will pivot the table.

## Issues

If you find any issue with the code after reading this file carefully, please feel free to submit an Issue in the GitHub
repository.
Please provide information about your system, your environment, the traceback of the exception (if any), the script
where the issue appeared, the input parameters and any other useful information.

## [WIP] Contributing

This repository is meant to be a collaborative project, where new dismantling algorithms can be easily integrated and
benchmarked.
To do so, we have built a simple framework that allows to easily integrate new algorithms and compare them with the
existing ones.
New algorithms can be integrated by implementing a simple function that takes as input an undirected graph-tool graph
and returns a list of nodes to remove.
The function should then be registered using the @dismantling_algorithm decorator.

MORE INFO TO COME ... also about edge dismantling

You can integrate a new algorithm just by providing a function
If you wish to contribute to this project, please feel free to submit a Pull Request in the GitHub repository.
**Keep in mind that this project is still under development, so please contact the authors before starting to work on a
new feature.**
In fact, we are currently working on a better way to integrate new dismantling algorithms, and also on a better way to
handle the output.


[1]: https://graph-tool.skewed.de/
[2]: https://www.anaconda.com/products/individual
