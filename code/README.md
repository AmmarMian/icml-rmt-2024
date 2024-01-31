# Frechet-Karcher Mean with low sample support

This repository contains code to reproduce numerical experiments presented in a submitted paper at ICML 2024:
> "Frechet-Karcher Mean with low sample support: exploiting random matrix theory and Riemannian geometry"

## Project organization

TODO

## Installing the running environment

Several options:

1. With a pre-existing python installation

Make sure to have the following packages installed:
* numpy
* scipy
* pandas
* scikit-learn
* pyrieman=0.6
* MOABB
* doit
* pyyaml
* wget

2. Creating a new conda environment

Create the conda environment by placing yourself in the repo directory and run:
``` console
conda env create -f environment/conda.yml
```

then activate it using:
``` console
conda activate icml-rmt-mean
```

3. Using an [apptainer](https://apptainer.org/) container

For complete reproduction of the running environment, you can create a container after installing apptainer by running:
```console
apptainer build environment/icml-rmt-mean.sif environment/icml-rmt-mean.def
```

then run any subsequent command using the syntax:
```console
apptainer run environment/icml-rmt-mean.sif [your command]
```

## Reproducing the paper

### Downloading datasets

1. EEG datasets:

The EEG datasets are available thanks to the [MOABB](https://github.com/NeuroTechX/moabb) project and will be downloaded automatically when needed.

2. Hyperspectral datasets

Hyperspectral datasets are available thanks to (https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes). You can download them manually or run the command:
```console
doit download_hyperspectral
```

### Producing numerical experiments 

A few [pydoit](https://pydoit.org/) taks have been created to replicate the results presented as presented in the following table. To run a task simply run:

``` console
doit task_name
```

| task_name   | description                                                                |
|-------------|----------------------------------------------------------------------------|
| indianpines | produce the table relative to indianpines hyperspectral dataset clustering |
| salinas     |                                                                            |
| paviau      |                                                                            |
| pavia       |                                                                            |
| ksc         |                                                                            |


