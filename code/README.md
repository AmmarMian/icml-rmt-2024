# Frechet-Karcher Mean with low sample support

This repository contains code to reproduce numerical experiments presented in a submitted paper at ICML 2024:
> "Frechet-Karcher Mean with low sample support: exploiting random matrix theory and Riemannian geometry"

## Project organization

```console

├── CONFIG.sh
├── data
├── dodo.py
├── environment
│  ├── conda.yml
│  └── icml-rmt-mean.def
├── experiments
│  ├── eeg
│  │  └── main.py
│  ├── hyperspectral
│  │  ├── configs
│  │  ├── data.py
│  │  ├── main.py
│  │  └── pipelines_utils.py
│  └── numerical
│     ├── batch_export_results.sh
│     ├── configs
│     ├── export_to_latex.py
│     ├── mse_grid.py
│     ├── mse_iteration.py
│     ├── mse_nmatrices.py
│     ├── mse_nsamples_cov.py
│     └── mse_samples.py
├── README.md
└── src
   ├── __init__.py
   ├── classification.py
   ├── covariance.py
   ├── distance.py
   ├── estimation.py
   ├── mean.py
   ├── spd_manifold.py
   └── utils.py
```

The main code for the methods is provided in `src/` directory. Notably in file `mean.py` for the computation of the new Fréchet mean. Experiments are provided in the `experiments/` directory. The `dodo.py` file is a [pydoit](https://pydoit.org/) file to run the experiments. The `environment/` directory contains the conda environment file and the definition file for the apptainer container. The `data/` directory is used to store the datasets.

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

where the task are reported in the following tables:

1. Synthetic experiments

| task_name         | associated command                                                                                                                  | description                                                                                                           |
|-------------------|-------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| mse_iteration     | `python experiments/numerical/mse_iteration.py --config experiments/numerical/configs/mse_iteration.yml`                              | produce visualization of error of MSE of estimated mean as a function of algorithm iteration. Not shown in the paper. |
| mse_iteration_cov | `python experiments/numerical/mse_nsamples_cov.py --config experiments/numerical/configs/mse_nsamples_cov/mse_nsamples_cov_64.yml`    | produces figure 1                                                                                                     |
| mse_nsamples      | `python experiments/numerical/mse_nsamples.py --config experiments/numerical/configs/mse_nsamples/mse_nfeatures_64.yml`               | produces figure 2                                                                                                     |
| mse_nmatrices     | `python experiments/numerical/mse_nmatrices.py --config experiments/numerical/configs/mmse_nmatrices/se_nfeatures_64_nsamples_128.yml` | produces figure 3

The scripts are done in a manner that one can ovveride the config file using an option '--parameter <parameter_value>'. Some other configuration files are also available in the 'experiments/numerical/configs' directory. To run different paramters use the python script and the --config option while ovveriding the parameters you want to change.

By default, the results are saved in the 'results' directory. The figures are saved in the `results/<task_name>` directory.

2. Hyperspectral experiments

| task_name   | description                                                                |
|-------------|----------------------------------------------------------------------------|
| indianpines | launch experiments relative to indianpines hyperspectral dataset clustering |
| salinas     | launch experiments relative to salinas hyperspectral dataset clustering     |
| pavia       | launch experiments relative to pavia hyperspectral dataset clustering       |
| ksc         | launch experiments relative to ksc hyperspectral dataset clustering         |

Depending on the task and the running computer, the experiment can take a lot of time. Before launching those tasks, you need to dowload the hyperspectral datasets using the command:
```console
doit download_hyperspectral
```

The results will be saved in the `results/hyperspectral/<task_name>` directory.

3. EEG experiments

For now, no task has been set. It will be done in the near future.

You can run an experiment using:
```console
python experiments/eeg/main.py <your_parameters>
```

where the parameters can be set as follows:
```console
usage: main.py [-h] [--paradigm PARADIGM] [--tmin TMIN] [--tmax TMAX] [--resample RESAMPLE] [--pipelines PIPELINES]
               [--dataset DATASET] [--subject SUBJECT [SUBJECT ...]] [--remove_subject REMOVE_SUBJECT]
               [--dataset_path DATASET_PATH] [--results_path RESULTS_PATH] [--seed SEED]

Run a BCI benchmark.

options:
  -h, --help            show this help message and exit
  --paradigm PARADIGM   Paradigm to use. (lr, mi)
  --tmin TMIN           Start time in seconds.
  --tmax TMAX           End time in seconds.
  --resample RESAMPLE   Resample rate in Hz.
  --pipelines PIPELINES
                        Pipelines to use separated by a comma.
  --dataset DATASET     Dataset to use. (bnci_2014_001, physionet, schirrmeister2017, munich_mi, cho2017, weibo2014,
                        zhou2016, lee2019, ofner2017, khushaba2017, lee2019_mi)
  --subject SUBJECT [SUBJECT ...]
                        Subject(s) to use.
  --remove_subject REMOVE_SUBJECT
                        Subject(s) to remove.
  --dataset_path DATASET_PATH
                        Path to read/download the dataset.
  --results_path RESULTS_PATH
                        Path to store the results.
  --seed SEED           Random seed.
```

The data will be downloaded directly by MOABB into `~/mne_data/`.
