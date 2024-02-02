import os
import glob
import sys
import yaml
import shlex
import subprocess
import wget
import rich

# Define the tasks
# Make it so that the default task is the project_info task
DOIT_CONFIG = {
    'verbosity': 2,
    'continue': True,
}


def task_show_readme():
    """Show the README.md file"""
    
    return {
        'basename': 'readme',
        'actions': ['python -m rich.markdown README.md'],
        'verbosity': 2
    }


def download_hyperspectral_data():
    """Download hyperspectral dataset from the web."""
    datasets = {
        'indianpines': [
            'https://www.ehu.eus/ccwintco/uploads/2/22/Indian_pines.mat',
            'https://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat',
            'https://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat'
        ],
        'salinas': [
            'https://www.ehu.eus/ccwintco/uploads/f/f1/Salinas.mat',
            'https://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat',
            'https://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat',
            ],
        'Pavia': [
            'https://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat',
            'https://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat'
            ],
        'PaviaU': [
            'https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
            'https://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat'
            ],
        'KSC': [
            'http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat',
            'http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat'
            ]
        }

    if not os.path.exists("data/hyperspectral"):
        os.makedirs("data/hyperspectral")

    for dataset_name in datasets.keys():
        rich.print(f"[bold] Downloading {dataset_name} dataset...[/bold]")
        if not os.path.exists(f"data/hyperspectral/{dataset_name}"):
            rich.print(f"    * [italic]Creating {dataset_name} directory...[/italic]")
            os.makedirs(f"data/hyperspectral/{dataset_name}")
        try:
            for url in datasets[dataset_name]:
                filename = url.split("/")[-1]
                if not os.path.exists(f"data/hyperspectral/{dataset_name}/{filename}"):
                    rich.print(f"    * [italic]Downloading {url}...[/italic]")
                    wget.download(url, out=f"data/hyperspectral/{dataset_name}")
                    print("\n")
                else:
                    rich.print(f"    * [italic]File {filename} already exists. Skipping...[/italic]")
        except Exception as e:
            rich.print(f"[bold red]Error downloading {dataset_name} dataset:[/bold red]")
            rich.print(f"[bold red]{e}[/bold red]")



def task_download_hyperspectral_data():
    """Download hyperspectral dataset from the web."""
    return {
        'basename': 'download_hyperspectral',
        'actions': [download_hyperspectral_data],
        'verbosity': 2
    }


def task_mse_estimation_cov():
    """Plot MSE of estimated covariance as a function of number of samples."""


    yield {
            'name': 'nfeatures_64',
            'actions': [["python",
                        "experiments/numerical/mse_nsamples_cov.py",
                        "--config", "experiments/numerical/configs/mse_nsamples_cov/mse_nsamples_cov_64.yml",
                        "--n_jobs", "-1", "--results_path", "results/numerical/mse_estimation_cov/64"]],
            'verbosity': 2
        }


def task_mse_iteration():
    """Plot MSE of estimated mean over simulated data as a function of algorithm iteration."""

    return {
            'basename': 'mse_iteration',
            'actions': [["python",
                         "experiments/numerical/mse_iteration.py",
                        "--config", "experiments/numerical/configs/mse_iteration.yml",
                        "--n_jobs", "-1", "--results_path", "results/numerical/mse_iteration"]],
            'verbosity': 2
        }


def tas_mse_nsamples():
    """Plot MSE of estimated mean over simulated data as a function of number of samples."""
    return {
            'basename': 'mse_nsamples',
            'actions': [["python",
                        "experiments/numerical/mse_nsamples.py",
                        "--config", "experiments/numerical/configs/mse_nsamples/mse_features_64.yml",
                        "--n_jobs", "-1", "--results_path", "results/numerical/mse_nsamples"]],
            'verbosity': 2
        }


def task_mse_nmatrices():
    """Plot MSE of estimated mean over simulated data as a function of number of matrices."""
    # yield {
    #         'name': '5_7',
    #         'actions': [["python",
    #                      "experiments/numerical/mse_nmatrices.py",
    #                     "--config", "experiments/numerical/configs/mse_nmatrices/mse_nfeatures_5_nsamples_7.yml",
    #                     "--n_jobs", "-1", "--results_path", "results/numerical/mse_nmatrices/5_7"]],
    #         'verbosity': 2
    #     }

    # yield {
    #         'name': '5_25',
    #         'actions': [["python",
    #                      "experiments/numerical/mse_nmatrices.py",
    #                     "--config", "experiments/numerical/configs/mmse_nmatrices/se_nfeatures_5_nsamples_25.yml",
    #                     "--n_jobs", "-1", "--results_path", "results/numerical/mse_nmatrices/5_25"]],
    #         'verbosity': 2
    #     }

    # yield {
    #         'name': '64_66',
    #         'actions': [["python",
    #                      "experiments/numerical/mse_nmatrices.py",
    #                     "--config", "experiments/numerical/configs/mmse_nmatrices/se_nfeatures_64_nsamples_66.yml",
    #                     "--n_jobs", "-1", "--results_path", "results/numerical/mse_nmatrices/64_66"]],
    #         'verbosity': 2
    #     }

    yield {
            'name': '64_128',
            'actions': [["python",
                         "experiments/numerical/mse_nmatrices.py",
                        "--config", "experiments/numerical/configs/mmse_nmatrices/se_nfeatures_64_nsamples_128.yml",
                        "--n_jobs", "-1", "--results_path", "results/numerical/mse_nmatrices/64_128"]],
            'verbosity': 2
        }

    # yield {
    #         'name': '64_512',
    #         'actions': [["python",
    #                     "experiments/numerical/mse_nmatrices.py",
    #                     "--config", "experiments/numerical/configs/mmse_nmatrices/se_nfeatures_64_nsamples_512.yml",
    #                     "--n_jobs", "-1", "--results_path", "results/numerical/mse_nmatrices/64_512"]],
    #         'verbosity': 2
    #     }


def task_indianpines():
    """Run K-means clustering comparison on Indian Pines dataset."""

    yield {
            'name': '5_5',
            'actions': [["python",
                        "experiments/hyperspectral/main.py",
                        "--config", "experiments/hyperspectral/configs/indianpines/indianpines_5_5.yml",
                        "--n_jobs", "-1", "--results_path", "results/hyperspectral/indianpines/5_5"]],
            'verbosity': 2
    }

    yield {
            'name': '10_5',
            'actions': [["python",
                        "experiments/hyperspectral/main.py",
                        "--config", "experiments/hyperspectral/configs/indianpines/indianpines_10_5.yml",
                        "--n_jobs", "-1", "--results_path", "results/hyperspectral/indianpines/10_5"]],
            'verbosity': 2
        }

    yield {
            'name': '16_5',
            'actions': [["python",
                        "experiments/hyperspectral/main.py",
                        "--config", "experiments/hyperspectral/configs/indianpines/indianpines_16_5.yml",
                        "--n_jobs", "-1", "--results_path", "results/hyperspectral/indianpines/16_5"]],
            'verbosity': 2
        }

    yield {
            'name': '20_7',
            'actions': [["python",
                        "experiments/hyperspectral/main.py",
                        "--config", "experiments/hyperspectral/configs/indianpines/indianpines_20_7.yml",
                        "--n_jobs", "-1", "--results_path", "results/hyperspectral/indianpines/20_7"]],
            'verbosity': 2
        }

    yield {
            'name': '32_7',
            'actions': [["python",
                        "experiments/hyperspectral/main.py",
                        "--config", "experiments/hyperspectral/configs/indianpines/indianpines_32_7.yml",
                        "--n_jobs", "-1", "--results_path", "results/hyperspectral/indianpines/32_7"]],
            'verbosity': 2
    }

    yield {
            'name': '64_9',
            'actions': [["python",
                        "experiments/hyperspectral/main.py",
                        "--config", "experiments/hyperspectral/configs/indianpines/indianpines_64_9.yml",
                        "--n_jobs", "-1", "--results_path", "results/hyperspectral/indianpines/64_9"]],
            'verbosity': 2
        }


def task_salinas():
    """Run K-means clustering comparison on Salinas dataset."""

    yield {
            'name': '5_5',
            'actions': [["python",
                        "experiments/hyperspectral/main.py",
                        "--config", "experiments/hyperspectral/configs/salinas/salinas_5_5.yml",
                        "--n_jobs", "-1", "--results_path", "results/hyperspectral/salinas/5_5"]],
            'verbosity': 2
    }
    yield {
            'name': '10_7',
            'actions': [["python",
                        "experiments/hyperspectral/main.py",
                        "--config", "experiments/hyperspectral/configs/salinas/salinas_10_7.yml",
                        "--n_jobs", "-1", "--results_path", "results/hyperspectral/salinas/10_7"]],
            'verbosity': 2
        }
    yield {
            'name': '16_11',
            'actions': [["python",
                        "experiments/hyperspectral/main.py",
                        "--config", "experiments/hyperspectral/configs/salinas/salinas_16_11.yml",
                        "--n_jobs", "-1", "--results_path", "results/hyperspectral/salinas/16_11"]],
            'verbosity': 2
        }


def task_paviaU():
    """Run K-means clustering comparison on PaviaU dataset."""
    yield {
            'name': '5_5',
            'actions': [["python",
                        "experiments/hyperspectral/main.py",
                        "--config", "experiments/hyperspectral/configs/paviaU/paviaU_5_5.yml",
                        "--n_jobs", "-1", "--results_path", "results/hyperspectral/paviaU/5_5"]],
            'verbosity': 2
    }
    yield {
            'name': '10_5',
            'actions': [["python",
                        "experiments/hyperspectral/main.py",
                        "--config", "experiments/hyperspectral/configs/paviaU/paviaU_10_5.yml",
                        "--n_jobs", "-1", "--results_path", "results/hyperspectral/paviaU/10_5"]],
            'verbosity': 2
        }



def task_pavia():
    """Run K-means clustering comparison on Pavia dataset."""
    yield {
            'name': '5_5',
            'actions': [["python",
                        "experiments/hyperspectral/main.py",
                        "--config", "experiments/hyperspectral/configs/pavia/pavia_5_5.yml",
                        "--n_jobs", "-1", "--results_path", "results/hyperspectral/pavia/5_5"]],
            'verbosity': 2
    }


def task_KSC():
    """Run K-means clustering comparison on KSC dataset."""
    yield {
            'name': '5_5',
            'actions': [["python",
                        "experiments/hyperspectral/main.py",
                        "--config", "experiments/hyperspectral/configs/KSC/KSC_5_5.yml",
                        "--n_jobs", "-1", "--results_path", "results/hyperspectral/KSC/5_5"]],
            'verbosity': 2
    }


