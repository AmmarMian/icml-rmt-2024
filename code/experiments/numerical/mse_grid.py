# Experiment file for comparing different algorithms of computing the Fréchet
# mean over SPD  matrices. The file produce a visualization of the MSE over a
# 3D grid of parameters n_features, n_matrices and n_samples.

import os
import sys
import yaml
import argparse
import time

import numpy as np
import pandas as pd

from tqdm import tqdm
from functools import partial
from itertools import product
from joblib import Parallel, delayed

# Setup logging
import logging
import rich
from rich.logging import RichHandler
FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]",
    handlers=[RichHandler(markup=True)],
)

# Import algorithms from src
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.spd_manifold import SPD
from src.utils import (
        random_SPD, random_points_from_mean,
        random_gaussian_data, riemannian_dist2,
        geometric_mean_2steps, parse_args)
from src.mean import geometric_mean, RMT_geometric_mean

import matplotlib.pyplot as plt
# Activate LaTex rendering for matplotlib
import matplotlib
matplotlib.rcParams['text.usetex'] = True

# Styling matplotlib
plt.style.use('dark_background')
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['grid.alpha'] = 0.15
matplotlib.rcParams['grid.linestyle'] = 'dotted'
matplotlib.rcParams['axes.facecolor'] = (0.1, 0.1, 0.1, 0.4)
matplotlib.rcParams['font.family'] = 'serif'
# matplotlib.rcParams['font.serif'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 11

if __name__ == "__main__":

    # Define parser
    parser = argparse.ArgumentParser(
            description="Plot the MSE over the iteration number for several"
            " algorithms. The options provided will overwrite the config file.")
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--n_features', type=str, default=None,
                        help='Number of features separated by commas in a string')
    parser.add_argument('--n_matrices', type=str, default=None,
                        help='Number of matrices separated by commas in a string')
    parser.add_argument('--n_samples', type=str, default=None,
                        help='Number of samples separated by commas in a string')
    parser.add_argument('--n_trials', type=int, default=None,
                        help='Number of MonteCarlo trials')
    parser.add_argument('--n_iterations_max', type=int, default=None,
                        help='Number of iterations max for RMT algorithm')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--condition_number', type=float, default=None,
                        help='Condition number of the mean')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of jobs for parallel computing')
    parser.add_argument('--plot_x', type=str, default='n_matrices',
                        help='Parameter for x axis')
    parser.add_argument('--results_path', type=str, default=None,
                        help='Path to store results')
    args = parser.parse_args()

    for key in ["n_features", "n_matrices", "n_samples"]:
        if getattr(args, key) is not None:
            setattr(args, key, [int(x) for x in getattr(args, key).split(",")])

    grid_parameters = ["n_features", "n_matrices", "n_samples"]
    x_index = grid_parameters.index(args.plot_x)
    y_index = (x_index + 1) % 3
    z_index = (x_index + 2) % 3

    config = parse_args(args, experiment="mse_iteration")

    parameters_string = ""
    for key, value in config.items():
        parameters_string += f"    * [bold]{key}[/bold]: {value}\n"
    logging.info(f"Parameters:\n{parameters_string}")

    # Create results folder
    logging.info(f"Results will be stored in [italic]{config['results_path']}")
    os.makedirs(config["results_path"], exist_ok=True)
    with open(os.path.join(config["results_path"], "config.yml"), "w") as f:
        yaml.dump(config, f)

    # Set random seed
    if config["seed"] is None:
        config["seed"] = time.time()
    np.random.seed(config["seed"])

    # Setting up experiment
    estimation_methods = dict(
        SCM = partial(geometric_mean_2steps,
                      cov_estimator="scm"),
        LW_linear = partial(geometric_mean_2steps,
                            cov_estimator="lw_linear",
                            return_iterates=False),
        OAS = partial(geometric_mean_2steps,
                      cov_estimator="oas",
                    return_iterates=False),
        LW_nonlinear = partial(geometric_mean_2steps,
                               cov_estimator="lw_nonlinear",
                               return_iterates=False),
        RMT = partial(RMT_geometric_mean, return_iterates=False)
    )

    error_function = riemannian_dist2

    # Iterator over the grid of parameters
    iterator = product(config["n_features"], config["n_matrices"],
                    config["n_samples"], range(config["n_trials"]))
    logging.info("Filtering unusable set of parameters")
    iterator = filter(lambda x: x[2] > x[0], iterator)
    # Estimate of the number of grid points
    unique_parameters = list(set(iterator))
    unique_parameters.sort()
    n_grid_points = len(unique_parameters)
    logging.info(f"Final number of grid points: {n_grid_points}")

    # One monte carlo trial
    def one_montecarlo_trial(trial_no: int, 
                            n_features: int, n_matrices: int,
                            n_samples: int) -> dict:
        # Just in case
        np.random.seed(trial_no+config["seed"])

        # Generate data
        SPD_mean = random_SPD(n_features,
                            condition_number=config["condition_number"])
        SPD_matrices = random_points_from_mean(SPD_mean,
                                            n_matrices,
                                            scale=0.1,
                                            mode="exact")
        data = random_gaussian_data(SPD_matrices, n_samples)

        # Compute estimates of geometric mean
        estimates = {
            'n_features': n_features,
            'n_matrices': n_matrices,
            'n_samples': n_samples,
            'trial_no': trial_no
        }
        for name, method in estimation_methods.items():
            estimated_mean = method(data)
            estimates[name] = error_function(SPD_mean, estimated_mean)
        
        return estimates

    # Monte carlo loop in parallel
    logging.info(f"Starting MonteCarlo loop with {config['n_trials']} trials")
    errors = Parallel(n_jobs=config['n_jobs'])(
            delayed(one_montecarlo_trial)(trial_no,
                                        n_features, n_matrices, n_samples)
            for n_features, n_matrices, n_samples, trial_no
            in tqdm(unique_parameters))
    logging.info("MonteCarlo loop finished")

    # Formatting results into a dataframe
    logging.info("Formatting results")
    columns = ["n_features", "n_matrices", "n_samples", "trial_no"] + \
            list(estimation_methods.keys())
    df = pd.DataFrame(columns=columns)
    for i, parameters in enumerate(unique_parameters):
        n_features, n_matrices, n_samples, trial_no = parameters
        df.loc[i] = [n_features, n_matrices, n_samples, trial_no] + \
            [errors[i][method] for method in estimation_methods.keys()]
    df.drop(columns=["trial_no"], inplace=True)
    df.sort_values(by=['n_features', 'n_matrices', 'n_samples'])
    df_mean = df.groupby(["n_features", "n_matrices", "n_samples"]).mean()
    df_5 = df.groupby(["n_features", "n_matrices", "n_samples"]).quantile(0.05)
    df_95 = df.groupby(["n_features", "n_matrices", "n_samples"]).quantile(0.95)
    logging.info("Results formatted")
    mean_string = "Mean:\n" + df_mean.to_string(float_format="{:.2f}".format)
    quantile_5_string = "5th percentile:\n" + \
            df_5.to_string(float_format="{:.2f}".format)
    quantile_95_string = "95th percentile:\n" + \
            df_95.to_string(float_format="{:.2f}".format)
    logging.info(mean_string)
    logging.info(quantile_5_string)
    logging.info(quantile_95_string)

    # Plotting
    colors = ["b", "g", "r", "c", "m"]
    unique_x = sorted(list(set(df_mean.index.get_level_values(x_index))))
    unique_y = sorted(list(set(df_mean.index.get_level_values(y_index))))
    unique_z = sorted(list(set(df_mean.index.get_level_values(z_index))))
    mse_min = min(df_mean.min())
    mse_max = max(df_mean.max())
    for y, z in product(unique_y, unique_z):
        y, z = int(y), int(z)
        data_mean = df_mean.query(f"{grid_parameters[y_index]} == {y} & " +\
                                  f"{grid_parameters[z_index]} == {z}")
        data_5 = df_5.query(f"{grid_parameters[y_index]} == {y} & " +\
                f"{grid_parameters[z_index]} == {z}")
        data_95 = df_95.query(f"{grid_parameters[y_index]} == {y} & " +\
                f"{grid_parameters[z_index]} == {z}")
        x_values = [index[x_index] for index in data_mean.index]
        if len(x_values) > 0:
            fig, ax = plt.subplots(figsize=(9,6))
            for i, method in enumerate(estimation_methods.keys()):
                ax.plot(x_values, list(data_mean[method]), label=method, color=colors[i])
                ax.fill_between(x_values,
                                list(data_5[method]),
                                list(data_95[method]),
                                alpha=0.3, color=colors[i])
            ax.legend(loc="upper right")
            ax.set_yscale("log")
            ax.set_xlabel(args.plot_x)
            ax.set_ylabel("MSE")
            ax.set_ylim([mse_min, mse_max])
            ax.set_title(f"MSE with {grid_parameters[y_index]}={y} and {grid_parameters[z_index]}={z}")
            file_path = os.path.join(
                config["results_path"],
                f"mse_iteration_{grid_parameters[y_index]}={y}_{grid_parameters[z_index]}={z}.png")
            plt.savefig(file_path)
            logging.info(f"Plot saved in {file_path}")
