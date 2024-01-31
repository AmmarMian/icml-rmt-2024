# Experiment file for comparing different algorithms of computing the Fréchet
# mean over SPD  matrices. The file produce a visualization of the MSE over the
# iteration number for each algorithm.

import os
import sys
import yaml
import argparse
import time

import numpy as np
import pandas as pd

from tqdm import trange
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
    parser.add_argument('--n_features', type=int, default=None,
                        help='Number of features')
    parser.add_argument('--n_matrices', type=int, default=None,
                        help='Number of matrices')
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Number of samples')
    parser.add_argument('--n_trials', type=int, default=None,
                        help='Number of MonteCarlo trials')
    parser.add_argument('--n_iterations_max', type=int, default=None,
                        help='Number of iterations max for RMT algorithm')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--condition_number', type=float, default=None,
                        help='Condition number of the mean')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of jobs for parallel computing')
    parser.add_argument('--results_path', type=str, default=None,
                        help='Path to store results')
    parser.add_argument('--show_plots', type=bool, default=False,
                        help='Show plots (will block execution)')
    args = parser.parse_args()

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
                      cov_estimator="scm",
                    return_iterates=True),
        LW_linear = partial(geometric_mean_2steps,
                            cov_estimator="lw_linear",
                            return_iterates=True),
        OAS = partial(geometric_mean_2steps,
                      cov_estimator="oas",
                    return_iterates=True),
        LW_nonlinear = partial(geometric_mean_2steps,
                               cov_estimator="lw_nonlinear",
                            return_iterates=True),
        RMT = partial(RMT_geometric_mean, return_iterates=True)
    )

    # Generating data common to all trials
    SPD_mean = random_SPD(config["n_features"],
                        condition_number=config["condition_number"])
    SPD_matrices = random_points_from_mean(SPD_mean,
                                           config["n_matrices"],
                                           scale=0.1,
                                           mode="exact")
    SPD_mean = SPD_mean
    error_function = riemannian_dist2

    # Pre-generate data
    logging.info("Pre-generating data")
    data_list = []
    for trial in trange(config["n_trials"]):
        data = random_gaussian_data(SPD_matrices,
                                    config["n_samples"])
        data_list.append(data)


    # One monte carlo trial
    def one_montecarlo_trial(trial_no: int, data: np.ndarray) -> dict:
        # Just in case
        np.random.seed(trial_no+config["seed"])

        # Compute estimates of geometric mean
        estimates = dict()
        for name, method in estimation_methods.items():
            _, iterates = method(data)
            errors = []
            for iterate in iterates:
                error = error_function(SPD_mean, iterate)
                errors.append(error)
            estimates[name] = errors
        
        return estimates

    # Monte carlo loop in parallel
    logging.info(f"Starting MonteCarlo loop with {config['n_trials']} trials")
    errors = Parallel(n_jobs=config['n_jobs'])(
            delayed(one_montecarlo_trial)(i, data_list[i])
            for i in trange(config["n_trials"]))
    logging.info("MonteCarlo loop finished")

    # Formatting results into a dataframe
    logging.info("Formatting results")
    max_iterations = max([len(errors[i][method]) 
                        for i, method in product(
                            range(config["n_trials"]),
                            estimation_methods.keys())])
    columns = ["trial", "method", "iteration", "error"]
    df = pd.DataFrame(columns=columns)
    for i, method in product(range(config["n_trials"]),
                            estimation_methods.keys()):
        for iteration in range(max_iterations):
            # complete iterations with NaN
            if iteration >= len(errors[i][method]):
                error = np.nan
            else:
                error = errors[i][method][iteration]
            df.loc[len(df)] = [i, method, iteration, error]

    # Compute mean, 5th and 95th percentiles for each method at each iteration
    # We ignore the NaNs
    df_mean = df.groupby(["method", "iteration"]).mean()
    df_5 = df.groupby(["method", "iteration"]).quantile(0.05)
    df_95 = df.groupby(["method", "iteration"]).quantile(0.95)

    # Drop column of trial number and make a 2D array with column as iteration
    # and row as method
    df_mean = df_mean.drop(columns="trial").unstack(level=1)
    df_5 = df_5.drop(columns="trial").unstack(level=1)
    df_95 = df_95.drop(columns="trial").unstack(level=1)
    logging.info("Results formatted")
    logging.info(f"Mean:\n{df_mean}")
    logging.info(f"5th percentile:\n{df_5}")
    logging.info(f"95th percentile:\n{df_95}")

    # Save results
    logging.info("Saving results")
    df_mean.to_csv(f"{os.path.join(config['results_path'], 'mean.csv')}")
    df_5.to_csv(f"{os.path.join(config['results_path'], '5.csv')}")
    df_95.to_csv(f"{os.path.join(config['results_path'], '95.csv')}")
    df.to_csv(f"{os.path.join(config['results_path'], 'raw.csv')}")
    string = "Results saved in:\n" +\
            f"{os.path.join(config['results_path'], 'mean.csv')}\n" +\
            f"{os.path.join(config['results_path'], '5.csv')}\n" +\
            f"{os.path.join(config['results_path'], '95.csv')}\n" +\
            f"{os.path.join(config['results_path'], 'raw.csv')}"
    logging.info(string)

    # Plotting
    logging.info("Plotting")
    fig, ax = plt.subplots(figsize=(9,6))
    colors = ["b", "g", "r", "c", "m"]
    for i, method in enumerate(estimation_methods.keys()):
        ax.plot(list(df_mean.loc[method]), label=method, color=colors[i])
        ax.fill_between([x[1] for x in list(df_5.loc[method].index)],
                    list(df_5.loc[method]),
                    list(df_95.loc[method]),
                    alpha=0.3, color=colors[i])
    ax.legend(loc="upper right")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE")
    ax.set_title("MSE over iteration for different algorithms")

    # Box to show parameters
    textstr = '\n'.join((
        r'$n_{\mathrm{features}}=%d$' % (config["n_features"], ),
        r'$n_{\mathrm{matrices}}=%d$' % (config["n_matrices"], ),
        r'$n_{\mathrm{samples}}=%d$' % (config["n_samples"], ),
        r'$n_{\mathrm{trials}}=%d$' % (config["n_trials"], ),
        r'$n_{\mathrm{iterations}}=%d$' % (config["n_iterations_max"], ),
        r'$\mathrm{seed}=%d$' % (config["seed"], ),
        r'$\mathrm{condition\ number}=%d$' % (config["condition_number"], )))
    props = dict(boxstyle='round', facecolor='white', alpha=0.2)
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)
    fig.savefig(os.path.join(config["results_path"], "mse_iteration.png"))
    logging.info(f"Plot saved in {os.path.join(config['results_path'], 'mse_iteration.png')}")

    if config["show_plots"]:
        logging.info("Showing plots")
        plt.show()

