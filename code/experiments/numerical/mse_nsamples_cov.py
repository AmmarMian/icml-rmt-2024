# Experiment file for comparing different algorithms of estimating covariance
# MSE over the number of samples

import os
import sys
import yaml
import argparse
import time

import numpy as np
import pandas as pd

from tqdm import tqdm
from itertools import product
from functools import partial
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
from src.mean import RMT_estimation_cov
from src.covariance import analytical_shrinkage_estimator
from sklearn.covariance import EmpiricalCovariance, LedoitWolf, OAS

import matplotlib.pyplot as plt
import tikzplotlib
# Activate LaTex rendering for matplotlib
# import matplotlib
# matplotlib.rcParams['text.usetex'] = True

# # Styling matplotlib
# plt.style.use('dark_background')
# matplotlib.rcParams['axes.spines.right'] = False
# matplotlib.rcParams['axes.spines.top'] = False
# matplotlib.rcParams['axes.grid'] = True
# matplotlib.rcParams['grid.alpha'] = 0.15
# matplotlib.rcParams['grid.linestyle'] = 'dotted'
# matplotlib.rcParams['axes.facecolor'] = (0.1, 0.1, 0.1, 0.4)
# matplotlib.rcParams['font.family'] = 'serif'
# # matplotlib.rcParams['font.serif'] = 'Times New Roman'
# matplotlib.rcParams['font.size'] = 11


def fromsklearnclass_tofunction(sklearn_class):
    def function(X):
        sklearn_object = sklearn_class()
        sklearn_object.fit(X)
        return sklearn_object.covariance_
    return function


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)

if __name__ == "__main__":

    # Define parser
    parser = argparse.ArgumentParser(
            description="Plot the MSE over the number of matrices used."
            " The options provided will overwrite the config file.")
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--n_features', type=int, default=None,
                        help='Number of features')
    parser.add_argument('--samples_power_end', type=float, default=None,
                        help='Max power of number of features for n_samples')
    parser.add_argument('--n_trials', type=int, default=None,
                        help='Number of MonteCarlo trials')
    parser.add_argument('--n_points', type=int, default=None,
                        help='Number of points in n_samples list')
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

    config = parse_args(args, experiment="mse_nsamples_cov")

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

    estimation_methods = dict(
            SCM = fromsklearnclass_tofunction(EmpiricalCovariance),
            LW_linear = fromsklearnclass_tofunction(LedoitWolf),
            OAS = fromsklearnclass_tofunction(OAS),
            RMT = partial(RMT_estimation_cov,
                  max_iterations=config["n_iterations_max"]),
            LW_nonlinear = partial(analytical_shrinkage_estimator, shrink=0),
            )

    error_function = riemannian_dist2
    n_samples_list = np.unique(
            np.logspace(1, config['samples_power_end'],
                        config['n_points'],
                        base=config["n_features"]).astype(int))
    iterator = product(range(config["n_trials"]), n_samples_list)

    # One monte carlo trial
    SPD_matrix = random_SPD(config["n_features"],
                          condition_number=config["condition_number"])
    def one_montecarlo_trial(trial_no: int, 
                            n_samples: int) -> dict:
        # Just in case
        np.random.seed(trial_no+config["seed"]+n_samples)

        # Generate data
        data = random_gaussian_data(SPD_matrix.reshape((1, config["n_features"],
                                                        config["n_features"])),
                                    n_samples=n_samples).squeeze()

        # Compute estimates of SPD matrix
        estimates = {
            'n_samples': n_samples,
            'trial_no': trial_no
        }
        for name, method in estimation_methods.items():
            estimated = method(data)
            estimates[name] = error_function(SPD_matrix, estimated)
        
        return estimates

    # Monte carlo loop in parallel
    logging.info(f"Starting MonteCarlo loop with {config['n_trials']} trials")
    errors = Parallel(n_jobs=config['n_jobs'])(
            delayed(one_montecarlo_trial)(trial_no, n_samples)
            for trial_no, n_samples
            in tqdm(iterator, total=config['n_trials']*len(n_samples_list)))
    logging.info("MonteCarlo loop finished")

    # Formatting results into a dataframe
    logging.info("Formatting results")
    df = pd.DataFrame(errors)
    df_mean = df.groupby(["n_samples"]).mean().drop(columns=["trial_no"])
    df_5 = df.groupby(["n_samples"]).quantile(0.05).drop(columns=["trial_no"])
    df_95 = df.groupby(["n_samples"]).quantile(0.95).drop(columns=["trial_no"])

    # logging.info("Saving results")
    # df_mean.to_csv(os.path.join(config["results_path"], "mean.csv"))
    # df_5.to_csv(os.path.join(config["results_path"], "5.csv"))
    # df_95.to_csv(os.path.join(config["results_path"], "95.csv"))
    # logging.info("Mean:\n" + str(df_mean))

    # Plotting results
    logging.info("Plotting results")
    fig, ax = plt.subplots(figsize=(8,4))
    colors = ['red', 'blue', 'green', 'orange', 'black']
    for i, method in enumerate(estimation_methods.keys()):
        ax.plot(n_samples_list, df_mean[method], color=colors[i], label=method)
        ax.fill_between(n_samples_list, df_5[method], df_95[method],
                        color=colors[i], alpha=0.2)
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend(loc='upper right')

    # # Box  to show parameters
    # textstr = '\n'.join((
    #     r'$n_{\mathrm{features}}=%d$' % (config['n_features'], ),
    #     r'$n_{\mathrm{trials}}=%d$' % (config['n_trials'], ),
    #     r'$n_{\mathrm{iterations}}=%d$' % (config['n_iterations_max'], ),
    #     r'$\mathrm{seed}=%d$' % (config['seed'], ),
    #     r'$\mathrm{condition\ number}=%d$' % (config['condition_number'], )))
    # props = dict(boxstyle='round', facecolor='white', alpha=0.25)
    # ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=10,
    #         verticalalignment='bottom', bbox=props)
    ax = tikzplotlib_fix_ncols(ax)
    tikzplotlib.save(os.path.join(config["results_path"], "plot.tex"))

    plt.tight_layout()
    plt.savefig(os.path.join(config["results_path"], "plot.pdf"))
    logging.info(f"Succesfully saved results in {config['results_path']}")

    if config["show_plots"]:
        logging.info("Showing plots")
        plt.show()
