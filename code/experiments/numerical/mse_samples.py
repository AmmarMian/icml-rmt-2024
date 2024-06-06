# Experiment file for comparing different algorithms of computing the Fréchet
# mean over SPD  matrices. The file produce a visualization of the MSE with a
# given n_samples and n_features

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
from src.mean import geometric_mean, RMT_geometric_mean

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
    parser.add_argument('--n_matrices', type=str, default=None,
                        help='Number of matrices') 
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Number of samples separated by commas')
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

    if isinstance(args.n_samples, str):
        args.n_samples = [int(n) for n in args.n_samples.split(",")]
    config = parse_args(args, experiment="mse_nsamples")

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

    # One monte carlo trial
    SPD_mean = random_SPD(config["n_features"],
                          condition_number=config["condition_number"])
    def one_montecarlo_trial(trial_no: int, 
                            n_matrices: int,
                            n_samples: int) -> dict:
        # Just in case
        np.random.seed(trial_no+config["seed"]+n_samples)

        # Generate data
        SPD_matrices = random_points_from_mean(SPD_mean,
                                            n_matrices,
                                            scale=0.1,
                                            mode="exact")
        data = random_gaussian_data(SPD_matrices, n_samples)

        # Compute estimates of geometric mean
        estimates = {
            'n_samples': n_samples,
            'trial_no': trial_no
        }
        for name, method in estimation_methods.items():
            estimated_mean = method(data)
            estimates[name] = error_function(SPD_mean, estimated_mean)
        
        return estimates

    # Monte carlo loop in parallel
    logging.info(f"Starting MonteCarlo loop with {config['n_trials']} trials")
    iterator = product(range(config['n_trials']), config['n_samples'])
    errors = Parallel(n_jobs=config['n_jobs'])(
            delayed(one_montecarlo_trial)(trial_no,
                                          config['n_matrices'],
                                          n_samples)
            for trial_no, n_samples in
            tqdm(iterator))
    logging.info("MonteCarlo loop finished")

    # Formatting results into a dataframe
    logging.info("Formatting results")
    df = pd.DataFrame(errors)
    df_mean = df.groupby(["n_samples"]).mean().drop(columns=["trial_no"])
    df_5 = df.groupby(["n_samples"]).quantile(0.05).drop(columns=["trial_no"])
    df_50 = df.groupby(["n_samples"]).quantile(0.5).drop(columns=["trial_no"])
    df_95 = df.groupby(["n_samples"]).quantile(0.95).drop(columns=["trial_no"])

    logging.info("Saving results")
    df_mean.to_csv(os.path.join(config["results_path"], "mean.csv"))
    df_5.to_csv(os.path.join(config["results_path"], "5.csv"))
    df_50.to_csv(os.path.join(config["results_path"], "50.csv"))
    df_95.to_csv(os.path.join(config["results_path"], "95.csv"))
    logging.info("Mean:\n" + str(df_mean))

    # Plotting results
    logging.info("Plotting results")
    fig, ax = plt.subplots(figsize=(8,4))
    colors = ['red', 'blue', 'green', 'orange', 'black']
    for i, method in enumerate(estimation_methods.keys()):
        ax.plot(df_50[method], color=colors[i], label=method)
        ax.fill_between(df_50.index, df_5[method], df_95[method],
                        color=colors[i], alpha=0.2)
    ax.set_xlabel("Number of matrices")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend(loc='upper right')
    ax = tikzplotlib_fix_ncols(ax)

    # # Box  to show parameters
    # textstr = '\n'.join((
    #     r'$n_{\mathrm{features}}=%d$' % (config['n_features'], ),
    #     r'$n_{\mathrm{samples}}=%d$' % (config['n_samples'], ),
    #     r'$n_{\mathrm{trials}}=%d$' % (config['n_trials'], ),
    #     r'$n_{\mathrm{iterations}}=%d$' % (config['n_iterations_max'], ),
    #     r'$\mathrm{seed}=%d$' % (config['seed'], ),
    #     r'$\mathrm{condition\ number}=%d$' % (config['condition_number'], )))
    # props = dict(boxstyle='round', facecolor='white', alpha=0.25)
    # ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=10,
    #         verticalalignment='bottom', bbox=props)

    plt.tight_layout()
    plt.savefig(os.path.join(config["results_path"], "plot.pdf"))
    tikzplotlib.save(os.path.join(config["results_path"], "plot.tex"))
    logging.info(f"Succesfully saved results in {config['results_path']}")

    if config["show_plots"]:
        logging.info("Showing plots")
        plt.show()
