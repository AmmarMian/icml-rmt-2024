# Experiment file to show when RMT estimation of covariance fails compared
# to SCM

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
from src.mean import RMT_estimation_cov
from sklearn.covariance import EmpiricalCovariance

import matplotlib.pyplot as plt
from tikzplotlib import save as tikz_save

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
            description="Plot the MSE over the iteration number for several"
            " algorithms. The options provided will overwrite the config file.")
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--n_features', type=int, default=None,
                        help='Number of features')
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Number of samples')
    parser.add_argument('--n_trials', type=int, default=None,
                        help='Number of MonteCarlo trials')
    parser.add_argument('--n_iterations_max', type=int, default=None,
                        help='Number of iterations max for RMT algorithm')
    parser.add_argument('--tol', type=float, default=-np.inf,
                        help='Tolerance for RMT algorithm')
    parser.add_argument('--tol_cost', type=float, default=-np.inf,
                        help='Cost tolerance for RMT algorithm')
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
    SCM = EmpiricalCovariance()
    RMT_fail = partial(RMT_estimation_cov,
                    max_iterations=config['n_iterations_max'],
                    tol=config['tol'], tol_cost=config['tol_cost'],
                    return_iterates=True)

    # Generating data common to all trials
    SPD_matrix = random_SPD(config["n_features"],
                        condition_number=config["condition_number"])
    error_function = riemannian_dist2

    # One monte carlo trial
    def one_montecarlo_trial(trial_no: int) -> dict:
        # Just in case
        np.random.seed(trial_no+config["seed"])
        data = random_gaussian_data(
                SPD_matrix.reshape((1, config['n_features'], config['n_features'])),
                config["n_samples"]).squeeze()
        SCM_estimate = SCM.fit(data).covariance_
        _, rmt_iterates = RMT_fail(data)
        data = dict(errors_scm = [], mse_rmt = [], trial_no=trial_no)
        for i, RMT_fail_estimate in enumerate(rmt_iterates):
            error_scm = error_function(SCM_estimate, RMT_fail_estimate)
            data["errors_scm"].append(error_scm)
            data["mse_rmt"].append(error_function(SPD_matrix, RMT_fail_estimate))
        return data

    # Monte carlo loop in parallel
    logging.info(f"Starting MonteCarlo loop with {config['n_trials']} trials")
    errors = Parallel(n_jobs=config['n_jobs'])(
            delayed(one_montecarlo_trial)(i)
            for i in trange(config["n_trials"]))
    logging.info("MonteCarlo loop finished")

    # Formatting results into a dataframe
    logging.info("Formatting results")
    max_iterations = max([len(errors[i]["errors_scm"]) 
                        for i in range(config["n_trials"])])
    errors_scm = np.nan*np.ones((config['n_trials'], max_iterations))
    mse_rmt = np.nan*np.ones((config['n_trials'], max_iterations))
    for i in range(config["n_trials"]):
        len_errors = len(errors[i]["errors_scm"])
        errors_scm[i, :len_errors] = errors[i]["errors_scm"]
        mse_rmt[i, :len_errors] = errors[i]["mse_rmt"]
    df_errors_scm = pd.DataFrame(errors_scm)
    df_errors_scm = df_errors_scm.melt(var_name="iteration", value_name="error")
    df_mse_rmt = pd.DataFrame(mse_rmt)
    df_mse_rmt = df_mse_rmt.melt(var_name="iteration", value_name="error")

    # Compute mean, 5th and 95th percentiles for each method at each iteration
    df_errors_scm_mean = df_errors_scm.groupby(["iteration"]).mean()
    df_errors_scm_5 = df_errors_scm.groupby(["iteration"]).quantile(0.05)
    df_errors_scm_95 = df_errors_scm.groupby(["iteration"]).quantile(0.95)
    df_mse_rmt_mean = df_mse_rmt.groupby(["iteration"]).mean()
    df_mse_rmt_5 = df_mse_rmt.groupby(["iteration"]).quantile(0.05)
    df_mse_rmt_95 = df_mse_rmt.groupby(["iteration"]).quantile(0.95)
    logging.info("Results formatted")
    logging.info(f"Mean:\n{df_errors_scm_mean}\n{df_mse_rmt_mean}")

    # Save results
    logging.info("Saving results")
    df_errors_scm_mean.to_csv(f"{os.path.join(config['results_path'], 'mean_err_scm.csv')}")
    df_errors_scm_5.to_csv(f"{os.path.join(config['results_path'], '5_err_scm.csv')}")
    df_errors_scm_95.to_csv(f"{os.path.join(config['results_path'], '95_err_scm.csv')}")
    df_mse_rmt_mean.to_csv(f"{os.path.join(config['results_path'], 'mean_mse_rmt.csv')}")
    df_mse_rmt_5.to_csv(f"{os.path.join(config['results_path'], '5_mse_rmt.csv')}")
    df_mse_rmt_95.to_csv(f"{os.path.join(config['results_path'], '95_mse_rmt.csv')}")

    string = "Results saved in:\n" +\
            f"{os.path.join(config['results_path'], 'mean_err_scm.csv')}\n" +\
            f"{os.path.join(config['results_path'], '5_err_scm.csv')}\n" +\
            f"{os.path.join(config['results_path'], '95_err_scm.csv')}\n" +\
            f"{os.path.join(config['results_path'], 'mean_mse_rmt.csv')}\n" +\
            f"{os.path.join(config['results_path'], '5_mse_rmt.csv')}\n" +\
            f"{os.path.join(config['results_path'], '95_mse_rmt.csv')}"


    # Plotting
    logging.info("Plotting")
    fig, ax = plt.subplots(figsize=(9,6))

    # Plotting error on SCM
    ax.plot(list(df_errors_scm_mean.index), list(df_errors_scm_mean["error"]),
            label=r"\hat{\delta}^2(SCM, RMT)", color="black")
    ax.fill_between(list(df_errors_scm_5.index),
                    list(df_errors_scm_5["error"]),
                    list(df_errors_scm_95["error"]),
                    alpha=0.2, color="black")
    
    # Plotting MSE on RMT
    ax.plot(list(df_mse_rmt_mean.index), list(df_mse_rmt_mean["error"]),
            label=r"\hat{\delta}^2(RMT, \Sigma)", color="red")
    ax.fill_between(list(df_mse_rmt_5.index),
                    list(df_mse_rmt_5["error"]),
                    list(df_mse_rmt_95["error"]),
                    alpha=0.2, color="red")
    ax.legend(loc="upper right")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("distance")
    plt.tight_layout()

    fig.savefig(os.path.join(config["results_path"], "rmt_fail.pdf"))
    logging.info(f"Plot saved in {os.path.join(config['results_path'], 'rmt_fail.pdf')}")
    ax = tikzplotlib_fix_ncols(ax)
    tikz_save(os.path.join(config["results_path"], "rmt_fail.tex"))
    logging.info(f"Plot saved in {os.path.join(config['results_path'], 'rmt_fail.tex')}")

    if config["show_plots"]:
        logging.info("Showing plots")
        plt.show()

