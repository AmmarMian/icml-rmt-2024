import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.estimation import Covariances_LWnonlinear
from src.classification import MDM_RMT

import matplotlib.pyplot as plt
import seaborn as sns
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


import argparse
import pickle

from moabb import setup_seed


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run a BCI benchmark.")
    parser.add_argument("--paradigm", type=str, default="mi",
                        help="Paradigm to use. (lr, mi)")
    parser.add_argument("--tmin", type=float, default=None,
                        help="Start time in seconds.")
    parser.add_argument("--tmax", type=float, default=None,
                        help="End time in seconds.")
    parser.add_argument("--resample", type=int, default=None,
                        help="Resample rate in Hz.")
    parser.add_argument("--pipelines", type=str, default=None,
                        help="Pipelines to use separated by a comma.")
    parser.add_argument("--dataset", type=str, default="bnci_2014_001",
                        help="Dataset to use. (bnci_2014_001, physionet, schirrmeister2017, munich_mi, cho2017, weibo2014, zhou2016, lee2019, ofner2017, khushaba2017, lee2019_mi)")
    parser.add_argument("--subject", type=int, default=None,
                        nargs='+', help="Subject(s) to use.")
    parser.add_argument('--remove_subject', type=str, default=None,
                        help="Subject(s) to remove.")
    parser.add_argument("--dataset_path", type=str, default='./data/mne_data',
                        help="Path to read/download the dataset.")
    parser.add_argument("--results_path", type=str, default='results/test',
                        help="Path to store the results.")
    parser.add_argument("--seed", type=float, default=42,
                        help="Random seed.")
    args = parser.parse_args()

    setup_seed(int(args.seed))

    # Parse remove_subject
    if args.remove_subject is not None:
        args.remove_subject = [int(x) for x in args.remove_subject.split(",")]

    # Parse pipelines
    if args.pipelines is not None:
        args.pipelines = [x.strip() for x in args.pipelines.split(",")]

    # Set download and storage directory
    print("Setting Dataset path to: {}".format(args.dataset_path))
    os.environ['MNE_DATA'] = args.dataset_path
    os.environ['MOABB_RESULTS'] = args.dataset_path
    import moabb
    from moabb.utils import set_download_dir
    from moabb.datasets import(
        BNCI2014_001, PhysionetMI, Schirrmeister2017, MunichMI,
        Cho2017, Weibo2014, Zhou2016, Lee2019, Ofner2017, 
        Lee2019_MI
    )
    from moabb.evaluations import CrossSessionEvaluation, WithinSessionEvaluation
    from moabb.paradigms import LeftRightImagery, MotorImagery
    set_download_dir(args.dataset_path)
    moabb.set_log_level("info")
    def parse_dataset_string(dataset: str):
        if dataset == "bnci_2014_001":
            dataset = BNCI2014_001()
        elif dataset == "physionet":
            dataset = PhysionetMI()
        elif dataset == "schirrmeister2017":
            dataset = Schirrmeister2017()
        elif dataset == "munich_mi":
            dataset = MunichMI()
        elif dataset == "cho2017":
            dataset = Cho2017()
        elif dataset == "weibo2014":
            dataset = Weibo2014()
        elif dataset == "zhou2016":
            dataset = Zhou2016()
        elif dataset == "lee2019":
            dataset = Lee2019()
        elif dataset == "ofner2017":
            dataset = Ofner2017()
        elif dataset == "lee2019_mi":
            dataset = Lee2019_MI()
        else:
            raise ValueError("Dataset not found.")
        return dataset


    # Put params in a dict
    # if any([x is None for x in [args.tmin, args.tmax, args.resample]]):
    #     paradigm_params = {}
    # else:
    #     paradigm_params = {
    #             'tmin': args.tmin,
    #             'tmax': args.tmax,
    #             'resample': args.resample
    #             }
    paradigm_params = {}
    for x, y in zip(['tmin', 'tmax', 'resample'], [args.tmin, args.tmax, args.resample]):
        if y is not None:
            paradigm_params[x] = y

    if isinstance(args.subject, int):
        args.subject = [args.subject]

    if isinstance(args.remove_subject, int):
        args.remove_subject = [args.remove_subject]

    print("Config:")
    print(vars(args))

    ##############################################################################
    # Create Pipelines
    # ----------------
    #
    # Pipelines must be a dict of sklearn pipeline transformer.
    #
    # The CSP implementation is based on the MNE implementation. We selected 8 CSP
    # components, as usually done in the literature.
    #
    # The Riemannian geometry pipeline consists in covariance estimation, tangent
    # space mapping and finally a logistic regression for the classification.

    pipelines = {}

    pipelines["CSP+LDA"] = make_pipeline(CSP(n_components=8), LDA())

    pipelines["RG+LR"] = make_pipeline(
        Covariances(), TangentSpace(), LogisticRegression(solver="lbfgs")
    )

    pipelines["MDM+SCM"] = make_pipeline(
        Covariances(), MDM()
    )

    pipelines["MDM+SCM-control"] = make_pipeline(
        Covariances(), MDM()
    )

    pipelines["MDM+LW-L"] = make_pipeline(
        Covariances(estimator='lwf'), MDM()
    )

    pipelines["MDM+LW-NL"] = make_pipeline(
        Covariances_LWnonlinear(), MDM()
    )

    pipelines["MDM-RMT"] = make_pipeline(
        MDM_RMT()
    )
    
    pipelines_new = {}
    if args.pipelines is not None:
        for key in pipelines.keys():
            if key in args.pipelines:
                pipelines_new[key] = pipelines[key]
    else:
        pipelines_new = pipelines
    pipelines = pipelines_new

    print("Pipelines:")
    for key in pipelines.keys():
        print("  - {}".format(key))

    ##############################################################################
    # Evaluation
    # ----------
    #
    # We define the paradigm (LeftRightImagery) and the dataset (BNCI2014_001).
    # The evaluation will return a DataFrame containing a single AUC score for
    # each subject / session of the dataset, and for each pipeline.
    #
    # Results are saved into the database, so that if you add a new pipeline, it
    # will not run again the evaluation unless a parameter has changed. Results can
    # be overwritten if necessary.

    if args.paradigm == "lr":
        paradigm = LeftRightImagery(**paradigm_params)
    elif args.paradigm == "mi":
        paradigm = MotorImagery(**paradigm_params)
    else:
        raise ValueError("Paradigm not found.")

    dataset = parse_dataset_string(args.dataset.lower())
    if args.subject is not None:
        dataset.subject_list = args.subject
    if args.remove_subject is not None:
        dataset.subject_list = [x for x in dataset.subject_list
                                if x not in args.remove_subject]
    datasets = [dataset]

    overwrite = True  # set to True if we want to overwrite cached results
    evaluation = WithinSessionEvaluation(
        paradigm=paradigm, datasets=datasets, overwrite=overwrite
    ) # suffix="examples"

    results = evaluation.process(pipelines)

    print(results.head())

    ##############################################################################
    # Plot Results
    # ----------------
    #
    # Here we plot the results. We first make a pointplot with the average
    # performance of each pipeline across session and subjects.
    # The second plot is a paired scatter plot. Each point representing the score
    # of a single session. An algorithm will outperform another is most of the
    # points are in its quadrant.

    # fig, axes = plt.subplots(1, 2, figsize=[8, 4], sharey=True)
    fig, axes = plt.subplots(facecolor="white", figsize=[8, 4])

    sns.stripplot(
        data=results,
        y="score",
        x="pipeline",
        ax=axes,
        jitter=True,
        alpha=0.5,
        zorder=1,
        palette="Set1",
    )
    sns.pointplot(data=results, y="score", x="pipeline", ax=axes, palette="Set1")

    axes.set_ylabel("ROC AUC")
    axes.set_ylim(0.2, 1)

    # paired = results.pivot_table(
    #     values="score", columns="pipeline", index=["subject", "session"]
    # )
    # paired = paired.reset_index()

    # sns.regplot(data=paired, y="RG+LR", x="CSP+LDA", ax=axes[1], fit_reg=False)
    # axes[1].plot([0, 1], [0, 1], ls="--", c="k")
    # axes[1].set_xlim(0.5, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(args.results_path, "results.png"))
    with open(os.path.join(args.results_path, "results.pkl"), "wb") as f:
        pickle.dump(results, f)

    print("Results saved to {}".format(args.results_path))
    # plt.show()
