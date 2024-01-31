import numpy as np
import numpy.linalg as la
from sklearn.metrics import accuracy_score

from pyriemann.estimation import Covariances
from sklearn.metrics import confusion_matrix

import os
import sys
import yaml
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.classification import KMeans_RMT
from src.utils import parse_args
from data import (
        read_salinas, read_indian_pines,
        read_pavia_center, read_pavia_university,
        read_kennedy_space_center, read_botswana
)
from pipelines_utils import (
        PCAImage, SlidingWindowVectorize,
        KmeansplusplusTransform, LabelsToImage,
        RemoveMeanImage, assign_segmentation_classes_to_gt_classes,
        compute_mIoU
)
from sklearn.pipeline import Pipeline

import argparse
from typing import Tuple, Dict, Callable, List
from numpy import ndarray
import cloudpickle

import matplotlib.pyplot as plt

# Setup logging
import logging
import rich
from rich.logging import RichHandler
FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]",
    handlers=[RichHandler(markup=True)],
)
from tqdm import tqdm, trange


def read_data(dataset_path: str, data_name: str) -> \
        Tuple[ndarray, ndarray, Dict[int, str]]:
    if data_name.lower() == 'salinas':
        return read_salinas(os.path.join(dataset_path, 'salinas'))
    elif data_name.lower() == 'indianpines':
        return read_indian_pines(os.path.join(dataset_path, 'indianpines'))
    elif data_name.lower() in ['pavia', 'paviacenter', 'pavia_center']:
        return read_pavia_center(os.path.join(dataset_path, 'Pavia'))
    elif data_name.lower() in ['paviau', 'paviauniversity', 'pavia_university']:
        return read_pavia_university(os.path.join(dataset_path, 'PaviaU'))
    elif data_name.lower() in ['kennedy', 'kennedyspacecenter', 
                               'kennedy_space_center', 'ksc']:
        return read_kennedy_space_center(os.path.join(dataset_path, 'KSC'))
    elif data_name.lower() == 'botswana':
        return read_botswana(os.path.join(dataset_path, data_name))
    else:
        raise ValueError('Unknown dataset name: {}'.format(data_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Kmeans pipelines on Hyperspectral data')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--data_name', type=str, default=None)
    parser.add_argument('--pca_dim', type=int, default=None)
    parser.add_argument('--window_size', type=int, default=None)
    parser.add_argument('--n_init', type=int, default=None,
                        help='Number of initializations')
    parser.add_argument('--max_iter', type=int, default=None,
                        help='Maximum number of iterations of RMT method')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--skip_zero_class', type=bool, default=None,
                        help='Skip the zero class in the accuracy computation')
    parser.add_argument('--n_jobs', type=int, default=None,
                        help='Number of jobs to run in parallel')
    parser.add_argument('--results_path', type=str, default="results/test",
                        help='Path to store results')
    parser.add_argument('--show_plots', type=bool, default=False,
                        help='Show plots (will block execution)')
    args = parser.parse_args()

    config = parse_args(args, experiment="hyperspectral")
    if config['n_jobs'] is None:
        config['n_jobs'] = 1

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

    # Read data
    logging.info('Reading data...')
    data, labels, labels_name = read_data(config['dataset_path'], config['data_name'])
    n_classes = len(labels_name)

    pipeline_scm = Pipeline([
        ('remove_mean', RemoveMeanImage()),
        ('pca', PCAImage(n_components=config['pca_dim'])),
        ('sliding_window', SlidingWindowVectorize(config['window_size'])),
        ('scm', Covariances('scm')),
        ('kmeans', KmeansplusplusTransform(
            n_clusters=n_classes, n_jobs=config['n_jobs'], n_init=config['n_init'],
            random_state=config['seed'],
            use_plusplus=False, verbose=1, max_iter=config['max_iter'])),
        ('labels_to_image', LabelsToImage(data.shape[0], data.shape[1],
                                        config['window_size']))
        ], verbose=True)

    pipeline_lwf = Pipeline([
        ('remove_mean', RemoveMeanImage()),
        ('pca', PCAImage(n_components=config['pca_dim'])),
        ('sliding_window', SlidingWindowVectorize(config['window_size'])),
        ('lwf', Covariances('lwf')),
        ('kmeans', KmeansplusplusTransform(
            n_clusters=n_classes, n_jobs=config['n_jobs'], n_init=config['n_init'],
            random_state=config['seed'], 
            use_plusplus=False, verbose=1, max_iter=config['max_iter'])),
        ('labels_to_image', LabelsToImage(data.shape[0], data.shape[1],
                                        config['window_size']))
        ], verbose=True)


    pipeline_rmt = Pipeline([
        ('remove_mean', RemoveMeanImage()),
        ('pca', PCAImage(n_components=config['pca_dim'])),
        ('sliding_window', SlidingWindowVectorize(config['window_size'])),
        ('kmeans', KMeans_RMT(n_clusters=n_classes, n_jobs=config['n_jobs'],
                            random_state=config['seed'], 
                            n_init=config['n_init'], verbose=1, max_iter=config['max_iter'])),
        ('labels_to_image', LabelsToImage(data.shape[0], data.shape[1],
                                        config['window_size']))
        ], verbose=True)


    pipelines = {
            "SCM-Kmeans": pipeline_scm,
            "RMT-Kmeans": pipeline_rmt,
            "LWF-Kmeans": pipeline_lwf
    }
    results = {}
    images = {}
    accuracies = {}
    ious = {}
    mious = {}

    for name, pipeline in pipelines.items():
        logging.info(f'Fitting and transforming {name}...')
        result = pipeline.fit_transform(data, labels)
        results[name] = result
        logging.info(f'Finished {name}.')

        logging.info(f'Computing accuracy for {name}...')

        # Padding to take into account the window size
        result_final = np.ones_like(data[:, :, 0])*np.nan
        pad_x = (data.shape[0] - result.shape[0])//2
        pad_y = (data.shape[1] - result.shape[1])//2
        result_final[pad_x:-pad_x, pad_y:-pad_y] = result

        # Match labels 
        result_final = assign_segmentation_classes_to_gt_classes(result_final, labels)

        # Replace class 0 with ground truth since no data is available
        if config['skip_zero_class']:
            mask = np.logical_and(np.logical_not(np.isnan(result_final)), labels==0)
            result_final[mask] = 0
            mask = np.logical_and(labels!=0, ~np.isnan(result_final))
        else:
            mask = ~np.isnan(result_final)
        accuracy = accuracy_score(labels[mask],
                                result_final[mask])
        logging.info(f'Overall accuracy {name}: {accuracy:.2f}')
        ious[name], mious[name] = compute_mIoU(result_final, labels)
        logging.info(f'mIoU {name}: {mious[name]}')
        logging.info(f'IoU {name}: {ious[name]}')

        images[name] = result_final
        accuracies[name] = accuracy

    # Store results
    toStore = {
            'config': config,
            'pipelines': pipelines,
            'results': results,
            'images': images,
            'accuracies': accuracies,
            'ious': ious,
            'mious': mious,
            'pipelines': pipelines
        }
    if config['results_path'] is not None:
        with open(os.path.join(config['results_path'], 'results.pkl'), 'wb') as f:
            cloudpickle.dump(toStore, f)
            logging.info(f'Results stored in {config["results_path"]}/results.pkl')

    # Show results
    fig = plt.figure(figsize=(7, 6))
    plt.imshow(labels, aspect='auto', cmap='tab20')
    plt.title('Ground truth')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(config['results_path'], 'gt.pdf'))

    for i, name in enumerate(pipelines.keys()):
        fig = plt.figure(figsize=(13, 5))
        plt.imshow(images[name], aspect='auto', cmap='tab20')
        if name in accuracies:
            accuracy = accuracies[name]
            mIoU = mious[name]
            iou = ious[name]
            print(name, accuracy, mIoU)
            plt.title(f'{name} (acc={accuracy:.2f}, mIoU={mIoU:.2f}')
        else:
            plt.title(f'{name}')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(config['results_path'], f'{name}.pdf'))
    
    if config['show_plots']:
        logging.info("Showing plots")
        plt.show()
