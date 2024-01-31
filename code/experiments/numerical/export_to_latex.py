import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
    parser = argparse.ArgumentParser("Export mean, q5, q95 results to pgfplots fillbetween plot")
    parser.add_argument("mean", help="Input mean CSV file")
    parser.add_argument("q5", help="Input q5 CSV file")
    parser.add_argument("q95", help="Input q95 CSV file")
    parser.add_argument("output", help="Output pgfplots file")
    parser.add_argument("--xlabel", help="X-axis label", default=r"$x$")
    parser.add_argument("--ylabel", help="Y-axis label", default=r"$y$")
    parser.add_argument("--xscale", help="X-axis scale", default="log")
    parser.add_argument("--yscale", help="Y-axis scale", default="log")
    args = parser.parse_args()

    # Read CSV files
    mean = pd.read_csv(args.mean)
    q5 = pd.read_csv(args.q5)
    q95 = pd.read_csv(args.q95)

    # Parse methods 
    parse_methods = list(mean.columns[1:])
    for i in range(len(parse_methods)):
        if '_' in parse_methods[i]:
            parse_methods[i] = parse_methods[i].replace('_', ' ')
    mean.columns = [mean.columns[0]] + parse_methods
    q5.columns = [q5.columns[0]] + parse_methods
    q95.columns = [q95.columns[0]] + parse_methods
    # Parse x_values
    x_values = mean.iloc[:, 0].to_numpy()

    # Plot
    fig, ax = plt.subplots()
    for method in parse_methods:
        ax.fill_between(x_values, q5[method], q95[method], alpha=0.2)
        ax.plot(x_values, mean[method], label=method)
    ax.legend()
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    ax.set_xscale(args.xscale)
    ax.set_yscale(args.yscale)
    tikzplotlib_fix_ncols(ax)
    save = tikz_save(args.output)
    print(f"Saved plot to {args.output}")

