# -*- coding: utf-8 -*-
import math
import re
from datetime import datetime
from itertools import islice
from pathlib import Path
from pprint import pprint

import iris
import matplotlib.pyplot as plt
import numpy as np
from joblib import Memory

memory = Memory(".cache", verbose=0)

DATA_DIR = Path("~/JULES_output/jules_output2").expanduser()

# Example filename:
# /work/scratch-nopw/alexkr/cruncep_test/jules_output/JULES.CRUNCEPv7SLURM.SPINUP.dump.20080101.0.nc


# @memory.cache
def get_all_cubes(DATA_DIR=DATA_DIR, pattern="*.nc"):
    cube_lists = {}
    for path in DATA_DIR.glob(pattern):
        filename = path.name
        date = datetime(
            *map(int, re.search("(\d{4})(\d{2})(\d{2})", filename).groups())
        )
        cube_lists[date] = iris.load(str(path))

    # Assume all cube lists have the same variables.
    variables = [cube.name() for cube in list(islice(cube_lists.values(), 0, 1))[0]]

    return cube_lists, variables


# @memory.cache
def get_single_variable(variable):
    var_cubes = {
        key: clist.extract_strict(iris.Constraint(name=variable))
        for key, clist in cube_lists.items()
    }
    return var_cubes


if __name__ == "__main__":
    cube_lists, variables = get_all_cubes(pattern="*dump*.nc")

    print("All variables:")
    pprint(variables)

    variable = "lai"

    var_cubes = get_single_variable(variable)

    plot_dates = np.array(list(var_cubes))
    sort_i = np.argsort(plot_dates)
    s_plot_dates = plot_dates[sort_i]

    land_is = (500, 0, 100, 1000, 2000, 3000)

    n_figs = len(land_is)
    nrows = math.floor(n_figs ** 0.5)
    ncols = math.ceil(n_figs / nrows)

    fig, axes = plt.subplots(
        nrows, ncols, squeeze=False, sharex=True, sharey=True, constrained_layout=True
    )
    fig.suptitle(variable)

    single_shape = np.squeeze(list(islice(var_cubes.values(), 0, 1))[0].data).shape

    legend = False
    for ax, land_i in zip(axes.ravel(), land_is):
        if len(single_shape) == 1:
            plot_data_list = [
                np.array([np.squeeze(cube.data)[land_i] for cube in var_cubes.values()])
            ]
            labels = [None]
            # No legend here.
        elif len(single_shape) == 2:
            legend = True
            plot_data_list = []
            labels = []
            for i in range(single_shape[0]):
                plot_data_list.append(
                    np.array(
                        [
                            np.squeeze(cube.data)[i, land_i]
                            for cube in var_cubes.values()
                        ]
                    )
                )
                labels.append(i)
        else:
            raise ValueError(f"Unsupported shape: {single.shape}")

        for plot_data, label in zip(plot_data_list, labels):
            ax.plot(s_plot_dates, plot_data[sort_i], label=label)

        if legend:
            ax.legend()
