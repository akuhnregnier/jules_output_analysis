# -*- coding: utf-8 -*-
from itertools import product
from pathlib import Path

import iris
import matplotlib as mpl
import matplotlib.pyplot as plt
from wildfires.data import homogenise_time_coordinate
from wildfires.utils import ensure_datetime

if __name__ == "__main__":
    mpl.rcdefaults()

    # memory = Memory(".cache", verbose=0)

    fig, axes = plt.subplots(2, 1, squeeze=False, constrained_layout=True)

    # variable = "Gridbox mean burnt area fraction"
    # variable = "frac"
    variable = "lai"
    fig.suptitle(variable)

    for ax, dir_i in zip(axes.ravel(), [2, 3]):
        ax.set_title(dir_i)

        DATA_DIR = Path(f"~/JULES_output/jules_output{dir_i}").expanduser()

        cube = homogenise_time_coordinate(
            iris.load(str(DATA_DIR / "*Monthly*.nc"), constraints=variable)
        ).concatenate_cube()

        dates = [
            ensure_datetime(cube.coord("time").cell(i).point)
            for i in range(len(cube.coord("time").points))
        ]

        # Iterate over different coordinates.
        template = [slice(None)] * len(cube.shape)

        # Select entire time axis.
        assert len(cube.coord_dims("time")) == 1
        time_dims = cube.coord_dims("time")
        time_indices = [[slice(None)]]

        # Selected land-index points.
        spatial_dims = cube.coord_dims("latitude")
        assert cube.coord_dims("latitude") == cube.coord_dims("longitude")
        assert len(cube.coord_dims("latitude")) == 2
        # Generate all possible indices corresponding to these dimensions.
        spatial_shape = [l for i, l in enumerate(cube.shape) if i in spatial_dims]
        spatial_indices = list(product(*(range(l) for l in spatial_shape)))
        # Choose certain spatial_indices.
        land_indices = [100, 200, 500]
        spatial_indices = [spatial_indices[i] for i in land_indices]

        # Generate all possible indices for all remaining coordinates.
        chosen_dims = time_dims + spatial_dims
        remaining_dims = [i for i in range(len(cube.shape)) if i not in chosen_dims]
        remaining_shape = tuple(cube.shape[i] for i in remaining_dims)
        remaining_indices = list(product(*(range(l) for l in remaining_shape)))

        for index_list in product(time_indices, spatial_indices, remaining_indices):
            for coord_dims, coord_index in zip(
                [time_dims, spatial_dims, remaining_dims], index_list
            ):
                for dim, index in zip(coord_dims, coord_index):
                    template[dim] = index

            ax.plot(dates, cube.data[template], label=index_list[1][1])

        ax.legend()
