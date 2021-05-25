# -*- coding: utf-8 -*-
from functools import partial
from itertools import product
from pathlib import Path

import iris
import numpy as np
from wildfires.data import (
    MonthlyDataset,
    dummy_lat_lon_cube,
    homogenise_time_coordinate,
    regrid,
)
from wildfires.utils import get_centres, get_land_mask, match_shape

from .utils import find_min_error, get_1d_to_2d_indices

dummy_lat_lon_cube = partial(dummy_lat_lon_cube, monthly=True)

data_dir = Path("~/JULES_data").expanduser()

n96e_lats = get_centres(np.linspace(-90, 90, 145))
n96e_lons = get_centres(np.linspace(-180, 180, 193))


def regrid_to_n96e(cube):
    return regrid(
        cube,
        new_latitudes=n96e_lats,
        new_longitudes=n96e_lons,
    )


def get_n96e_land_mask():
    land_mask = get_land_mask(n_lon=2000)
    n96e_land_mask = regrid_to_n96e(dummy_lat_lon_cube(land_mask)).data
    return n96e_land_mask


def get_climatology_cube(cube):
    """Calculate the climatology for the given cube."""
    # Create a new Dataset for each cube.
    proc_inst = type(
        cube.long_name.replace(" ", "") if cube.long_name is not None else "dummy",
        (MonthlyDataset,),
        {
            "__init__": lambda self: None,
            "frequency": "monthly",
        },
    )()

    proc_inst.cubes = iris.cube.CubeList([cube])

    # Ensure data is C-contiguous.
    proc_inst.cube.data = np.ma.MaskedArray(
        data=np.ascontiguousarray(proc_inst.cube.data.data),
        mask=np.ascontiguousarray(proc_inst.cube.data.mask),
    )

    return proc_inst.get_climatology_dataset(
        proc_inst.min_time, proc_inst.max_time
    ).cube


def cube_1d_to_2d(cube, temporal_dim="time", latitudes=None, longitudes=None):
    """Convert JULES output on 1D grid to 2D grid.

    the output `iris.cube.Cube` will be on an N96e grid with latitudes `n96e_lats` and
    longitudes `n96e_lons`.

    """
    land_grid_coord = -1  # The last axis is associated with the spatial domain.

    if latitudes is not None or longitudes is not None:
        assert latitudes is not None and longitudes is not None
        orig_lats = latitudes
        orig_lons = longitudes
    else:
        orig_lats = cube.coord("latitude").points.data.ravel()
        orig_lons = cube.coord("longitude").points.data.ravel()

    lat_step = np.unique(np.diff(np.sort(orig_lats)))[1]
    lon_step = np.unique(np.diff(np.sort(orig_lons)))[1]

    # Use the latitude and longitude steps from above to determine the number of
    # latitude and longitude steps.
    n_lat = round(180 / lat_step)
    n_lon = round(360 / lon_step)

    # Ensure that these represent a regular grid.
    assert np.isclose(n_lat * lat_step, 180)
    assert np.isclose(n_lon * lon_step, 360)

    # Create a grid of ..., lat, lon points to match the shape of the given cube.
    new_shape = tuple(list(cube.shape[:land_grid_coord]) + [n_lat, n_lon])

    # Now convert the 1D data to the 2D array created above, using a mask.

    # Pick the grid configuration that best matches the existing grid.
    grid_lats1 = np.linspace(-90, 90, n_lat, endpoint=False)
    grid_lons1 = np.linspace(0, 360, n_lon, endpoint=False)
    grid_lats2 = get_centres(np.linspace(-90, 90, n_lat + 1, endpoint=True))
    grid_lons2 = get_centres(np.linspace(0, 360, n_lon + 1, endpoint=True))

    unique_orig_lats = np.unique(orig_lats)
    unique_orig_lons = np.unique(orig_lons)

    if find_min_error(grid_lats1, unique_orig_lats) < find_min_error(
        grid_lats2, unique_orig_lats
    ):
        grid_lats = grid_lats1
    else:
        grid_lats = grid_lats2

    if find_min_error(grid_lons1, unique_orig_lons) < find_min_error(
        grid_lons2, unique_orig_lons
    ):
        grid_lons = grid_lons1
    else:
        grid_lons = grid_lons2

    indices_1d_to_2d = get_1d_to_2d_indices(
        orig_lats,
        orig_lons,
        grid_lats,
        grid_lons,
    )

    if len(np.squeeze(cube.data).shape) == 1:
        # Simply assign based on the mask.
        new_data = np.ma.MaskedArray(
            np.zeros((n_lat, n_lon), dtype=np.float64), mask=True
        )
        new_data[indices_1d_to_2d] = np.squeeze(cube.data)
    elif len(np.squeeze(cube.data).shape) > 1:
        # Iterate over earlier dimensions.
        new_data = np.ma.MaskedArray(np.zeros(new_shape, dtype=np.float64), mask=True)
        for indices in product(*(range(s) for s in cube.shape[:-1])):
            sel = (*indices, slice(None))
            new_data[sel][indices_1d_to_2d] = cube.data[sel]
    else:
        raise ValueError(f"Invalid cube shape {cube.shape}")

    # XXX: Assumes more than 1 lat, lon coord, and destroys other dimensions (except
    # time, if applicable).
    new_data = np.squeeze(new_data)

    if (cube.coords(temporal_dim) and cube.coord_dims(temporal_dim)) and (
        cube.coord_dims(temporal_dim) == (0,) and cube.shape[0] == 1
    ):
        # Reinstate the time dimension that was removed in the previous step.
        new_data = new_data[None]

    n_dim = len(new_data.shape)
    lat_dim = n_dim - 2
    lon_dim = n_dim - 1

    time_coord_and_dims = []
    if cube.coords(temporal_dim) and cube.coord_dims(temporal_dim):
        time_coord_and_dims.append(
            (cube.coord(temporal_dim), cube.coord_dims(temporal_dim)[0])
        )

    new_cube = iris.cube.Cube(
        new_data,
        dim_coords_and_dims=time_coord_and_dims
        + [
            (
                iris.coords.DimCoord(
                    grid_lats, standard_name="latitude", units="degrees"
                ),
                lat_dim,
            ),
            (
                iris.coords.DimCoord(
                    grid_lons, standard_name="longitude", units="degrees"
                ),
                lon_dim,
            ),
        ],
    )

    new_cube.metadata = cube.metadata
    return regrid(
        new_cube,
        new_latitudes=n96e_lats,
        new_longitudes=n96e_lons,
        scheme=iris.analysis.Nearest(),
    )


def _process_jules_cube(cube, land_mask, n_pfts, frac_cube_shape):
    cube_2d = cube_1d_to_2d(cube)

    if len(cube_2d.shape) == 3:
        # Grid
        if land_mask:
            cube_2d.data.mask |= match_shape(~get_n96e_land_mask(), cube_2d.shape)
    elif len(cube_2d.shape) == 4:
        # PFT
        if land_mask:
            cube_2d.data.mask |= match_shape(
                match_shape(~get_n96e_land_mask(), cube_2d.shape[1:]), cube_2d.shape
            )
        cube_2d.add_dim_coord(
            iris.coords.DimCoord(np.arange(cube_2d.shape[1]), long_name="pft"), 1
        )

        if frac_cube_shape is not None:
            # Additional error checking.
            assert cube_2d.shape == (
                frac_cube_shape[0],
                n_pfts,
                frac_cube_shape[2],
                frac_cube_shape[3],
            )
    else:
        raise ValueError(f"Unexpected shape {cube_2d.shape}")

    return regrid_to_n96e(cube_2d)


def load_jules_data(
    files, variables, land_mask=True, n_pfts=13, frac_cube=None, single=False
):
    """Load JULES output into 2D cubes.

    Args:
        files (str or iterable of str): Filenames or patterns to load data from.
        variables (str or iterable of str): Target variables to load.
        land_mask (bool): If True, add land mask.
        n_pfts (int): The number of expected PFTs. Only applicable for cubes with 4
            dimensions.

    """
    if isinstance(files, str):
        files = (files,)
    if isinstance(variables, str):
        variables = (variables,)

    cubes = homogenise_time_coordinate(
        iris.load(files, constraints=variables)
    ).concatenate()

    proc_cubes = iris.cube.CubeList(
        [
            _process_jules_cube(
                cube,
                land_mask=land_mask,
                n_pfts=n_pfts,
                frac_cube_shape=(frac_cube.shape if frac_cube is not None else None),
            )
            for cube in cubes
        ]
    )
    if single:
        if len(proc_cubes) != 1:
            raise ValueError(f"Expected 1 cube, got {proc_cubes}.")
        return proc_cubes[0]
    return proc_cubes


def frac_weighted_mean(cube_2d, frac_cube, n_pfts=13):
    """Take weighted mean, weighted by frac, but only the X natural PFTs (e.g. 13)."""
    # Select the specified PFTs. This assumes the veg PFTs of interest are first along
    # their axis.
    pft_cube_2d = cube_2d[:, :n_pfts]
    pft_frac_cube = frac_cube[:, :n_pfts]

    assert pft_cube_2d.shape == pft_frac_cube.shape, (
        "Data (VEG PFTs) cube and frac cube should have the same shape. "
        f"Data shape: {pft_cube_2d.shape}, frac shape: {pft_frac_cube.shape}."
    )
    agg_data_2d = np.sum(pft_cube_2d.data * pft_frac_cube.data, axis=1) / np.sum(
        pft_frac_cube.data, axis=1
    )
    assert agg_data_2d.shape == (
        pft_cube_2d.shape[0],
        pft_cube_2d.shape[2],
        pft_cube_2d.shape[3],
    ), (
        "PFT (frac) dimension should no longer exist after weighted averaging. "
        f"Expected shape: {(pft_cube_2d.shape[0], pft_cube_2d.shape[2], pft_cube_2d.shape[3])}, got shape: {agg_data_2d.shape}."
    )
    return agg_data_2d


def get_1d_data_cube(data, lats, lons):
    # 1D JULES land points only.
    assert data.shape == (7771,)
    cube = iris.cube.Cube(data[np.newaxis])
    cube.add_aux_coord(lats, data_dims=(0, 1))
    cube.add_aux_coord(lons, data_dims=(0, 1))
    return cube[0]
