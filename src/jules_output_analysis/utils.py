# -*- coding: utf-8 -*-

from enum import Enum

import iris
import numpy as np
from joblib import Memory
from numba import njit
from sklearn.model_selection import train_test_split

memory = Memory(".cache", verbose=0)

# Training and validation test splitting.
train_test_split_kwargs = dict(random_state=1, shuffle=True, test_size=0.3)

# Specify common RF (training) params.
n_splits = 5

default_param_dict = {"random_state": 1, "bootstrap": True}

# XXX
param_dict = {
    **default_param_dict,
    "ccp_alpha": 2e-9,
    "max_depth": 15,
    "max_features": "auto",
    "min_samples_leaf": 4,
    "min_samples_split": 2,
    "n_estimators": 100,
}

PFTs = Enum("PFTs", ["VEG5", "VEG5_ALL", "VEG13", "VEG13_ALL", "NON_VEG"])

pft_names = {
    PFTs.VEG5: (
        "Broadleaf trees",
        "Needleleaf trees",
        "C3 (temperate) grass",
        "C4 (tropical) grass",
        "Shrubs",
    ),
    PFTs.VEG13: (
        "BDT",
        "BE-Tr",
        "BE-Te",
        "NDT",
        "NET",
        "C3G",
        "C3C",
        "C3P",
        "C4G",
        "C4C",
        "C4P",
        "SbD",
        "SbE",
    ),
    PFTs.NON_VEG: ("Urban", "Lake", "Bare Soil", "Ice"),
}

pft_acronyms = {
    PFTs.VEG5: (
        "BT",
        "NT",
        "C3G",
        "C4G",
        "Sb",
    ),
    PFTs.VEG13: (
        "BDT",
        "BE-Tr",
        "BE-Te",
        "NDT",
        "NET",
        "C3G",
        "C3C",
        "C3P",
        "C4G",
        "C4C",
        "C4P",
        "SbD",
        "SbE",
    ),
    PFTs.NON_VEG: (
        "Ub",
        "Lk",
        "BS",
        "Ice",
    ),
}

pft_names[PFTs.VEG5_ALL] = tuple(
    list(pft_names[PFTs.VEG5]) + list(pft_names[PFTs.NON_VEG])
)
pft_names[PFTs.VEG13_ALL] = tuple(
    list(pft_names[PFTs.VEG13]) + list(pft_names[PFTs.NON_VEG])
)
pft_acronyms[PFTs.VEG5_ALL] = tuple(
    list(pft_acronyms[PFTs.VEG5]) + list(pft_acronyms[PFTs.NON_VEG])
)
pft_acronyms[PFTs.VEG13_ALL] = tuple(
    list(pft_acronyms[PFTs.VEG13]) + list(pft_acronyms[PFTs.NON_VEG])
)


def get_mm_indices(master_mask):
    mm_valid_indices = np.where(~master_mask.ravel())[0]
    mm_valid_train_indices, mm_valid_val_indices = train_test_split(
        mm_valid_indices, **train_test_split_kwargs
    )
    return mm_valid_indices, mm_valid_train_indices, mm_valid_val_indices


def get_mm_data(x, master_mask, kind):
    """Return masked master_mask copy and training or validation indices.

    The master_mask copy is filled using the given data.

    Args:
        x (array-like): Data to use.
        master_mask (array):
        kind ({'train', 'val'})

    Returns:
        masked_data, mm_indices:

    """
    mm_valid_indices, mm_valid_train_indices, mm_valid_val_indices = get_mm_indices(
        master_mask
    )
    masked_data = np.ma.MaskedArray(
        np.zeros_like(master_mask, dtype=np.float64), mask=np.ones_like(master_mask)
    )
    if kind == "train":
        masked_data.ravel()[mm_valid_train_indices] = x
    elif kind == "val":
        masked_data.ravel()[mm_valid_val_indices] = x
    else:
        raise ValueError(f"Unknown kind: {kind}")
    return masked_data


@njit
def isclose(a, b, atol=1e-4):
    return np.abs(a - b) < atol


@njit
def find_gridpoint(land_lat, land_lon, grid_lats, grid_lons):
    """Mapping from a single land coordinate to the matching grid indices."""
    for lat_i, grid_lat in enumerate(grid_lats):
        for lon_i, grid_lon in enumerate(grid_lons):
            if isclose(land_lat, grid_lat) and isclose(land_lon, grid_lon):
                return lat_i, lon_i
    raise RuntimeError("Matching gridpoint not found.")


@memory.cache
def get_grid_mask(mask, orig_lats, orig_lons, grid_lats, grid_lons):
    """Calculate mask to transition from one grid to another.

    Note:
        This probably relies on the contiguity structure of the arrays.

    """
    mask[
        ...,
        np.rint((orig_lats - grid_lats[0]) / (grid_lats[1] - grid_lats[0])).astype(
            "int64"
        ),
        np.rint((orig_lons - grid_lons[0]) / (grid_lons[1] - grid_lons[0])).astype(
            "int64"
        ),
    ] = True
    return mask


@njit
def find_min_error(x, y):
    """For two arrays `x` and `y` of potentially unequal lengths find"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lengths = np.array([len(x), len(y)])
    sh = (x, y)[np.argsort(lengths)[0]]
    lo = (x, y)[np.argsort(lengths)[-1]]
    diff = np.abs(np.diff(lengths)[0])

    min_error = np.inf
    for i in range(diff + 1):
        error = np.sum((sh - lo[i : len(lo) - diff + i]) ** 2)
        if error < min_error:
            min_error = error
    return min_error


def collapse_cube_dim(cube, collapse_dim):
    """Take the mean over a given cube dimension, which need not be named.

    Warning: Currently realises the data.

    """
    # Make sure we are not affecting any existing coordinates.
    for coord in cube.coords():
        assert collapse_dim not in cube.coord_dims(coord)

    data = cube.data
    # Collapse data.
    if data.shape[collapse_dim] > 1:
        new_data = np.mean(data, axis=collapse_dim)
    else:
        selection = [slice(None)] * len(data.shape)
        selection[collapse_dim] = 0
        new_data = data[tuple(selection)]

    def dims_remap(dims):
        out = []
        for dim in dims:
            if dim > collapse_dim:
                out.append(dim - 1)
            else:
                out.append(dim)
        return tuple(out)

    dim_coords = cube.dim_coords
    dim_coord_dims = [cube.coord_dims(coord) for coord in dim_coords]
    dim_coords_and_dims = [
        (coord, dims_remap(dims)[0]) for coord, dims in zip(dim_coords, dim_coord_dims)
    ]

    aux_coords = cube.aux_coords
    aux_coord_dims = [cube.coord_dims(coord) for coord in aux_coords]
    aux_coords_and_dims = [
        (coord, dims_remap(dims)) for coord, dims in zip(aux_coords, aux_coord_dims)
    ]

    # Construct new cube.
    new_cube = iris.cube.Cube(
        new_data,
        dim_coords_and_dims=dim_coords_and_dims,
        aux_coords_and_dims=aux_coords_and_dims,
    )
    new_cube.metadata = cube.metadata

    return new_cube
