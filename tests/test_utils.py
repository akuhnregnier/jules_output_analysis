# -*- coding: utf-8 -*-
import numpy as np

from jules_output_analysis.utils import find_gridpoint, isclose


def test_isclose():
    assert isclose(1, 1)
    assert isclose(1, 1 + 1e-5)
    assert not isclose(1, 1 + 1e-5, atol=1e-6)


def test_find_gridpoint():
    assert find_gridpoint(0, 0, np.array([-1, 0, 1]), np.array([-1, 0, 1])) == (1, 1)
    assert find_gridpoint(
        0.5, 0.5, np.array([-1, 0, 0.5, 1]), np.array([-1, 0, 0.25, 0.5, 0.75, 1])
    ) == (2, 3)
    assert (
        find_gridpoint(
            0,
            0,
            np.linspace(-90, 90, 100, endpoint=False),
            np.linspace(-90, 90, 100, endpoint=False),
        )
        == (50, 50)
    )
