"""Toy 2D experiments: datasets and architectures."""

from sharpeuler.toys.datasets import (
    make_branchy_flower_dataset,
    make_rotated_gaussian_grid,
    make_spiral,
)
from sharpeuler.toys.models import FMNet, FMNet2, FMNetBig

__all__ = [
    "make_branchy_flower_dataset",
    "make_rotated_gaussian_grid",
    "make_spiral",
    "FMNet",
    "FMNet2",
    "FMNetBig",
]
