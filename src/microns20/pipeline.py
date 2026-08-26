"""Compatibility imports for the canonical domain-module stage runners."""

from microns20.cave import run_stage00
from microns20.candidates import run_stage01
from microns20.morphology import run_stage02, run_stage07
from microns20.connectivity import run_stage03
from microns20.selection import run_stage04
from microns20.validation import run_stage05, run_stage10
from microns20.functional import run_stage06
from microns20.synapses import run_stage08
from microns20.sonata import run_stage09


__all__ = [
    "run_stage00",
    "run_stage01",
    "run_stage02",
    "run_stage03",
    "run_stage04",
    "run_stage05",
    "run_stage06",
    "run_stage07",
    "run_stage08",
    "run_stage09",
    "run_stage10",
]
