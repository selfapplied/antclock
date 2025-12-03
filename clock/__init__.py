"""
Clock Module - AntClock: Pascal Curvature Clock System

A modular implementation of the AntClock system with separated concerns:
- pascal_core: Mathematical primitives
- homology_engine: Topological computations
- clock_mechanics: Walker implementations
- analysis_framework: Chaos analysis tools

Author: Joel
"""

from .antclock import AntClock, quick_trajectory
from .pascal import *
from .homology import *
from .mechanics import *
from .analysis import *

__version__ = "1.0.0"
__author__ = "Joel"
