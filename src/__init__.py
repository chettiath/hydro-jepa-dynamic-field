"""
hydro_jepa_dynamic_field.src

Core modules for the 1D dynamic neural field + JEPA predictor project.
"""

from .configs import TRAIN_CONFIG
from .dynamic_field import DynamicField
from .stimulus_dataset import StimulusDataset
from .jepa_head import JEPAHead