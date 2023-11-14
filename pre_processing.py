import warnings

from instances.dataset_instances import SF, NGSIM
from steps.step_01_pre_processing import pre_processing
from steps.step_03_baseline_computation import compute_baseline
warnings.filterwarnings("ignore")


pre_processing(NGSIM)

compute_baseline(NGSIM)
