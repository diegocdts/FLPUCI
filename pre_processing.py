from instances.dataset_instances import SF
from steps.step_01_pre_processing import pre_processing
from steps.step_02_sample_generation import SampleHandler, heat_maps_samples_view

pre_processing(SF)

sh = SampleHandler(SF)
dataset, indices = sh.samples_as_list(5, 7)
heat_maps_samples_view(dataset, 9, 4)
