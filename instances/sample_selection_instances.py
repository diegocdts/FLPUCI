from helpers.parameters_helper import SampleSelectionType
from helpers.types_helper import SampleSelectionParameters

SLI_1 = SampleSelectionParameters(SampleSelectionType.SLI, window_size=1)
SLI_3 = SampleSelectionParameters(SampleSelectionType.SLI, window_size=3)
ACC = SampleSelectionParameters(SampleSelectionType.ACC, window_size=10)
