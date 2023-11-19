import warnings
import nest_asyncio

from instances.dataset_instances import NGSIM
from instances.fcae_properties_instances import AE_PROPERTIES
from instances.sample_selection_instances import SLI_3
from instances.training_instances import FEDERATED_TRAINING_1, CENTRALIZED_TRAINING_1
from instances.validation_instances import NEXT
from steps.step_02_sample_generation import SampleHandler

from communities_identification import cen_communities_identification, fed_communities_identification

warnings.filterwarnings("ignore")
nest_asyncio.apply()

cen_communities_identification(dataset=NGSIM,
                               sample_handler=SampleHandler(NGSIM),
                               ae_properties=AE_PROPERTIES,
                               training_parameters=CENTRALIZED_TRAINING_1,
                               sli_selection=SLI_3,
                               first_interval=0,
                               last_interval=7,
                               validate_at=NEXT,
                               acc_run=True)
fed_communities_identification(dataset=NGSIM,
                               ae_properties=AE_PROPERTIES,
                               training_parameters=FEDERATED_TRAINING_1,
                               sli_selection=SLI_3,
                               first_interval=0,
                               last_interval=7,
                               validate_at=NEXT,
                               acc_run=True)
