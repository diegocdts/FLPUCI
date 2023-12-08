import warnings
import nest_asyncio

from instances.dataset_instances import SF, RT
from instances.fcae_properties_instances import AE_PROPERTIES_SF
from instances.sample_selection_instances import SLI_3
from instances.training_instances import FEDERATED_TRAINING_1, CENTRALIZED_TRAINING_1
from instances.validation_instances import NEXT
from steps.step_02_sample_generation import SampleHandler

from communities_identification import cen_communities_identification, fed_communities_identification

warnings.filterwarnings("ignore")
nest_asyncio.apply()

fed_communities_identification(dataset=RT,
                               ae_properties=AE_PROPERTIES_SF,
                               training_parameters=FEDERATED_TRAINING_1,
                               sli_selection=SLI_3,
                               first_interval=0,
                               last_interval=11,
                               validate_at=NEXT,
                               acc_run=True)
cen_communities_identification(dataset=RT,
                               sample_handler=SampleHandler(RT),
                               ae_properties=AE_PROPERTIES_SF,
                               training_parameters=CENTRALIZED_TRAINING_1,
                               sli_selection=SLI_3,
                               first_interval=0,
                               last_interval=11,
                               validate_at=NEXT,
                               acc_run=True)
