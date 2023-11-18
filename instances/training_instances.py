from helpers.types_helper import TrainingParameters

FEDERATED_TRAINING_1 = TrainingParameters(epochs=3, batch_size=2, shuffle_buffer=10, prefetch_buffer=-1, rounds=20)
FEDERATED_TRAINING_2 = TrainingParameters(epochs=3, batch_size=2, shuffle_buffer=10, prefetch_buffer=-1, rounds=5)
CENTRALIZED_TRAINING_1 = TrainingParameters(epochs=150, batch_size=300)
CENTRALIZED_TRAINING_2 = TrainingParameters(epochs=15, batch_size=5)
