from enum import Enum


class TimeUnits:

    def __init__(self, epoch_size):
        if epoch_size == 13:
            self.HOUR = 3600000
            self.DAY = 86400000
        else:
            self.HOUR = 3600
            self.DAY = 86400


class ImageMetric(Enum):
    MSE = 'mse'
    SSIM = 'ssim'
    ARI = 'ari'


class SampleSelectionType(Enum):
    SLI = 'sli'
    ACC = 'acc'

    def __str__(self):
        return str(self.value).upper()


class IntervalToValidate(Enum):
    END = 0
    NEXT = 1
    EQUIVALENT_12h = 14
    EQUIVALENT_24h = 7
