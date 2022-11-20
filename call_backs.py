from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import SampleBatch

class MyCallBacks(DefaultCallbacks):
    def __init__(self):
        super.__init__()

    def on_sample_end(self, *, worker: "RolloutWorker", samples: SampleBatch, **kwargs) -> None:
        return None