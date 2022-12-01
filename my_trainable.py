from ray import tune
from typing import Dict, Optional, Union


class Trainable(tune.Trainable):
    def setup(self, config: Dict):
        print("config", config)

    def step(self):
        pass

    def reset_config(self, new_config: Dict):
        pass

    def save_checkpoint(self, checkpoint_dir: str) -> Optional[Union[str, Dict]]:
        pass

    def load_checkpoint(self, checkpoint: Union[Dict, str]):
        pass

    def validate_config(config):
        pass

