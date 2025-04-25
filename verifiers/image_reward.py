import ImageReward as RM

import utils.cache
import utils.device

from .base import Verifier


class ImageRewardVerifier(Verifier):
    model_id = "ImageReward-v1.0"

    def __init__(self):
        self.verifier = RM.load(
            self.model_id,
            device=utils.device.DEVICE,
            download_root=utils.cache.CACHE_DIR,
        )

    def get_reward(self, prompt, image):
        return self.verifier.score(prompt, image)
