import ImageReward as RM

from .base import Verifier


class ImageRewardVerifier(Verifier):
    model_id = "ImageReward-v1.0"

    def __init__(self):
        self.verifier = RM.load(self.model_id)

    def get_reward(self, prompt, image):
        return self.verifier.score(prompt, image)
