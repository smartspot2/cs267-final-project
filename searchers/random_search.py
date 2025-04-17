from .base import Searcher
from verifiers.base import Verifier


class RandomSearch(Searcher):
    def search(self, n_samples: int):
        """
        Generates N random samples, and pick the best one according to the verifier.
        """
        pass
