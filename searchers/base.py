import abc

from models.base import PretrainedModel
from verifiers.base import Verifier


class Searcher(abc.ABC):
    def __init__(self, model: PretrainedModel, verifier: Verifier):
        self.model = model
        self.verifier = verifier

    @abc.abstractmethod
    def search(self, *args, **kwargs):
        """
        Search for the best initial noise, given algorithm parameters.
        """
        # TODO: make arguments more specific? after finalizing what is needed for search
