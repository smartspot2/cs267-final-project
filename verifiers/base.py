import abc
from typing import Any


class Verifier(abc.ABC):

    @abc.abstractmethod
    def get_reward(self, *args, **kwargs) -> Any:
        """
        Given a prompt/image/noise as input (among other parameters),
        computes the reward given by the verifier.
        """
        # TODO: make arguments more specific? after finalizing what is needed for verification
        return NotImplemented
