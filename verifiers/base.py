import abc
from typing import Any

import torch


class Verifier(abc.ABC):

    @abc.abstractmethod
    def get_reward(self, prompt: str | list[str], image) -> Any:
        """
        Given a prompt/image/noise as input (among other parameters),
        computes the reward given by the verifier.
        """
        # TODO: make arguments more specific? after finalizing what is needed for verification
        return NotImplemented
