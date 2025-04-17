import abc
from typing import Any


class PretrainedModel(abc.ABC):
    """Abstract class describing common methods for all pretrained models"""

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """
        Given a prompt/image/noise as input (among other parameters),
        produces the raw output of the model.
        """
        # TODO: make arguments more specific (rather than using *args, **kwargs)
        return NotImplemented
