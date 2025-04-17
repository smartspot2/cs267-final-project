from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)

from .base import PretrainedModel


class StableDiffusionModel(PretrainedModel):
    model_id = "stabilityai/stable-diffusion-2"

    def __init__(self):
        self.pipeline = StableDiffusionPipeline.from_pretrained(self.model_id)

    def forward(self, *args, **kwargs):
        # TODO: fill in with a call to the model;
        # the pipeline automatically performs inference steps,
        # so we'd need to look at how to only get the raw model output
        pass
