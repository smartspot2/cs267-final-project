from .base import Denoiser


# TODO: may be better to split this into DDPM, flow matching, DDIM, etc. depending on the model used
class SerialDenoiser(Denoiser):
    def denoise(self, *args, **kwargs):
        # TODO: fill in with algorithm to denoise noise
        pass
