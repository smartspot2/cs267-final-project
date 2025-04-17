from denoisers.serial_denoiser import SerialDenoiser
from models.stable_diffusion import StableDiffusionModel
from searchers.random_search import RandomSearch
from verifiers.image_reward import ImageRewardVerifier


def main():
    model = StableDiffusionModel()
    verifier = ImageRewardVerifier()
    denoiser = SerialDenoiser(model)

    # TODO: could also specify to use the parallel sampler here?
    searcher = RandomSearch(model, verifier)

    # search for the best initial noise to use
    initial_noise = searcher.search(n_samples=100)

    # generate the output given the initial noise
    output = denoiser.denoise(initial_noise)

    return output


if __name__ == "__main__":
    main()
