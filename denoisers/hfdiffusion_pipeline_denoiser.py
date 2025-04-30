import torch
from typing import Union, Any
from models.base import PretrainedModel
from .base import Denoiser

class DiffusionPipelineDenoiser(Denoiser):
    """
    A concrete implementation of the Denoiser abstract base class
    that uses HuggingFace Diffusion Pipeline.
    """
    
    def __init__(self, model: PretrainedModel):
        super().__init__(model)
        # Pipeline is initialized based on the pretrained model
        self.pipe = self.model.get_pipeline()
        
        # Move to CUDA if available
        if torch.cuda.is_available():
            self.pipe = self.pipe.to("cuda:0")
        
        # Disable progress bar for cleaner output
        self.pipe.set_progress_bar_config(disable=True)
    
    def denoise(
        self,
        initial_latents: torch.Tensor,
        prompt: Union[str, list[str]],
        height: int,
        width: int,
        num_images_per_prompt: int = 1,
        num_inference_steps: int = 50,
        **kwargs
    ):
        """
        Implements the abstract method from the Denoiser base class.
        Denoises the input using the diffusion pipeline.
        
        Args:
            initial_latents: Initial noise tensors
            prompt: Text conditioning
            height: Image height
            width: Image width
            num_images_per_prompt: Number of images to generate per prompt
            num_inference_steps: Number of diffusion steps
            **kwargs: Additional arguments to pass to the pipeline
        
        Returns:
            The output from the diffusion pipeline
        """
        # Ensure prompts are in the correct format
        if isinstance(prompt, str):
            batched_prompts = [prompt] * len(initial_latents)
        else:
            batched_prompts = prompt
            
        # Prepare additional arguments
        pipeline_args = {
            "prompt": batched_prompts,
            "latents": initial_latents,
            "height": height,
            "width": width,
            "num_images_per_prompt": num_images_per_prompt,
            "num_inference_steps": num_inference_steps,
        }
        
        # Add any additional kwargs
        pipeline_args.update(kwargs)
        
        # Call the diffusion pipeline
        result = self.pipe(**pipeline_args)
        
        return result
    
    def to(self, device: str):
        """
        Moves the pipeline to the specified device.
        
        Args:
            device: The device to move the pipeline to (e.g., "cuda:0", "cpu")
            
        Returns:
            self for method chaining
        """
        self.pipe = self.pipe.to(device)
        return self