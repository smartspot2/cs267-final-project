import torch
from typing import Union, List
from models.base import PretrainedModel
from diffusers import DiffusionPipeline
from .base import Denoiser
import utils.cache

class DiffusionPipelineDenoiser(Denoiser):
    """
    A concrete implementation of the Denoiser abstract base class
    that uses HuggingFace Diffusion Pipeline.
    """
    
    def __init__(self, model: Union[PretrainedModel, DiffusionPipeline]):
        super().__init__(model)
        
        # Handle both PretrainedModel and direct DiffusionPipeline inputs
        if isinstance(model, DiffusionPipeline):
            self.pipe = model
        else:
            # Assuming model.get_pipeline() returns a DiffusionPipeline
            self.pipe = self.model.get_pipeline()
        
        # Get scheduler from pipeline
        self.scheduler = self.pipe.scheduler
        
        # Move to CUDA if available
        if torch.cuda.is_available():
            self.pipe = self.pipe.to("cuda:0")
        
        # Disable progress bar for cleaner output
        self.pipe.set_progress_bar_config(disable=True)
    
    def denoise(
        self,
        initial_latents: torch.Tensor,
        prompt: Union[str, List[str]],
        height: int = 512,
        width: int = 512,
        num_images_per_prompt: int = 1,
        num_inference_steps: int = 50,
        parallel: int = 1,
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
            parallel: Number of parallel denoising operations
            **kwargs: Additional arguments to pass to the pipeline
        
        Returns:
            The output processed images from the diffusion pipeline
        """
        # Ensure prompts are in the correct format
        if isinstance(prompt, str):
            batched_prompts = [prompt] * (initial_latents.size(0) // num_images_per_prompt)
        else:
            batched_prompts = prompt
            
        # Handle potential batch processing based on parallel parameter
        batch_size = len(batched_prompts)
        results = []
        
        for i in range(0, batch_size, parallel):
            batch_end = min(i + parallel, batch_size)
            batch_prompts = batched_prompts[i:batch_end]
            batch_latents = initial_latents[i * num_images_per_prompt:(batch_end) * num_images_per_prompt]
            
            # Prepare pipeline arguments
            pipeline_args = {
                "prompt": batch_prompts,
                "latents": batch_latents,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "num_images_per_prompt": num_images_per_prompt,
                "return_dict": True
            }
            
            # Add any additional kwargs
            for key, value in kwargs.items():
                if key != "denoiser_kwargs":  # Skip nested denoiser_kwargs
                    pipeline_args[key] = value
            
            # Call the diffusion pipeline
            batch_result = self.pipe(**pipeline_args)
            
            # Extract images from the result
            if hasattr(batch_result, "images"):
                results.extend(batch_result.images)
            else:
                # Fallback if images attribute is not available
                results.extend(batch_result)
        
        return results
    
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