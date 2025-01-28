import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import requests
from io import BytesIO

class ImageTextToImagePipeline:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device="cuda"):
        """
        Initialize the pipeline with a specific model.
        
        Args:
            model_id (str): HuggingFace model ID
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
    
    def load_image(self, image_path_or_url):
        """
        Load an image from a local path or URL.
        
        Args:
            image_path_or_url (str): Local path or URL to the image
            
        Returns:
            PIL.Image: Loaded image
        """
        if image_path_or_url.startswith(('http://', 'https://')):
            response = requests.get(image_path_or_url)
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_path_or_url)
        
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        return image
    
    def preprocess_image(self, image, target_size=(768, 768)):
        """
        Preprocess the image to the required size while maintaining aspect ratio.
        
        Args:
            image (PIL.Image): Input image
            target_size (tuple): Desired output size
            
        Returns:
            PIL.Image: Preprocessed image
        """
        # Resize while maintaining aspect ratio
        image.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Create new image with padding
        new_image = Image.new("RGB", target_size, (255, 255, 255))
        new_image.paste(image, ((target_size[0] - image.size[0]) // 2,
                              (target_size[1] - image.size[1]) // 2))
        
        return new_image
    
    def generate(self, 
                init_image,
                prompt,
                strength=0.75,
                guidance_scale=7.5,
                num_inference_steps=50,
                negative_prompt=None):
        """
        Generate a new image based on the input image and text prompt.
        
        Args:
            init_image (str or PIL.Image): Initial image path/URL or PIL Image
            prompt (str): Text prompt describing desired modifications
            strength (float): How much to transform the image (0-1)
            guidance_scale (float): Higher values increase prompt adherence
            num_inference_steps (int): Number of denoising steps
            negative_prompt (str): Text prompt describing what to avoid
            
        Returns:
            PIL.Image: Generated image
        """
        # Load and preprocess image if necessary
        if isinstance(init_image, str):
            init_image = self.load_image(init_image)
        init_image = self.preprocess_image(init_image)
        
        # Generate image
        with torch.inference_mode():
            output = self.pipeline(
                prompt=prompt,
                image=init_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                negative_prompt=negative_prompt
            )
        
        return output.images[0]
    
    def __call__(self, *args, **kwargs):
        """Convenience method to call generate directly"""
        return self.generate(*args, **kwargs)
