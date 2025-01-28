import argparse
import sys
from pathlib import Path
from stable_diffusion_pipeline import ImageTextToImagePipeline  # Assuming previous code is saved as stable_diffusion_pipeline.py

def parse_args():
    parser = argparse.ArgumentParser(description='Generate images using Stable Diffusion with image and text inputs')
    
    # Required arguments
    parser.add_argument('--input-image', '-i', required=True,
                       help='Path or URL to the input image')
    parser.add_argument('--prompt', '-p', required=True,
                       help='Text prompt describing desired modifications')
    parser.add_argument('--output', '-o', required=True,
                       help='Output path for generated image')
    
    # Optional arguments
    parser.add_argument('--model-id', default="runwayml/stable-diffusion-v1-5",
                       help='HuggingFace model ID (default: runwayml/stable-diffusion-v1-5)')
    parser.add_argument('--device', default="mps",
                       choices=['mps', 'cpu', 'cuda'],
                       help='Device to run the model on (default: mps. cpu and cuda if available)')
    parser.add_argument('--strength', '-s', type=float, default=0.75,
                       help='Transformation strength, 0-1 (default: 0.75)')
    parser.add_argument('--guidance-scale', '-g', type=float, default=7.5,
                       help='Guidance scale (default: 7.5)')
    parser.add_argument('--steps', '-n', type=int, default=50,
                       help='Number of inference steps (default: 50)')
    parser.add_argument('--negative-prompt', 
                       help='Negative prompt describing what to avoid')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize pipeline
        print(f"Initializing pipeline with model {args.model_id} on {args.device}...")
        pipeline = ImageTextToImagePipeline(
            model_id=args.model_id,
            device=args.device
        )
        
        # Generate image
        print("Generating image...")
        output_image = pipeline(
            init_image=args.input_image,
            prompt=args.prompt,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            negative_prompt=args.negative_prompt
        )
        
        # Save output
        output_image.save(args.output)
        print(f"Image saved to {args.output}")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
