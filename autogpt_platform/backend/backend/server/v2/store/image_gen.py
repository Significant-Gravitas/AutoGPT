import io
import logging
import replicate
import replicate.exceptions
from enum import Enum
from backend.util.settings import Settings
from backend.data.graph import Graph
import requests

logger = logging.getLogger(__name__)

class ImageSize(str, Enum):
    LANDSCAPE = "1024x768"

class ImageStyle(str, Enum):
    DIGITAL_ART = "digital art"

async def generate_agent_image(agent: Graph) -> io.BytesIO:
    """
    Generate an image for an agent using Flux model via Replicate API.
    
    Args:
        agent (Graph): The agent to generate an image for
        
    Returns:
        io.BytesIO: The generated image as bytes
    """
    try:
        settings = Settings()
        
        if not settings.secrets.replicate_api_key:
            raise ValueError("Missing Replicate API key in settings")

        # Construct prompt from agent details
        prompt = f"App store image for AI agent that gives a cool visual representation of the agent does: - {agent.name} - {agent.description}"

        # Set up Replicate client
        client = replicate.Client(api_token=settings.secrets.replicate_api_key)

        # Model parameters
        input_data = {
            "prompt": prompt,
            "width": 1024,
            "height": 768,
            "aspect_ratio": "4:3",
            "output_format": "jpg",
            "output_quality": 90,
            "num_inference_steps": 30,
            "guidance": 3.5,
            "negative_prompt": "blurry, low quality, distorted, deformed",
            "disable_safety_checker": True
        }

        try:
            # Run model
            output = client.run(
                "black-forest-labs/flux-pro",
                input=input_data
            )

            if not output:
                raise RuntimeError("No output generated")

            # Get image URL
            image_url = next(iter(output))
            logger.info(f"Generated image URL: {image_url}")

            # Download generated image
            try:
                import os
                # Handle case where image_url is bytes
                if isinstance(image_url, bytes):
                    return io.BytesIO(image_url)

                response = requests.get(image_url)
                if response.status_code != 200:
                    logger.error(f"Failed to retrieve image. Status code: {response.status_code}")
                    raise RuntimeError(f"Failed to download image: HTTP {response.status_code}")

                # Write image to local file for debugging/backup
                current_dir = os.path.dirname(os.path.abspath(__file__))
                os.makedirs(os.path.join(current_dir, "generated_images"), exist_ok=True)
                filename = os.path.join(current_dir, "generated_images", f"agent_{agent.id}.jpg")
                with open(filename, "wb") as f:
                    f.write(response.content)
                logger.info(f"Saved generated image to {filename}")

                return io.BytesIO(response.content)
            except Exception as e:
                logger.error(f"Failed to download and save image: {str(e)}")
                raise RuntimeError(f"Failed to process image: {str(e)}")

        except replicate.exceptions.ReplicateError as e:
            if e.status == 401:
                raise RuntimeError("Invalid Replicate API token") from e
            raise RuntimeError(f"Replicate API error: {str(e)}") from e

    except Exception as e:
        logger.exception("Failed to generate agent image")
        raise RuntimeError(f"Image generation failed: {str(e)}")
