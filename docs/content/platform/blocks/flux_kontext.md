# Flux Kontext

## What it is
An internal block that performs text-based image editing using BlackForest Labs' Flux Kontext models.

## What it does
Takes a prompt describing the desired transformation and optionally a reference image, then returns a new image URL.

## How it works
The block sends your prompt, image, and settings to the selected Flux Kontext model on Replicate. The service processes the request and returns a link to the edited image.

## Inputs
| Input | Description |
|-------|-------------|
| API Key | Your Replicate API key |
| Prompt | Instruction describing how to edit the image |
| Input Image | Optional reference image URI to modify |
| Aspect Ratio | Target aspect ratio for the output |
| Seed | Optional seed for reproducible results |
| Model | Choose between Flux Kontext Pro and Flux Kontext Max |

## Outputs
| Output | Description |
|--------|-------------|
| image_url | URL of the transformed image |
| error | Error message if something goes wrong |

## Possible use case
Enhance a marketing image by requesting "add soft lighting and a subtle vignette" while providing the original asset as the reference image.
