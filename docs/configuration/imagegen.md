# 🖼 Image Generation configuration

| Config variable  | Values                          |                      |
| ---------------- | ------------------------------- | -------------------- |
| `IMAGE_PROVIDER` | `dalle` `huggingface` `sdwebui` | **default: `dalle`** |

## DALL-e

In `.env`, make sure `IMAGE_PROVIDER` is commented (or set to `dalle`):
``` ini
# IMAGE_PROVIDER=dalle    # this is the default
```

Further optional configuration:

| Config variable  | Values             |                |
| ---------------- | ------------------ | -------------- |
| `IMAGE_SIZE`     | `256` `512` `1024` | default: `256` |

## Hugging Face

To use text-to-image models from Hugging Face, you need a Hugging Face API token.
Link to the appropriate settings page: [Hugging Face > Settings > Tokens](https://huggingface.co/settings/tokens)

Once you have an API token, uncomment and adjust these variables in your `.env`:
``` ini
IMAGE_PROVIDER=huggingface
HUGGINGFACE_API_TOKEN=your-huggingface-api-token
```

Further optional configuration:

| Config variable           | Values                 |                                          |
| ------------------------- | ---------------------- | ---------------------------------------- |
| `HUGGINGFACE_IMAGE_MODEL` | see [available models] | default: `CompVis/stable-diffusion-v1-4` |

[available models]: https://huggingface.co/models?pipeline_tag=text-to-image

## Stable Diffusion WebUI

It is possible to use your own self-hosted Stable Diffusion WebUI with Auto-GPT:
``` ini
IMAGE_PROVIDER=sdwebui
```

!!! note
    Make sure you are running WebUI with `--api` enabled.

Further optional configuration:

| Config variable | Values                  |                                  |
| --------------- | ----------------------- | -------------------------------- |
| `SD_WEBUI_URL`  | URL to your WebUI       | default: `http://127.0.0.1:7860` |
| `SD_WEBUI_AUTH` | `{username}:{password}` | *Note: do not copy the braces!*  |

## Selenium
``` shell
sudo Xvfb :10 -ac -screen 0 1024x768x24 & DISPLAY=:10 <YOUR_CLIENT>
```
