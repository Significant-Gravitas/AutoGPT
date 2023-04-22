## 🖼 Image Generation

By default, Auto-GPT uses DALL-e for image generation. To use Stable Diffusion, a [Hugging Face API Token](https://huggingface.co/settings/tokens) is required.

Once you have a token, set these variables in your `.env`:

``` shell
IMAGE_PROVIDER=huggingface
HUGGINGFACE_API_TOKEN="YOUR_HUGGINGFACE_API_TOKEN"
```

## Selenium
``` shell
sudo Xvfb :10 -ac -screen 0 1024x768x24 & DISPLAY=:10 <YOUR_CLIENT>
```