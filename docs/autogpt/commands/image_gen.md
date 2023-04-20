## `Image Generation Module for AutoGPT.`

This module contains functions to generate images from prompt using different image providers like HuggingFace, DALL-E, and SD WebUI.

### `generate_image`

This function generates an image from a given prompt.

**Arguments:**

- `prompt` _(str)_: The prompt to use.
- `size` _(int, optional)_: The size of the image. Defaults to 256. _(Not supported by HuggingFace)_

**Returns:**

- `str`: The filename of the image.

### `generate_image_with_hf`

This function generates an image using HuggingFace's image-generation API.

**Arguments:**

- `prompt` _(str)_: The prompt to use.
- `filename` _(str)_: The filename to save the image to.

**Returns:**

- `str`: The filename of the image.

### `generate_image_with_dalle`

This function generates an image using DALL-E image-generation API.

**Arguments:**

- `prompt` _(str)_: The prompt to use.
- `filename` _(str)_: The filename to save the image to.

**Returns:**

- `str`: The filename of the image.

### `generate_image_with_sd_webui`

This function generates an image using Stable Diffusion WebUI.

**Arguments:**

- `prompt` _(str)_: The prompt to use.
- `filename` _(str)_: The filename to save the image to.
- `size` _(int, optional)_: The size of the image. Defaults to 256.
- `negative_prompt` _(str, optional)_: The negative prompt to use. Defaults to "".
- `extra` _(dict, optional)_: Extra parameters to pass to the API. Defaults to {}.

**Returns:**

- `str`: The filename of the image.