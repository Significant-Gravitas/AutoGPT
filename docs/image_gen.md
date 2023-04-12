## Function `generate_image(prompt)`

This function generates an image based on a given prompt using one of two image generation providers DALL-E or STABLE DIFFUSION. The generated image is then saved to the local disk.

### Arguments
- `prompt`: A string containing the prompt for generating the image.

### Example Usage
```python
generate_image("A cute puppy sitting on the grass")
```

### Returns
- A string containing the filename of the saved image.

### Dependencies
- requests
- io
- os.path
- PIL
- config
- uuid
- openai
- base64