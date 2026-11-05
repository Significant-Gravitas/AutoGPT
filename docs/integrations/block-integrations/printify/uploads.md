# Printify Uploads
<!-- MANUAL: file_description -->
Blocks for adding designs to the Printify media library.
<!-- END MANUAL -->

## Printify Upload Image

### What it is
Upload a design image to your Printify media library and get the image id that Create Printify Product prints.

### How it works
<!-- MANUAL: how_it_works -->
A public image URL is passed to Printify as is, so Printify downloads it (its recommendation for files over 5 MB). An uploaded file, workspace file or data URI is read safely through the platform's file handling and sent as base64. The block returns the Printify image id plus the preview URL and dimensions.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| image | The design: a public image URL, an uploaded file or a data URI | str (file) | Yes |
| file_name | Name to file it under in Printify (defaults to the image's name) | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| image_id | Printify image id, for products | str |
| preview_url | Printify's preview of the image | str |
| width | Image width in pixels | int |
| height | Image height in pixels | int |

### Possible use case
<!-- MANUAL: use_case -->
Upload each new sticker design from a Drive folder or an image generator, then pass the image id to Create Printify Product.
<!-- END MANUAL -->

---
