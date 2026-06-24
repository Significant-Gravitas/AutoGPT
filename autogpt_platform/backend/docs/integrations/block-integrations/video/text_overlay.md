# Video Text Overlay
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Video Text Overlay

### What it is
Add text overlay/caption to video

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| video_in | Input video (URL, data URI, or local path) | str (file) | Yes |
| text | Text to overlay on video | str | Yes |
| position | Position of text on screen | "top" \| "center" \| "bottom" \| "top-left" \| "top-right" \| "bottom-left" \| "bottom-right" | No |
| start_time | When to show text (seconds). None = entire video | float | No |
| end_time | When to hide text (seconds). None = until end | float | No |
| font_size | Font size | int | No |
| font_color | Font color (hex or name) | str | No |
| bg_color | Background color behind text (None for transparent) | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| video_out | Video with text overlay (path or data URI) | str (file) |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
