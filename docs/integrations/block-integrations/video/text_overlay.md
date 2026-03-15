# Video Text Overlay
<!-- MANUAL: file_description -->
This block adds customizable text captions or titles to videos, with control over positioning, timing, and styling.
<!-- END MANUAL -->

## Video Text Overlay

### What it is
Add text overlay/caption to video

### How it works
<!-- MANUAL: how_it_works -->
The block uses MoviePy's TextClip and CompositeVideoClip to render text onto video frames. The text is created as a separate clip with configurable font size, color, and optional background color, then composited over the video at the specified position. Timing can be controlled to show text only during specific portions of the video. Position options include center alignments (top, center, bottom) and corner positions (top-left, top-right, bottom-left, bottom-right). The output is encoded with H.264 video codec and AAC audio codec.
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
- Adding titles or chapter headings to video content
- Creating lower-thirds with speaker names or captions
- Watermarking videos with branding text
- Adding call-to-action text at specific moments in a video
<!-- END MANUAL -->

---
