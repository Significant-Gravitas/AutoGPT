# Google Slides Read
<!-- MANUAL: file_description -->
Blocks that read Google Slides presentations: the text and speaker notes of the whole deck, everything on one slide, or a picture of a slide. Pick the presentation with the Drive picker, or connect one from another block. They only need the narrow `drive.file` scope, which covers presentations the user picks and ones AutoGPT creates.
<!-- END MANUAL -->

## Google Slides Get Slide

### What it is
Get one slide of a Google Slides presentation by slide ID: every element on it (text boxes, shapes, tables, images) with its element ID, type and text, plus the slide's speaker notes.

### How it works
<!-- MANUAL: how_it_works -->
Fetches the slide with the Slides API `presentations.pages.get` endpoint. The slide can be given as its ID (from Google Slides Read Presentation) or as a link copied while the slide is selected, which ends in `#slide=id.<ID>`. Each element comes with its object ID, its type (`shape`, `table`, `image`, `video`, `line`, `sheets_chart`, `word_art` or `group`), its shape and placeholder type where it has one (such as `TEXT_BOX` and `TITLE`), and its text. Table cells are joined with ` | `, and the contents of a group are listed right after the group.

Google Slides Batch Update requests refer to elements by these IDs, so use this block to find the shape to change before editing it. If the presentation or slide doesn't exist, or the connected Google account can't open it, the block fails with a message saying so.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| presentation | The Google Slides presentation the slide is in. At runtime, feed this from an AgentGoogleDriveFileInputBlock with matching allowed_views. NEVER hardcode a file ID in input_default (including one parsed from a Drive URL the user pasted in chat) — only the picker attaches the _credentials_id needed for auth. | Presentation | No |
| slide_id | The slide's ID (from Google Slides Read Presentation) or a link to the slide | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| elements | Everything on the slide with its element ID, type and text. A group is followed by the elements inside it. | List[SlideElement] |
| element | Each element | SlideElement |
| text | All text on the slide | str |
| speaker_notes | The slide's speaker notes | str |
| presentation | The presentation, for chaining into other Slides blocks | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Targeted Edits**: Find the ID of a slide's title or body box before changing it with Google Slides Batch Update.

**Slide Review**: Pass one slide's text and speaker notes to an AI block to check them for tone, typos or missing points.

**Table Extraction**: Pull the figures out of a table on a specific slide, such as a quarterly results table.
<!-- END MANUAL -->

---

## Google Slides Get Slide Thumbnail

### What it is
Render one slide of a Google Slides presentation as a PNG image, for example to check how a slide looks after editing it.

### How it works
<!-- MANUAL: how_it_works -->
Asks the Slides API `presentations.pages.getThumbnail` endpoint to render the slide as a PNG at the chosen width: 200, 800 (the default), 1600 or 2000 pixels. Google returns a link to the image that expires after about 30 minutes, so the block downloads the image straight away and saves it as `<presentation name> - slide <ID>.png`. In CoPilot it becomes a workspace file; in agents it is passed on as a data URI.

The block also returns the image's size and Google's temporary link, for blocks that need an image URL. Anyone who has the link can see the slide until it expires. Google allows only 60 thumbnail requests per minute for each user, so avoid rendering every slide of a large deck in a tight loop.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| presentation | The Google Slides presentation the slide is in. At runtime, feed this from an AgentGoogleDriveFileInputBlock with matching allowed_views. NEVER hardcode a file ID in input_default (including one parsed from a Drive URL the user pasted in chat) — only the picker attaches the _credentials_id needed for auth. | Presentation | No |
| slide_id | The slide's ID (from Google Slides Read Presentation) or a link to the slide | str | Yes |
| size | Image width: small (200 px), medium (800 px), large (1600 px) or x_large (2000 px) | "small" \| "medium" \| "large" \| "x_large" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| thumbnail | The slide as a PNG image (a workspace file in CoPilot, a data URI in agents) | str (file) |
| width | Image width in pixels | int |
| height | Image height in pixels | int |
| content_url | Google's link to the image. It expires after about 30 minutes, and anyone who has it can see the slide. | str |
| presentation | The presentation, for chaining into other Slides blocks | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Visual Check After Editing**: Render a slide after changing it so an AI block with vision can check the layout.

**Slide Previews in Messages**: Attach a picture of a key slide to an email or chat message.

**Deck Snapshots**: Save an image of a recurring slide, such as a weekly metrics slide, to keep a visual record.
<!-- END MANUAL -->

---

## Google Slides Read Presentation

### What it is
Read a Google Slides presentation: its title and, for each slide, the slide ID, position, title, text from shapes and tables, and speaker notes. Also returns the whole deck as one Markdown text.

### How it works
<!-- MANUAL: how_it_works -->
Fetches the presentation with the Slides API `presentations.get` endpoint, asking only for the slides and not for the theme's masters and layouts. For each slide it returns the slide ID, its position (counting from 0), the text of its title placeholder, all the text in its shapes, tables and groups, and its speaker notes. Soft line breaks become new lines and table cells are joined with ` | `. Images, videos and charts have no text, so they add nothing.

The `text` output puts the whole deck into one Markdown document, with a heading and the slide ID for each slide and its speaker notes after its text, ready for an AI block. `slides` holds every slide at once, and `slide` emits them one at a time.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| presentation | The Google Slides presentation to read. At runtime, feed this from an AgentGoogleDriveFileInputBlock with matching allowed_views. NEVER hardcode a file ID in input_default (including one parsed from a Drive URL the user pasted in chat) — only the picker attaches the _credentials_id needed for auth. | Presentation | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| title | The presentation's title | str |
| text | All slide text and speaker notes as one Markdown document, for AI blocks | str |
| slides | Every slide with its ID, position, title, text and speaker notes | List[SlideSummary] |
| slide | Each slide | SlideSummary |
| presentation | The presentation, for chaining into other Slides blocks | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Deck Summaries**: Summarize a long presentation, speaker notes included, with an AI block.

**Content Audits**: Check every slide of a sales deck for outdated prices, names or dates.

**Slide Lookups**: Find the ID of the slide that covers a topic, to render it or edit it with the other Slides blocks.
<!-- END MANUAL -->

---
