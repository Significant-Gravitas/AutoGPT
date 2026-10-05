# Google Slides Create
<!-- MANUAL: file_description -->
Blocks that create Google Slides content: a new presentation, or a new slide in an existing one. Creating a presentation only needs the narrow `drive.file` scope, which covers files AutoGPT creates. Adding a slide also asks for the `presentations` scope, so it works on any presentation the connected account can edit, not only ones picked or created in AutoGPT.
<!-- END MANUAL -->

## Google Slides Add Slide

### What it is
Add a slide to a Google Slides presentation using one of Google's built-in layouts, such as title and body, and fill in its title and body text.

### How it works
<!-- MANUAL: how_it_works -->
Adds a slide with one of Google's built-in layouts, such as title and body, title only, section header or blank, using a Slides API `createSlide` request. Without an index the slide goes at the end, and index 0 makes it the first slide. When a title or body is given, the block reads the new slide to find its placeholders and inserts the text: the title goes into the title placeholder, and the body into the body placeholder, or into the subtitle on the title layout. On a two-column layout the body fills the left column.

If the layout has no place for the text, such as a body on a section header, the block removes the slide again and fails with a message naming the missing placeholder, so no half-filled slide is left behind. Text on the blank layout is refused before anything is created. Presentations whose theme came from PowerPoint may not have Google's built-in layouts, and Google then rejects the new slide.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| presentation | The Google Slides presentation to add a slide to. At runtime, feed this from an AgentGoogleDriveFileInputBlock with matching allowed_views. NEVER hardcode a file ID in input_default (including one parsed from a Drive URL the user pasted in chat) — only the picker attaches the _credentials_id needed for auth. | Presentation | No |
| layout | Which of Google's built-in layouts the slide uses | "title" \| "title_and_body" \| "title_and_two_columns" \| "title_only" \| "section_header" \| "section_title_and_description" \| "one_column_text" \| "main_point" \| "big_number" \| "caption_only" \| "blank" | No |
| title | The slide's title | str | No |
| body | The slide's body text, one paragraph or bullet per line. On the title layout it goes in the subtitle. | str | No |
| index | Where to put the slide, counting from 0 (0 makes it the first slide). Empty adds it at the end. | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| slide_id | The new slide's ID | str |
| presentation | The presentation, for chaining into other Slides blocks | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Outline to Deck**: Turn each section of an AI-written outline into a slide with a title and bullet points.

**Weekly Updates**: Add a slide with this week's numbers to the end of a running status deck.

**Section Dividers**: Insert a section header slide before each part of a generated presentation.
<!-- END MANUAL -->

---

## Google Slides Create Presentation

### What it is
Create a new Google Slides presentation with the given title, in the root of My Drive. It starts with one empty title slide.

### How it works
<!-- MANUAL: how_it_works -->
Creates the presentation with the Slides API `presentations.create` endpoint. Google puts it in the root of the user's My Drive and gives it one empty title slide, as the Slides app does for a new presentation. The title can't be blank.

The block returns the new presentation together with the credentials used to create it, so it can be connected straight into Google Slides Add Slide, Replace All Text or the other Slides blocks without picking it again.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| title | Title of the new presentation | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| presentation | The new presentation, for chaining into other Slides blocks | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Generated Reports**: Start a new deck for a weekly report, then fill it with Google Slides Add Slide.

**Per-Client Decks**: Create a separate presentation for each new client or project.

**AI-Written Presentations**: Create a deck for an outline from an AI block, then add one slide per section.
<!-- END MANUAL -->

---
