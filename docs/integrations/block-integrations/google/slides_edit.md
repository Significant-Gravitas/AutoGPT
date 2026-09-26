# Google Slides Edit
<!-- MANUAL: file_description -->
Blocks that change existing Google Slides presentations: fill in templates, set speaker notes, or send any Slides API change. They ask for the `presentations` scope as well as `drive.file`, so they work on any presentation the connected account can edit. To fill in a template without changing the original, copy it first with Google Drive Copy File and edit the copy.
<!-- END MANUAL -->

## Google Slides Batch Update

### What it is
Change a Google Slides presentation with Slides API batchUpdate requests: add shapes, tables, images and slides, insert or delete text, restyle, reorder or delete objects. All requests succeed together or none are applied.

### How it works
<!-- MANUAL: how_it_works -->
Sends the requests unchanged to the Slides API `presentations.batchUpdate` endpoint. Each request is an object with one request type, such as `createShape`, `insertText`, `deleteObject`, `updateTextStyle`, `createImage` or `replaceAllText`; the Slides API reference lists them all. Google applies the requests in order and all together: if one is invalid, none are applied, and the block fails with Google's explanation.

The block returns one reply per request, in the same order. Requests that create something return its ID, and the rest return an empty object. Use Google Slides Get Slide to find the IDs of the slides and elements to change.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| presentation | The Google Slides presentation to change. At runtime, feed this from an AgentGoogleDriveFileInputBlock with matching allowed_views. NEVER hardcode a file ID in input_default (including one parsed from a Drive URL the user pasted in chat) — only the picker attaches the _credentials_id needed for auth. | Presentation | No |
| requests | Slides API batchUpdate requests, each an object with one request type, e.g. {"deleteObject": {"objectId": "g2f3c4d5e6_0_0"}}. See https://developers.google.com/workspace/slides/api/reference/rest/v1/presentations/request | List[Dict[str, Any]] | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| replies | One reply per request, in order, such as the IDs of created objects. Requests with nothing to report get an empty object. | List[Dict[str, Any]] |
| presentation | The presentation, for chaining into other Slides blocks | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Custom Layouts**: Add text boxes, shapes, tables or images at exact positions on a slide.

**Bulk Restyling**: Change fonts, colours or alignment across many elements in one call.

**Deck Cleanup**: Delete, reorder or duplicate slides and elements after a deck is generated.
<!-- END MANUAL -->

---

## Google Slides Replace All Text

### What it is
Replace text everywhere in a Google Slides presentation, with several find-and-replace pairs at once. Use it to fill in a template's placeholders, such as {{client}}, and see how often each one was replaced.

### How it works
<!-- MANUAL: how_it_works -->
Sends one `replaceAllText` request per pair in a single Slides API batch update, so either every replacement is made or none is. Text is replaced wherever it appears in shapes and tables, in the order the pairs are given, and matching is case-sensitive by default. The search can be limited to some slides by giving their IDs or links.

The block returns how many times each text was replaced, and the total. A count of 0 means that text wasn't found, which usually points to a typo in a template placeholder. The presentation is edited in place, so copy a template with Google Drive Copy File first to keep the original.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| presentation | The Google Slides presentation to change. At runtime, feed this from an AgentGoogleDriveFileInputBlock with matching allowed_views. NEVER hardcode a file ID in input_default (including one parsed from a Drive URL the user pasted in chat) — only the picker attaches the _credentials_id needed for auth. | Presentation | No |
| replacements | Text to find, and what to replace it with, e.g. {"{{client}}": "Acme Corp", "{{date}}": "1 October 2026"} | Dict[str, str] | Yes |
| match_case | Only replace text whose upper and lower case match exactly | bool | No |
| slide_ids | Only replace text on these slides (IDs or links). Empty means every slide. | List[str] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| occurrences_changed | How many replacements were made in total | int |
| occurrences | How many times each text was replaced; 0 means it wasn't found | Dict[str, int] |
| presentation | The presentation, for chaining into other Slides blocks | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Template Filling**: Replace placeholders like `{{client}}` and `{{date}}` in a copy of a proposal template.

**Rebranding**: Update a product or company name on every slide of a deck.

**Personalized Decks**: Produce one deck per customer from the same template, with their name and figures.
<!-- END MANUAL -->

---

## Google Slides Set Speaker Notes

### What it is
Set the speaker notes of one slide in a Google Slides presentation, replacing any notes it already has.

### How it works
<!-- MANUAL: how_it_works -->
Reads the slide to find its speaker notes, then clears any existing notes and inserts the new text in a single batch update. If the slide has never had notes, Google creates the notes box when the text is inserted. Empty text clears the notes, and clearing notes that are already empty changes nothing.

The slide can be given as its ID or as a link to the slide. An ID that belongs to something other than a slide, such as a layout, fails with a message saying so.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| presentation | The Google Slides presentation the slide is in. At runtime, feed this from an AgentGoogleDriveFileInputBlock with matching allowed_views. NEVER hardcode a file ID in input_default (including one parsed from a Drive URL the user pasted in chat) — only the picker attaches the _credentials_id needed for auth. | Presentation | No |
| slide_id | The slide's ID (from Google Slides Read Presentation) or a link to the slide | str | Yes |
| notes | The new speaker notes. They replace the slide's current notes; empty text clears them. | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| presentation | The presentation, for chaining into other Slides blocks | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Presenter Scripts**: Write an AI-generated talk track into the notes of each slide.

**Rehearsal Notes**: Add timing cues or key points to the slides of an upcoming talk.

**Handover Notes**: Leave context for a colleague who will present the deck.
<!-- END MANUAL -->

---
