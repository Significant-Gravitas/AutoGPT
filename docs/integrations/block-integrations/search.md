# Search
<!-- MANUAL: file_description -->
Blocks for web searching, content extraction, and information retrieval from various search engines and APIs.
<!-- END MANUAL -->

## Get Wikipedia Summary

### What it is
This block fetches the summary of a given topic from Wikipedia.

### How it works
<!-- MANUAL: how_it_works -->
The block sends a request to Wikipedia's API with the provided topic. It then extracts the summary from the response and returns it. If there's an error during this process, it will return an error message instead.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| topic | The topic to fetch the summary for | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the summary cannot be retrieved | str |
| summary | The summary of the given topic | str |

### Possible use case
<!-- MANUAL: use_case -->
A student researching for a project could use this block to quickly get overviews of various topics, helping them decide which areas to focus on for more in-depth study.
<!-- END MANUAL -->

---

## Google Maps Search

### What it is
Search Google Maps for businesses and other places that match a text query. Returns each place's name, address, phone, rating, review count, website, place ID, coordinates and Google Maps link.

### How it works
<!-- MANUAL: how_it_works -->
Calls the Places API (New) Text Search endpoint with the query and an `X-Goog-FieldMask` header, so Google returns only the fields the block outputs. Results come 20 to a page, and the block follows `nextPageToken` until it has `max_results` places (up to 60). Each place has its name, address, phone, rating, review count, website, place ID, coordinates and Google Maps link, so you can pass it to other Google Maps blocks.

Name the area in the query, e.g. `pizza in Rome`. Google only applies a radius around a centre point, which this block doesn't take, so `radius` is kept for existing agents but isn't sent. The Maps API key's Google Cloud project needs Places API (New) enabled. Because the block asks for the phone, website and rating, Google bills each page at its Enterprise Text Search rate.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | Search query for local businesses | str | Yes |
| radius | Not used: Google needs a centre point to apply a radius, so name the area in the query instead. Kept so existing agents still load. | int | No |
| max_results | Maximum number of results to return (max 60) | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| place | Place found | Place |

### Possible use case
<!-- MANUAL: use_case -->
**Lead Generation**: Find businesses in a specific area for sales outreach.

**Competitive Analysis**: Search for competitors in target locations to analyze their presence and ratings.

**Local SEO**: Gather data on local businesses for market research or directory building.
<!-- END MANUAL -->

---
