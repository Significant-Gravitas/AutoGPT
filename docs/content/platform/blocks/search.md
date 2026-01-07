# Get Wikipedia Summary

### What it is
This block fetches the summary of a given topic from Wikipedia.

### What it does
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
This block searches for local businesses using Google Maps API.

### What it does
This block searches for local businesses using Google Maps API.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | Search query for local businesses | str | Yes |
| radius | Search radius in meters (max 50000) | int | No |
| max_results | Maximum number of results to return (max 60) | int | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| place | Place found | Place |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
