# CoPilot Tools - Future Ideas

## Multimodal Image Support for CoPilot

**Problem:** CoPilot uses a vision-capable model but can't "see" workspace images. When a block generates an image and returns `workspace://abc123`, CoPilot can't evaluate it (e.g., checking blog thumbnail quality).

**Backend Solution:**
When preparing messages for the LLM, detect `workspace://` image references and convert them to proper image content blocks:

```python
# Before sending to LLM, scan for workspace image references
# and inject them as image content parts

# Example message transformation:
# FROM: {"role": "assistant", "content": "Generated image: workspace://abc123"}
# TO:   {"role": "assistant", "content": [
#         {"type": "text", "text": "Generated image: workspace://abc123"},
#         {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
#       ]}
```

**Where to implement:**
- In the chat stream handler before calling the LLM
- Or in a message preprocessing step
- Need to fetch image from workspace, convert to base64, add as image content

**Considerations:**
- Only do this for image MIME types (image/png, image/jpeg, etc.)
- May want a size limit (don't pass 10MB images)
- Track which images were "shown" to the AI for frontend indicator
- Cost implications - vision API calls are more expensive

**Frontend Solution:**
Show visual indicator on workspace files in chat:
- If AI saw the image: normal display
- If AI didn't see it: overlay icon saying "AI can't see this image"

Requires response metadata indicating which `workspace://` refs were passed to the model.

---

## Output Post-Processing Layer for run_block

**Problem:** Many blocks produce large outputs that:
- Consume massive context (100KB base64 image = ~133KB tokens)
- Can't fit in conversation
- Break things and cause high LLM costs

**Proposed Solution:** Instead of modifying individual blocks or `store_media_file()`, implement a centralized output processor in `run_block.py` that handles outputs before they're returned to CoPilot.

**Benefits:**
1. **Centralized** - one place to handle all output processing
2. **Future-proof** - new blocks automatically get output processing
3. **Keeps blocks pure** - they don't need to know about context constraints
4. **Handles all large outputs** - not just images

**Processing Rules:**
- Detect base64 data URIs → save to workspace, return `workspace://` reference
- Truncate very long strings (>N chars) with truncation note
- Summarize large arrays/lists (e.g., "Array with 1000 items, first 5: [...]")
- Handle nested large outputs in dicts recursively
- Cap total output size

**Implementation Location:** `run_block.py` after block execution, before returning `BlockOutputResponse`

**Example:**
```python
def _process_outputs_for_context(
    outputs: dict[str, list[Any]],
    workspace_manager: WorkspaceManager,
    max_string_length: int = 10000,
    max_array_preview: int = 5,
) -> dict[str, list[Any]]:
    """Process block outputs to prevent context bloat."""
    processed = {}
    for name, values in outputs.items():
        processed[name] = [_process_value(v, workspace_manager) for v in values]
    return processed
```
