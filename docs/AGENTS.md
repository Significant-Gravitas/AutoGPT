# Documentation Guidelines

## Where a Document Goes

- **Published pages** live in `docs/home/`, `docs/platform/` and `docs/integrations/`. GitBook publishes them at agpt.co/docs, and a page appears on the site once that folder's `SUMMARY.md` lists it. Put a page there only if it is written for people using or self-hosting AutoGPT.
- **Engineering notes** live in `docs/engineering/`: architecture, internal references, analytics and tracking plans, rollout plans, runbooks and maintainer guides. They are never published. Don't list one in a `SUMMARY.md`.

Every Markdown file under `docs/platform/` must be listed in `docs/platform/SUMMARY.md`. A page that isn't meant for the site belongs in `docs/engineering/`. `.github/workflows/scripts/test_docs_layout.py` checks both on every pull request.

## Block Documentation Manual Sections

When updating manual sections (`<!-- MANUAL: ... -->`) in block documentation files (e.g., `docs/integrations/basic.md`), follow these formats:

### How It Works Section

Provide a technical explanation of how the block functions:
- Describe the processing logic in 1-2 paragraphs
- Mention any validation, error handling, or edge cases
- Use code examples with backticks when helpful (e.g., `[[1, 2], [3, 4]]` becomes `[1, 2, 3, 4]`)

Example:
```markdown
<!-- MANUAL: how_it_works -->
The block iterates through each list in the input and extends a result list with all elements from each one. It processes lists in order, so `[[1, 2], [3, 4]]` becomes `[1, 2, 3, 4]`.

The block includes validation to ensure each item is actually a list. If a non-list value is encountered, the block outputs an error message instead of proceeding.
<!-- END MANUAL -->
```

### Use Case Section

Provide 3 practical use cases in this format:
- **Bold Heading**: Short one-sentence description

Example:
```markdown
<!-- MANUAL: use_case -->
**Paginated API Merging**: Combine results from multiple API pages into a single list for batch processing or display.

**Parallel Task Aggregation**: Merge outputs from parallel workflow branches that each produce a list of results.

**Multi-Source Data Collection**: Combine data collected from different sources (like multiple RSS feeds or API endpoints) into one unified list.
<!-- END MANUAL -->
```

### Style Guidelines

- Keep descriptions concise and action-oriented
- Focus on practical, real-world scenarios
- Use consistent terminology with other blocks
- Avoid overly technical jargon unless necessary
