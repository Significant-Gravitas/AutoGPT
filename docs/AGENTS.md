# Documentation Guidelines

## Where a Document Goes

- **Published pages** live in `docs/home/`, `docs/platform/` and `docs/integrations/`. GitBook publishes them at agpt.co/docs, and a page appears on the site once that folder's `SUMMARY.md` lists it. Put a page there only if it is written for people outside the team: people using, self-hosting or contributing to AutoGPT.
- **Engineering notes** live in `docs/engineering/`: the team's working documents, such as analytics and tracking plans, rollout plans, runbooks, internal references, and design and architecture notes. They are not on the docs site. Don't list one in a `SUMMARY.md`.

`docs/engineering/` is not private. The repository is public and AutoPilot's documentation search indexes the folder, so keep secrets and confidential material out of it, as with any file in the repository.

When unsure, write the document in `docs/engineering/`. Listing a page in a `SUMMARY.md` publishes it, so do that only for a page meant for the site.

The site follows `master`. `.github/workflows/docs-gitbook-sync.yml` copies `master`'s `docs/` to the `gitbook` branch, and GitBook publishes that branch. Change docs through a pull request to `dev`. Never commit to `gitbook`, open a pull request against it, or edit pages in GitBook: the next copy overwrites anything that isn't on `master`.

Every Markdown page under `docs/platform/` must be listed in `docs/platform/SUMMARY.md`: every `.md` file other than `SUMMARY.md` itself and GitBook's own files under a `.gitbook/` directory. A `SUMMARY.md` may list only pages inside its own folder. `.github/workflows/scripts/test_docs_layout.py` checks those two things on every pull request. It cannot tell whether a listed page is meant for the public; that is for the author and the reviewer to judge.

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
