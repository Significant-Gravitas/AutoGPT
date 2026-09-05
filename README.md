Actual AutoGPT ToolChain renderer screenshots with controlled run_agent fixtures.

Before: dev 977c47767ba4874e9810475dec6abfcd5f7a5d7a
After: fix 467fa002f26595d337c8e27049cb3045951ae9ec

Both use library_agent_id b71fd24c-7623-4a73-a000-000000000000, wait_for_result 30, and an input-available streaming call. The after fixture applies a data-tool-display event resolving the name to Daily briefing through the production display projection. Vitest renders actual component DOM, served as static snapshots with production Tailwind CSS and a local Geist font for browser capture. Chat framing is a temporary fixture. These are renderer screenshots, not authenticated end-to-end workflow execution.
