Actual AutoGPT ToolChain renderer screenshots with controlled run_block fixtures.

Before: PR head dc3bdce02e5d0098cb6b7352a2003130932a6dda
After: 3bd75b6b7db17d923f67ec5419b1eef6e0604ea9

Both fixtures use block_id db7d8f02-2f44-4c55-ab7a-eae0941f0c30 and the same completed block output named FillTextTemplateBlock. The after fixture also applies the production data-tool-display projection for the resolved name. The chain and output card are expanded through their actual React click handlers. Vitest renders the production component DOM, served as static snapshots with production Tailwind CSS and local Geist fonts for browser capture. Chat framing is a controlled fixture. These screenshots do not represent authenticated end-to-end block execution.
