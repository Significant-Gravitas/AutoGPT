# Clay & Rock 2.1: the whole Expert family

Follows [Clay & Rock V1](clay-rock-v1.md). Design reference: [expert-design-system](https://github.com/Significant-Gravitas/expert-design-system), Edition 9 (`expert-family-v2.1/`).

## Why

PR #14858 gave the remaining 31 Experts Codex-generated clay artwork with its own rules: cream caps and spots on heads, lighter and darker shades per person inside a category, 52 per-category recolors that swapped when a marketplace filter changed, a shifted palette (Marketing `#C45F36`, Sales gold, Support rose, Content on General's warm stone), and a generator that let users choose shade, cream placement and accent count. It also moved Maria off the approved v1.1 asset. Each of those contradicts the design system's category rules and construction rules.

## What changes

- All 34 identities (32 built-ins, Otto, General) render through the managed image renderer from versioned library paths. Otto, Maria and Mina keep their v1.1 files; the other 31 come from `expert-family-v2.1`: the v2 registry masters plus the promoted sculptural masters for Jules, Remy, Maya, Zara, Marco, Noor and Frankie, re-framed to the v2 tile framing and exported with the same size ladder (WebP 24–1024 px, PNG 24–512 px, hashes in `frontend/public/autogpt-characters/manifest.json`).
- The palette is the registry's: terracotta `#C47F5C`, ochre `#C9A35B`, sage `#A5B09A`, muted coral `#CB9182`, slate blue `#98AFC6`, olive `#AAA77A`, muted teal `#81AAA6`, charcoal `#777570`, warm stone `#B5ADA0`, Otto's lavender `#B6A4C8`. Category dots, chips, card bands and covers use it as a 24 % tint; the artwork is never tinted.
- An Expert's artwork and family color come from its identity, not from the active filter or its role text. Maria under Content is the same Maria as under Marketing. A custom appearance takes the first stored category for its surfaces and General with none.
- The retired `/experts/clay/v1`–`v5` files are removed; `avatar_catalog.json` maps every old default (Notion SVGs, roster SVGs, clay identity files and per-category variants) to its identity, and the five shared v1 category sheets to General. Reads resolve them at once; the roster seed moves hired copies that still sit on a retired default. Uploads and generated images are untouched.
- The picker offers a custom Expert the General fallback, an upload or a generated candidate, and a hired built-in its saved identity as well; never another Expert's face. The copilot raise tool no longer takes a palette; a raised Expert starts on General.
- Generation is constrained to the design system: category (fixed color), head volume, base, tilt, cream route on the lower form, expression. References are Maria, Mina and an accepted peer from the library (512 px exports, hash-checked). Output is an opaque 1024 px studio tile. `EXPERT_AVATAR_MODEL` now defaults to the pinned `gpt-image-2-2026-04-21`.

## Release dependencies and safe rollout

1. Ship the `v2.1` public files with the backend catalog; the v1.1 files stay. No database migration.
2. Run the roster seed after deployment so hired copies on retired clay defaults move to their identity; reads already resolve them, so this is cosmetic consistency, not a prerequisite.
3. Generation needs the backend `OPENAI_API_KEY` and a model that supports image edits with several references and opaque PNG output. The first live call with the pinned snapshot has not been made from this change; verify it before enabling the feature cohort.
4. Human recognition of the promoted sculptural identities has not been measured. Promoting them is a product decision recorded in the design system on 25 September 2026, not a test result.
