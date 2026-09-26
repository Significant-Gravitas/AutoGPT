# Expert avatars: the managed Clay & Rock library

The source of truth is the [expert-design-system](https://github.com/Significant-Gravitas/expert-design-system) (Edition 9): its category rules, generation standard and engineering handoff. This document says how the application applies them; it does not restate them.

## One identity per Expert

`avatar_catalog.json` (identical copy beside the frontend `ExpertAvatar` molecule; `avatar_catalog_test.py` keeps them equal and checks every file) lists 34 managed identities: the 32 built-in Experts, Otto and the General fallback. Each identity has an asset ID, a revision, a versioned library path and a `visual_category`, the palette family its material belongs to. The saved `avatarUrl` (`/autogpt-characters/<library>/<asset id>/neutral/128.webp`) is the binding; the renderer derives every size from it.

Otto, Maria and Mina stay on the `v1.1` pack shipped by #14822. The other 31 identities come from `expert-family-v2.1` in the design repository: the v2 baseline registry masters plus the promoted sculptural masters for Jules, Remy, Maya, Zara, Marco, Noor and Frankie, re-framed to the same tile framing and exported with the same size ladder (`frontend/public/autogpt-characters/v2.1/`, WebP 24–1024 px, PNG 24–512 px; `manifest.json` next to it records every hash). Adding, replacing or promoting artwork means a new versioned library path; released paths are never overwritten.

The identity never changes with the Expert's name, role, skills, category or the marketplace filter. Filtering Maria under Content shows the same Maria as under Marketing. A category edit changes the label and the filter, not the artwork. There are no per-category variants, no per-person shades and no runtime hue, saturation or tint filters; the palette hex only tints the surfaces around the tile (card band, cover, chip glyphs), and the tile itself stays opaque on light and dark surfaces.

## Palette

`palette` carries the registry's material anchors: Marketing terracotta `#C47F5C`, Sales ochre `#C9A35B`, Finance sage `#A5B09A`, Support muted coral `#CB9182`, Operations slate blue `#98AFC6`, Research olive `#AAA77A`, Content muted teal `#81AAA6`, Development charcoal `#777570`, General warm stone `#B5ADA0`, and Otto's reserved mineral lavender `#B6A4C8`, which no specialist may use. The eight stored category values are unchanged; `general` and `otto` exist only in the appearance registry.

## Legacy defaults and seeding

`legacy` maps every default that no longer ships to the identity it stood for: the old Notion SVGs and `/experts/*.svg`, and the retired `/experts/clay/v1`–`v5` PNGs from #14858 (those files are gone; the five shared v1 category sheets were never an identity and resolve to General). `resolve_avatar_url` applies the map on every API read, so no database migration is needed. `resolve_builtin_avatar_url` is the name-aware version used for templates and for the roster seed's hired-copy backfill: only a URL that was once that template's own default moves; uploads, generated images and the General fallback stay exactly as saved. An unknown `/avatars/notion/...` pick becomes General. An image that fails to load falls back to PNG, then to the Expert's monogram, never to Otto or another named face.

## Custom Experts

A raised Expert starts on the General fallback. From the Team page its owner can keep that look, upload a picture (PNG/JPEG/WebP, 5 MB, scanned as before) or generate a candidate. A hired built-in can keep its saved identity or change appearance the same way; the picker never offers another Expert's face.

## Generation

`ExpertAvatarRequest` only exposes what the design system lets a new identity vary: the category (which fixes the one material color; `general` when none is chosen), the head volume (12 sculptural shapes), the base, the tilt, the cream route (seven lower-form routes) and a static expression. There is no shade, color, accent placement or accent count.

The prompt in `avatar_generation.py` is the construction specification and the 25 September material policy: two touching masses, one category hue, low-sheen clay, cream `#EAE2D5` only on the lower form with a recessed groove, the gentle charcoal face placed optically, no anatomical readings, no purple, opaque warm studio tile. References follow the generation standard in order: Maria (finish, light, face, cream material), Mina (silhouette range, cream placement) and an accepted peer of the requested family as the color reference. They are the library's own 512 px exports, copied into `avatar_references/` with their hashes in `manifest.json`; a hash mismatch refuses to generate. Content has no accepted peer yet and takes its color from the hex anchor alone.

`EXPERT_AVATAR_MODEL` defaults to the design system's pinned snapshot `gpt-image-2-2026-04-21` and needs the backend `OPENAI_API_KEY`; any override must support image edits with several reference images and opaque PNG output. The request asks for one 1024 px opaque PNG at high quality; validation checks format, size, the 5 MB limit, that the tile is opaque and that it carries artwork. Validation cannot prove a render follows every visual rule, which is why the user reviews the candidate and chooses **Use this avatar** before anything is saved. Jobs, limits (five per user per 24 hours, four minutes apart, failures count), Redis storage and the scanned media upload path are unchanged from #14858. A generated candidate is that Expert's own appearance; it is not added to the managed library.

## Before release

Confirm a real provider request with the deployment's credentials and the pinned model, then an authenticated create/edit flow; automated tests mock the provider. Run the roster seed after deployment so hired copies still sitting on a retired clay default move to their identity (reads already resolve them). Human recognition of the promoted sculptural identities has not been measured; the promotion is a product decision recorded in the design system on 25 September 2026.
