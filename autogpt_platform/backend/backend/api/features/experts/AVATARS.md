# Clay expert avatars (draft)

The source design is [expert-design-system, edition 6](https://github.com/Significant-Gravitas/expert-design-system/tree/0e4001a).

Otto, Maria, and Mina keep the managed Clay & Rock assets shipped on `dev` by #14822, including WebP/PNG size variants and fallbacks. Otto uses the reserved lavender octopus. Experts use an expanded palette of 26 mineral colors, two irregular clay masses, a small face, and a cream inlay on the lower base. Static expression choices affect new images only; activity still uses the existing status dots.

## Catalog and existing experts

`avatar_catalog.json` maps all 32 built-in SVG URLs to distinct identities: the managed Maria/Mina assets and 30 named PNGs. It also records each identity's former shared default and color. The creator keeps the eight starter catalog choices. The frontend keeps a copy for old cached data and local previews; `avatar_catalog_test.py` checks both copies and the assets. Update both catalogs together. Seeding and API responses use the same mapping. Template reads also replace recorded v1 shared defaults by template name. Roster seeding updates hired copies only when their URL still matches that template's recorded legacy/default avatar, using the source-template and updated-at guards; different saved catalog choices and custom URLs stay intact. Custom Notion SVGs get the warm stone default. Uploaded URLs and saved generated images stay unchanged. This needs no database migration.

The roster no longer shares one look per category. New named PNGs live in `frontend/public/experts/clay/v2/`, with prompts and source notes. Final art approval remains open. Use a new version path when replacing shipped art to avoid stale browser caches. Run the existing roster seed after deployment to refresh unchanged hired defaults; no schema migration is needed.

## Live generation

Set `OPENAI_API_KEY` on the backend. `EXPERT_AVATAR_MODEL` defaults to `gpt-image-1.5`; any override must support image edits, PNG output, and transparency. See [OpenAI image generation](https://developers.openai.com/api/docs/guides/image-generation).

The authenticated creator sends enums for color (26), head shape (12), base (3), tilt (3), cream path (3), and expression (4). Color is independent of category; old requests without a color retain their category default. The backend builds the prompt and selects a matching head-shape reference from `avatar_references/`, rather than reusing one image for every silhouette. It requests one transparent 1024px PNG. Validation checks format, dimensions, a 5 MB byte limit, and alpha; it cannot prove that the image follows every visual rule. The user reviews the image before saving it.

A POST returns a pending job; the picker polls an owner-scoped GET every two seconds. Redis stores jobs for one hour and atomically limits each user to five attempts per 24-hour window, at least four minutes apart. Failed attempts also count. Redis failures stop generation. Provider retries are off to avoid duplicate charges.

A bounded FastAPI background task runs the edit and sends the PNG through the existing scanned media upload path. Saving an expert still uses the existing create/update routes. Jobs do not survive an API process restart; after four minutes a pending job reports failure. A durable queue and cleanup of unchosen media can follow if usage warrants them. Closing the picker does not cancel a paid request.

## Validation before release

Automated tests mock the paid provider and media store. Confirm a real provider request with the deployment's credentials, scanned media storage, and an authenticated create/edit flow before release. Review the generated art at sidebar sizes and on both themes. No dynamic expression system ships in this draft.
