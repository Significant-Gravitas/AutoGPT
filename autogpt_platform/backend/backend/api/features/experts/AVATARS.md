# Clay expert avatars (draft)

The source design is [expert-design-system, edition 6](https://github.com/Significant-Gravitas/expert-design-system/tree/0e4001a).

Otto, Maria, and Mina keep the managed Clay & Rock assets shipped on `dev` by #14822, including WebP/PNG size variants and fallbacks. Otto uses the reserved lavender octopus. Experts use eight mineral colors, two irregular clay masses, a small face, and a cream inlay on the lower base. Static expression choices affect new images only; activity still uses the existing status dots.

## Catalog and existing experts

`avatar_catalog.json` maps all 32 built-in SVG URLs to the managed Maria/Mina assets or eight starter PNGs. The frontend keeps a copy for old cached data and local previews; `avatar_catalog_test.py` checks both copies and the assets. Update both catalogs together. Seeding and API responses use the same mapping. Custom Notion SVGs get the warm stone default. Uploaded URLs and saved generated images stay unchanged. This needs no database migration.

The draft shares one look per category. Distinct silhouettes for the remaining built-in experts and approval of the eight new catalog images remain review tasks. New PNGs live in `frontend/public/experts/clay/v1/`; use a new version path when replacing shipped art to avoid stale browser caches.

## Live generation

Set `OPENAI_API_KEY` on the backend. `EXPERT_AVATAR_MODEL` defaults to `gpt-image-1.5`; any override must support image edits, PNG output, and transparency. See [OpenAI image generation](https://developers.openai.com/api/docs/guides/image-generation).

The authenticated creator sends only category, shape, and expression enums. The backend builds the prompt and includes `avatar_reference.png`, copied from the warm stone catalog asset. It requests one transparent 1024px PNG. Validation checks format, dimensions, a 5 MB byte limit, and alpha; it cannot prove that the image follows every visual rule. The user reviews the image before saving it.

A POST returns a pending job; the picker polls an owner-scoped GET every two seconds. Redis stores jobs for one hour and atomically limits each user to five attempts per 24-hour window, at least four minutes apart. Failed attempts also count. Redis failures stop generation. Provider retries are off to avoid duplicate charges.

A bounded FastAPI background task runs the edit and sends the PNG through the existing scanned media upload path. Saving an expert still uses the existing create/update routes. Jobs do not survive an API process restart; after four minutes a pending job reports failure. A durable queue and cleanup of unchosen media can follow if usage warrants them. Closing the picker does not cancel a paid request.

## Validation before release

Automated tests mock the paid provider and media store. Confirm a real provider request with the deployment's credentials, scanned media storage, and an authenticated create/edit flow before release. Review the generated art at sidebar sizes and on both themes. No dynamic expression system ships in this draft.
