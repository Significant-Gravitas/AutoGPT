# Clay expert avatars (draft)

The source design is [expert-design-system, edition 6](https://github.com/Significant-Gravitas/expert-design-system/tree/0e4001a).

Otto and Mina keep the managed Clay & Rock assets shipped on `dev` by #14822, including WebP/PNG size variants and fallbacks. Otto uses the reserved lavender octopus. Experts use eight category color families with lighter and darker shades, two irregular clay masses, a small face, and cream accents on the head, body, or both. Static expression choices affect new images only; activity still uses the existing status dots.

## Catalog and existing experts

`avatar_catalog.json` maps all 32 built-in SVG URLs to distinct identities: the managed Mina asset and 31 named PNGs. It also records each identity's former shared default and color. The creator no longer offers the catalog as a choice: it asks for a category and generates from there, using that category's catalog artwork as the placeholder and fallback. The frontend keeps a copy for old cached data and local previews; `avatar_catalog_test.py` checks both copies and the assets. Update both catalogs together. Seeding and API responses use the same mapping. Template reads also replace recorded v1 shared and v2 named defaults by template name. Roster seeding updates hired copies only when their URL still matches that template's recorded legacy/default avatar, using the source-template and updated-at guards; different saved catalog choices and custom URLs stay intact. Custom Notion SVGs get the warm stone default. Uploaded URLs and saved generated images stay unchanged. This needs no database migration.

The roster no longer shares one look per category. Named PNGs live in `frontend/public/experts/clay/v2/`, `v3/`, `v4/`, and `v5/`, with prompts and source notes. Final art approval remains open. Use a new version path when replacing shipped art to avoid stale browser caches. Run the existing roster seed after deployment to refresh unchanged hired defaults; no schema migration is needed.

## Live generation

Set `OPENAI_API_KEY` on the backend. `EXPERT_AVATAR_MODEL` defaults to `gpt-image-1.5`; any override must support image edits, PNG output, and transparency. See [OpenAI image generation](https://developers.openai.com/api/docs/guides/image-generation).

The request carries enums for category (8), shade (3), head shape (12), base (3), tilt (3), accent shape (6), placement (3), count per part (3), and expression (4). The creator asks the user for the category alone and rolls the rest, so regenerating shuffles the figure while the body hue stays in the category family. The optional color field remains supported for older API clients; an explicit shade takes priority. The backend builds the prompt and selects a matching head-shape reference from `avatar_references/`, rather than reusing one image for every silhouette. It requests one transparent 1024px PNG. Validation checks format, dimensions, a 5 MB byte limit, and alpha; it cannot prove that the image follows every visual rule. The user reviews the image before saving it.

A POST returns a pending job; the picker polls an owner-scoped GET every two seconds. Redis stores jobs for one hour and atomically limits each user to ten attempts per 24-hour window, at least fifteen seconds apart (`DAILY_LIMIT` and `COOLDOWN` in `avatar_jobs.py`). Failed attempts also count. Redis failures stop generation. Provider retries are off to avoid duplicate charges.

A bounded FastAPI background task runs the edit and sends the PNG through the existing scanned media upload path. Saving an expert still uses the existing create/update routes. Jobs do not survive an API process restart; after four minutes a pending job reports failure. A durable queue and cleanup of unchosen media can follow if usage warrants them. Closing the picker does not cancel a paid request.

## Validation before release

Automated tests mock the paid provider and media store. Confirm a real provider request with the deployment's credentials, scanned media storage, and an authenticated create/edit flow before release. Review the generated art at sidebar sizes and on both themes. No dynamic expression system ships in this draft.

## Topic backgrounds and accent revision

Team and Marketplace use the category hex values from the shared catalog as 24% tints mixed with white. Bodies follow the same category family, with shade, outline, and white accents providing variety. Otto alone uses mineral lavender (#B6A4C8). Team covers use the same plain topic tint. The sidebar adds a circular 1px #e3e3e3 border. Managed identities use their existing transparent cutouts on tinted surfaces.

The v3 revision changes eight built-ins to show head caps, patches, multiple spots, and split body accents. User-approved exceptions to edition 6 now allow cream on either or both masses and one to three accents per selected part. Default API requests retain one body sweep. Cream should occupy 8–25% of the figure, keep the face clear, and remain flush with the clay surface.

The v4 catalog records primary categories and category-specific variants for all 52 assignments across the 32 built-ins. Marketplace filters use the selected category’s artwork, background tint, and filter dot. All/Team use the primary category. Exact legacy identity URLs can resolve to a matching variant; arbitrary uploaded and generated URLs remain unchanged. The primary-category PNG replaces only known prior defaults during seeding.

The v5 palette separates Marketing (terracotta #C45F36), Sales (gold #D5AB24), and Support (rose #C45B88). Their 18 category variants and three creator defaults use matching hues. This is a user-approved palette change from edition 6; it keeps Otto’s lavender reserved. Dots and icons use the category color; backgrounds mix it with white.
