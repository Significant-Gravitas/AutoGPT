# Built-in Clay & Rock identities

Each of the 32 built-ins has a distinct image URL and artwork. Maria and Mina retain the managed assets from `autogpt-characters`; the other 30 use named PNGs here. Otto remains unchanged.

The expanded 26-color mineral palette adds light and dark tones across rose, rust, sand, green, teal, blue, and gray. Color is independent of job category. These are draft additions to [expert-design-system edition 6](https://github.com/Significant-Gravitas/expert-design-system/tree/0e4001a), pending final art approval.

Codex's built-in imagegen tool generated 24 new images; its model version was not exposed. `PROMPTS.json` records the generation and follow-up edit prompts. Jules, Nadia, Max, Devon, Riley, and Frankie retain one of the existing v1 designs each. New images use the v1 material/face style and an early rounded-slab study as references; Theo and Lena started from text and received a face-style edit using Jordan as reference. Images were resized to 512 × 512 with macOS `sips`, preserving alpha.

The backend includes 256px copies of 12 silhouettes in `avatar_references/`, selected by the shape enum. The palette, identity names, previous defaults, and URL map live in `avatar_catalog.json`, mirrored in the frontend catalog. Tests check parity, unique built-in URLs and file hashes, transparent PNGs, and complete shape reference coverage.

Review the whole roster at 48px as well as marketplace size. A different filename or color alone does not establish a distinct identity. Keep the saved image stable across surfaces; expressions are static generation choices, not activity states.
