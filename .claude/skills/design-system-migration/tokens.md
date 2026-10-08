# Foundations

Everything that lands before the first component changes.

## The registry

Kobra ships as a namespaced shadcn registry. In the project's
`components.json`:

```json
"registries": {
  "@kobra": {
    "url": "https://kobra.systems/r/{name}.json",
    "headers": {
      "Authorization": "Bearer ${KOBRA_TOKEN}"
    }
  }
}
```

The token is minted once on the account's dashboard and exported in the
shell profile, never committed. `${KOBRA_TOKEN}` is shadcn's own
environment expansion. The free components need no token.

A project that has never run shadcn needs `npx shadcn@latest init` first,
which writes `components.json`, the `cn` helper and the base theme. Choose
the `base-nova` style and the neutral base color, so what lands matches what
Kobra was drawn on. If the project has an existing `components.json` from
shadcn/ui, keep it and add the registry block.

Then, per component:

```bash
npx shadcn@latest add @kobra/button
```

Dependencies, including other Kobra components, are resolved through the
same namespace and installed first.

## Tokens

Kobra's components read semantic tokens, not colors. Map the old palette
onto them once, in the global stylesheet, and the whole library retints:

| Token                                   | What it is                                             |
| --------------------------------------- | ------------------------------------------------------ |
| `--background`, `--foreground`          | The page and its text                                  |
| `--card`, `--card-foreground`           | Raised surfaces                                        |
| `--popover`, `--popover-foreground`     | Floating surfaces                                      |
| `--primary`, `--primary-foreground`     | The one accent, and text on it                         |
| `--secondary`, `--secondary-foreground` | Quiet fills                                            |
| `--muted`, `--muted-foreground`         | Backgrounds behind content, secondary text             |
| `--accent`, `--accent-foreground`       | Hover and selected rows                                |
| `--destructive`                         | Red, and only for removal                              |
| `--border`, `--input`, `--ring`         | Hairlines, field edges, the focus ring                 |
| `--chart-1` … `--chart-5`               | Series colors                                          |
| `--radius`                              | The base corner, from which `sm`/`md`/`lg`/`xl` derive |

Old libraries usually have a larger palette (primary 50 through 900,
success, info). Collapse it: the accent becomes `--primary`, the neutrals
become background, muted and border at three steps, and semantic greens and
ambers become Badge and Alert tones rather than tokens. Write the mapping in
the ledger with the old names beside the new ones.

Use `oklch` for the values so light and dark variants share hue and chroma
and differ only in lightness.

## Dark mode

Kobra's dark theme is a `.dark` class on the root that reassigns the same
tokens. If the old library used a data attribute or a media query, add a
line in the theme switcher that sets the class as well, and keep the old
mechanism until the last old component is gone. Components never check the
theme; they read tokens.

## Fonts and radius

Set `--font-sans` and `--font-mono` once. Set `--radius` to the old
library's base corner if the team wants continuity, or to Kobra's default if
the migration is also a refresh. Decide before the first screen; changing
radius later re-reviews every screen.

## Two systems on one page

For the length of the migration, old and new components share documents.
Three things keep them from fighting:

- **Preflight is Tailwind's.** If the old library ships a CSS reset
  (Bootstrap's reboot, MUI's `CssBaseline`, Chakra's global styles), stop
  loading it. One reset per document.
- **Scope the old library's globals.** Wrap its remaining global rules in
  `@layer legacy` so utilities and Kobra's rules win on any contested
  property. If the old library injects styles at runtime, give it a
  container class and prefix its selectors with it.
- **Do not prefix Tailwind.** A prefix to avoid class collisions with the
  old library changes every Kobra class and makes the registry files wrong
  on arrival. Resolve collisions by scoping the old side instead.

Two icon sets on one surface will also fight. Kobra uses Lucide at 1.5–2px
stroke. Map the old icons to Lucide as each screen moves, and keep the two
sets on separate screens in the meantime.

## The sound layer

Install it once, early, and wrap the app in it:

```bash
npx shadcn@latest add @kobra/sound
```

Every Kobra component carries `data-slot` attributes that the layer reads to
choose a cue. Keep them as written when adapting a component. Old components
stay silent until they are replaced, which is a useful tell for what is left.

## The `cn` helper and the alias

Kobra components compose classes with `cn` from `@/lib/utils` and import
each other from `@/components/ui`. If the project's alias differs, set the
`aliases` in `components.json` before installing and shadcn rewrites the
imports on the way in.

## The first pull request

Foundations only: registry block, tokens, fonts, radius, dark mode hook,
preflight decision, sound layer, the empty ledger. No component changes.
Review it with the old screens open to confirm nothing moved. This is the
one PR in the migration that touches everything and changes nothing
visible, and it has to land first.
