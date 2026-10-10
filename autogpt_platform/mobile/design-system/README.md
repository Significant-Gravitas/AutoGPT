# Shared AutoGPT design assets

These files reuse the current web design system. Native projects should bundle the shared files rather than introduce a separate logo or font family.

## Assets and provenance

| File                       | Source                                                                                                                                       | Details                                                                                                                                                                                               |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `AutoGPTLogo.svg`          | [`AutoGPTLogo.tsx`](../../frontend/src/components/atoms/AutoGPTLogo/AutoGPTLogo.tsx)                                                         | Exact static render of the current component with default wordmark color `#000030`, all eight paths and five gradients preserved, `viewBox="0 0 89 40"`.                                              |
| `AutoGPTLogo.png`          | [`AutoGPTLogo.svg`](AutoGPTLogo.svg)                                                                                                         | Transparent 801 × 360 PNG rendered from the exact component SVG using the installed `sharp` 0.34.5. Suitable for the web login's 128-point width, including 3× screens. No surrounding launcher tile. |
| `AutoGPTMark.svg`          | [`frontend/public/gpt_dark_RGB.svg`](../../frontend/public/gpt_dark_RGB.svg)                                                                 | Unchanged colored standalone mark, `viewBox="0 0 2000 2000"`.                                                                                                                                         |
| `AutoGPTMark.png`          | [`frontend/src/app/apple-icon.png`](../../frontend/src/app/apple-icon.png)                                                                   | Unchanged transparent standalone mark, 180 × 180.                                                                                                                                                     |
| `fonts/Geist-Regular.ttf`  | `frontend/node_modules/geist/dist/fonts/geist-sans/Geist-Regular.ttf`                                                                        | Installed `geist` package 1.5.1; PostScript name `Geist-Regular`.                                                                                                                                     |
| `fonts/Geist-Medium.ttf`   | `frontend/node_modules/geist/dist/fonts/geist-sans/Geist-Medium.ttf`                                                                         | Installed `geist` package 1.5.1; PostScript name `Geist-Medium`.                                                                                                                                      |
| `fonts/Poppins-Medium.ttf` | [Official Google Fonts source](https://github.com/google/fonts/blob/7dc16b7de42db624b902f2292b68ed9e489e5053/ofl/poppins/Poppins-Medium.ttf) | Unchanged Poppins Medium, weight 500; PostScript name `Poppins-Medium`.                                                                                                                               |

Geist and Poppins are distributed under the SIL Open Font License 1.1. Their original license texts are included as `fonts/Geist-OFL.txt` and `fonts/Poppins-OFL.txt`. AutoGPT branding retains the licensing of its source repository assets. `SHA256SUMS` records the bundled asset bytes. Fonts and standalone marks are copied unchanged; the full logo is rendered directly from the component, rather than the older public PNG.

The web's exact full-logo vector geometry is in [`AutoGPTLogo.tsx`](../../frontend/src/components/atoms/AutoGPTLogo/AutoGPTLogo.tsx), with `viewBox="0 0 89 40"`. The colored mark uses `#000030`, `#4285F4`, `#9900FF`, and `#669CF6`. The white logo variant is for dark marketing surfaces; the dark square notification/launcher image is not the in-content login logo.

## Native screen tokens

Use the resolved custom palette in [`colors.ts`](../../frontend/src/components/styles/colors.ts), rather than Tailwind's default zinc palette:

| Role                      | Web token  | Value     |
| ------------------------- | ---------- | --------- |
| Primary action            | zinc-800   | `#3E3E43` |
| Primary pressed/hover     | zinc-900   | `#2C2C30` |
| Secondary action          | white      | `#FEFEFE` |
| Subtle surface            | zinc-50    | `#F9F9FA` |
| Border                    | zinc-200   | `#DADADC` |
| Secondary text            | zinc-600   | `#68686F` |
| Muted text                | zinc-500   | `#83838C` |
| Default text              | black      | `#141414` |
| White surface/action text | white      | `#FEFEFE` |
| Input focus               | purple-400 | `#925CF7` |

Typography comes from [`Text/helpers.ts`](../../frontend/src/components/atoms/Text/helpers.ts) and [`fonts.ts`](../../frontend/src/components/styles/fonts.ts):

- Authentication heading (`h3`): Poppins Medium 28 points, 40-point line height, approximately −0.21-point letter spacing. The login page overrides the default text color to slate-950 (`#020617`).
- Body: Geist Regular 14/22 points. Larger supporting text uses 16/26 points.
- Button labels: Geist Medium 14 points.
- Small supporting text: Geist Regular 12/18 points.

[`Button/helpers.ts`](../../frontend/src/components/atoms/Button/helpers.ts) defines a 46-point large action, pill radius, 16-point horizontal padding, and 8-point icon/label gap. Secondary actions use the same shape, a white surface, zinc-800 text, and a zinc-200 border. Their pressed surface is zinc-50 with a zinc-300 border. Maintain native text scaling and allow controls to grow when needed.

[`AuthSplitLayout.tsx`](../../frontend/src/components/auth/AuthSplitLayout/AuthSplitLayout.tsx) uses a white authentication surface, 24-point horizontal insets, a 416-point maximum content width, and a centered 128-point-wide full logo with 40 points below it. The mobile web login adds a very faint 10%-opacity blue/lavender Aurora background. The form heading is left aligned. The app's initial web theme is light.

[`Input.tsx`](../../frontend/src/components/atoms/Input/Input.tsx) uses a 46-point field, 12-point radius, 1-point zinc-200 border, and 16-point horizontal padding. The standalone [`Card`](../../frontend/src/components/atoms/Card/Card.tsx) uses a 16-point radius and 24-point padding. Spacing is based on a 4-point grid; common section gaps are 16, 24, and 32 points.

Current web icons use Hugeicons stroke-rounded paths through [`Icon.tsx`](../../frontend/src/components/atoms/Icon/Icon.tsx), with stroke width 2. Large-button icons are 18 points; small-button icons are 16 points. Preserve native accessibility hit targets independently of the visible icon size.
