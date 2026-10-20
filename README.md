# AutoGPT PR #12652 screenshots

Captured on 2026-10-20 from the refreshed PR's actual React components in Storybook.
The release fixture reproduces the newest row of the public GitBook changelog index
at capture time: June 18 - July 9, with file folders, shareable chats, and Discord bot.
The account is fictional (`Alex`, `alex@example.com`).

| Image | CSS viewport | Surface |
| --- | --- | --- |
| `desktop-popup.png` | 1280 x 800 | Changelog notification |
| `mobile-popup.png` | 390 x 844 | Changelog notification |
| `desktop-menu.png` | 1280 x 800 | Classic profile menu |
| `mobile-menu.png` | 390 x 844 | Classic mobile menu |

The root images are cropped for readability. Uncropped browser captures are in
`full/`. Mobile captures retain their original 2x pixel density.

Reproduce in the PR checkout with `pnpm storybook`, then open
`Molecules/ChangelogPopup` and select `LatestRelease`, `ProfileMenu`, or `MobileMenu`.
The notification story pauses on hover so the preview remains available to inspect.
The production component still auto-dismisses after eight seconds when not hovered
or focused. These are component previews, not a deployed full-stack environment.
