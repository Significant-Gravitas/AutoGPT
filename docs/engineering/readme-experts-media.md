# README Experts media

The root README introduces Experts and Otto using the marketing site's current
artwork and an edited recording of a real product run. This note records the
sources and edits so the assets can be refreshed together.

## Sources

Retrieved from agpt.co on 2026-10-16:

- [Brand artwork](https://agpt.co/images/og/experts-v1.png): the dark AutoGPT logo.
- [Otto](https://agpt.co/images/experts/clay-v2/otto.webp),
  [Maria](https://agpt.co/images/experts/clay-v2/expert-maria.webp),
  [Max](https://agpt.co/images/experts/clay-v2/expert-max.webp), and
  [Frankie](https://agpt.co/images/experts/clay-v2/expert-frankie.webp): unchanged character artwork.
- [Geist Regular](https://agpt.co/fonts/geist-400.woff2) and
  [Semibold](https://agpt.co/fonts/geist-600.woff2): the site's typeface.
- [Team demo](https://agpt.co/videos/otto-team-demo-v1-1080.mp4) and its
  [poster](https://agpt.co/videos/otto-team-demo-v1-poster.webp): the existing
  marketing film, approximately 45 seconds long.
- [Clip's persona](https://github.com/Significant-Gravitas/skills-catalog/blob/main/experts/clip.yml)
  and [video skill](https://github.com/Significant-Gravitas/skills-catalog/blob/main/skills/code-motion-ad-15s/SKILL.md):
  the production method, adapted to a landscape README highlight.

## Outputs

Files live in `docs/content/imgs/readme/`:

- `experts-team.webp`: 1600 × 440 banner composed from the real logo, portraits,
  typeface, and the homepage's headline. No new character or product UI artwork.
- `experts-demo-poster.webp`: the existing poster with a play call to action.
  The README links it to the full film hosted on agpt.co.
- `experts-demo.mp4`: a 15-second highlight of that same film, 1920 × 1080,
  H.264/yuv420p, 30 fps, 450 frames, with AAC stereo audio and fast-start metadata.
  This is a new edit of existing footage, not a newly captured product session.

The highlight uses source intervals 2.6–4.8, 6.8–9.0, 10.8–13.0, 14.6–17.8,
23.8–26.0, 32.0–34.2, and 41.5–43.5 seconds. They show the team, brief,
delegation approval, parallel work, a deliverable, work held for review, and the
closing call to action. There are 0.2-second crossfades at output seconds 2, 4,
6, 9, 11, and 13. The original narration is replaced by an original synthesized
120 BPM instrumental with transition cues, generated at 48 kHz stereo.

The montage compresses time and omits intermediate interaction. The product UI,
example work, and original redactions remain as they appear in the public film.
Do not treat the example's approval choices as a promise that every possible
external action is always gated. The README describes what happens in this run.

Only this exact MP4 is exempted from the 500 KB pre-commit limit. Keep replacement
encodes small; the longer source film stays on the marketing site.

## Verification

Decode the complete video and audio, check codec/dimensions/frame count/duration,
measure the decoded audio peak, and inspect frames from every shot and transition.
Verify the linked poster, video fallback, and banner on GitHub at desktop and
mobile widths in light and dark themes. These checks do not replace a human
playback review of the finished film.
