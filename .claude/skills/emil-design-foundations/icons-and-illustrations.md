# Icons & Illustrations

Craft rules for iconography and illustration. Same format as the main skill: each item is a paired right/wrong call with the reason the better choice wins.

## Iconography

1. Don't reuse a 24px icon's stroke width at 16px — thin the stroke to match the reduced scale, or the small version looks heavy.
2. Snap icon coordinates to whole pixels. Half-pixel coordinates blur on 1x screens.
3. Below ~12px, 1px-stroke icons disappear — swap to the filled variant at tiny sizes.
4. Outlined vs filled for active/selected state is a trick question: either works. Consistency across the product is what matters.
5. Optical centering beats mathematical centering: a play triangle centered mathematically in a circle looks left-heavy — nudge it ~1px right.
6. Never mix two icon libraries on one surface, even similar-looking ones. Differences in stroke weight and corner radius compound across a card set.
7. Icons don't scale up: a 16px icon blown up to 48px keeps its now-spindly stroke. Redraw at 48px with a heavier stroke weight.
8. The floppy disk means nothing to anyone under 30. Ambiguous metaphors get a text label — a labelled "Save" button beats a bare icon.
9. Square stroke caps usually read as unfinished in UI contexts; default to round caps.
10. Union multipath icons into a single path with one fill. Multiple fills mean multiple things to restyle, theme, and break.
11. Match icon stroke weight to adjacent text weight. A thin-stroke icon beside bold text looks like a mistake.
12. Favicons need a simplified mark pixel-hinted for 16px. A full logo at favicon size is mush.
13. One icon, one meaning: if a chevron means both "expand accordion" and "go to next page", users reliably learn neither. Use a distinct arrow variant for navigation.
14. Icon pairs that differ only by 180° rotation (upload/download) are too subtle at a glance — add a second distinguishing signal. This matters doubly for colorblind users.
15. Alert shapes trick question: exclamation-in-circle typically reads as info/neutral, exclamation-in-triangle as warning — but conventions vary and both are used. Match your system's convention.
16. Give icon-plus-label buttons a 6–8px gap. With no breathing room they look collided.
17. Don't use the star for both "favourite" and "rating" in one product — the double meaning makes state ambiguous. Heart for favourite, star only for rating.

## Illustrations

1. Match illustration stroke weight to the icon system. 2px illustration strokes beside 1px icon strokes reads as two different products.
2. Keep perspective consistent within a scene. Objects in one-point perspective mixed with a three-quarter-view character reads as uncanny.
3. Recolor stock/community illustration packs to the UI's color tokens. Dropped in with their own palette, they look bolted on.
4. Match illustration complexity to context: a tooltip gets a spot illustration, not a full narrative scene.
5. The generic amorphous-blob-plus-character style signals "no budget or thought". A bespoke typographic treatment — or no illustration — beats generic.
6. Design negative space into the illustration itself. Forcing whitespace around it afterward looks like a clip-art drop-in.
7. Illustrations don't scale up either: at 3× the intended size, strokes go thick and detail turns coarse. Redraw for the larger dimensions.
8. Limit illustrations to ~3 colors. Six colors pulled from across the brand palette reads as indecisive; a reduced palette reads as intentional.
9. Match the illustration's shadow/light direction to the UI's shadow direction. Opposing light sources create a subliminal wrongness.
