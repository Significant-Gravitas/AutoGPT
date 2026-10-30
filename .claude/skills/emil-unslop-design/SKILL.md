---
name: emil-unslop-design
description: "Strip the visual tells that mark an interface as AI-generated — indigo gradients, glassmorphism, emoji feature grids, the centered-hero sameness — then put a point of view back in. Use when a screen \"looks AI-generated\", \"looks vibe-coded\", \"looks like every other landing page\", or generic; when reviewing agent-built UI before it ships; or as a pass after generating any marketing page, dashboard, or app screen. Triggers on: slop, AI slop, design slop, de-slop, unslop, looks AI generated, vibe coded, generic, template look, cookie cutter, purple gradient, indigo, violet, gradient text, bg-clip-text, glassmorphism, backdrop-blur, glow, neon, gradient orb, blob, bento grid, feature grid, icon cards, emoji icons, sparkles, ✨, 🚀, centered hero, announcement badge, pill badge, stats row, testimonial grid, logo wall, dark mode default, Inter, tracking-tight, font-black, rounded-2xl, hover scale, fade-in on scroll, \"make it look less AI\", \"make it unique\", \"give it personality\", distinctive design."
---

# Unslop Design

Nobody can point at the one element that makes a screen look AI-generated. They recognize the accumulation: indigo gradient, glass cards, emoji icons, centered hero, three feature boxes — each choice defensible alone, and together a watermark. The cause is mechanical: a model reproduces the statistically most common pattern in its training data, and it does so for every decision on the page at once. Fifty median choices don't average out to a median design. They compound into a recognizable one.

This skill is a pass over any generated interface, in two halves. First strip the tells. Then put a point of view back in — this half is not optional, because a screen scrubbed of tells but containing no opinionated choice is still generic, just more quietly.

## The pass

1. Squint test first: could this screen belong to any of ten thousand other products? Find the elements that tie it to *this* product specifically. If the answer is "the logo", everything else is up for review.
2. Subtract before you restyle (see "Subtract first"). Half the tells on a generated screen are elements that shouldn't exist at all, and there's no point picking a better color for something you're about to delete.
3. Sweep what's left in order: color, then typography, then layout, then decoration, then iconography, then the copy on the screen, then motion. Color goes first because it's the loudest tell — a page can survive a common layout, but not indigo-to-purple.
4. Put a point of view back in (see "Looking designed by someone").
5. Final read: "What here would make a designer suspect a model built this?" Fix it. Repeat until the answer is nothing.

## Subtract first

Agents have an additive bias, and it's the same one you've seen in code: the belt-and-suspenders try-catch, the re-implemented utility, the plan with three fallback steps nobody asked for. In UI it shows up as extra everything — a caption under a heading that restates the heading, an icon in front of every label, a divider between sections that spacing already separates, a badge on the card, a second CTA, a helper line under the input, a border on a surface that also has a shadow. The result is a screen that looks better than most engineers would build by hand and is still bad, because every element is competing and none of it was chosen.

So before restyling anything, walk every element on the screen — every icon, line, badge, caption, container, and word — and ask one question: **what breaks if this is gone?** If the answer is "nothing", delete it. Not "it adds visual interest", not "it fills the space" — those are the reasons the model put it there. Run the walk again after the first round; deletions expose each other, because an element that survived round one was often only earning its keep by balancing something you just removed.

This is the highest-leverage step in the whole pass. Deletion is free, it can't introduce new tells, and every element that survives gets hierarchy for nothing — the fewer things on the screen, the more each one reads as deliberate.

## Color

### Indigo is not the brand color

`bg-indigo-500`, violet, and the blue-purple family are the single strongest tell — models absorbed them from years of Tailwind examples, and now every generated UI on earth defaults to them. Purple is not banned; *unchosen* purple is. If the palette wasn't derived from the product's domain, brand, or content, derive one (the color skill covers building it in OKLCH). A finance tool, a farming app, and an AI startup should not share a hue by accident.

### One gradient per screen, chosen — or none

The purple-to-cyan wash, the gradient hero background, the gradient border, the gradient text all on one page is decoration standing in for a decision. Gradients also band badly when they're this casual (the surfaces skill covers easing them). Keep at most one, give it a job, and build it from the palette you actually chose.

### Dark-and-neon is a costume

Dark-by-default with glowing borders and neon accents is the model's idea of "premium tech". Dark mode is a real feature with real rules — shadows stop working, surfaces need their own logic (see surfaces) — not an aesthetic shortcut. Choose the default theme from where and when the product is used, not from vibes.

## Typography

### No gradient text

`bg-clip-text text-transparent bg-gradient-to-r` on the headline is the typographic equivalent of the purple gradient, and it costs you contrast and legibility to boot. Headlines get a solid color with real contrast.

### The default stack is a shrug

Inter at `font-black tracking-tight`, centered, is what a model writes when nobody chose anything. Inter is a fine typeface — as a decision, paired and sized deliberately, not as an absence of one. Make one distinctive typographic choice per project (the typography skill covers selection and pairing); it does more to de-slop a page than any other single change.

## Layout

### The hero every model builds

Pill badge ("✨ Announcing v2.0"), centered headline, one-line subhead, primary button plus "Learn more", gradient orb behind it. Every element here is a statistical default, which is why the assembly reads as generated even when each piece is executed cleanly. Break the template where the content justifies it: lead with the actual product (a real screenshot, a live demo) instead of an abstraction, left-align if the copy runs long, cut the badge unless there is genuinely an announcement.

### Three cards, an icon, a heading, two lines

The icon-title-description grid is the model's answer to "features section" regardless of what the features are. The tell isn't the grid — it's that the layout was chosen before the content. Let the features dictate the form: one feature that matters gets a full-width treatment with a real screenshot; six small ones get a compact list; unequal features get unequal space. Symmetry that the content didn't ask for is the giveaway.

### Bento is the new three-card grid

The bento grid became the "I know the three-card grid is slop" alternative, which made it slop with better kerning. Same rule: if the cells were sized before anyone knew what goes in them, it's decoration.

## Decoration

### Glassmorphism needs glass

`backdrop-blur` with a translucent fill makes sense when there is something meaningful behind the surface — a photo, a video, layered content. Floating a frosted card over a flat dark background is applying the effect without its reason, which is exactly how a model uses it.

### Nothing floats behind the hero

Blurred gradient orbs and animated blobs exist to fill space the design didn't know what to do with. Delete them. If the background feels empty afterward, the foreground content is too weak — fix that instead.

### One depth cue per surface

Border, shadow, ring, *and* gradient on the same card is a model hedging. Pick the depth system — shadows or borders, per the surfaces skill — and apply it consistently. And stop `rounded-2xl` from being the answer to every radius: radii come from a scale, and nested radii are derived, not repeated.

## Iconography

### Emoji are not icons

🚀⚡🔒✨ as feature markers is the fastest way to make a page look generated. Emoji render differently on every platform, can't take your palette, and read as placeholder because that's what they are. Use a real icon set — one set, one stroke width — or none; a features list without icons is more honest than one with rocket ships.

### Sparkles don't mean AI

The ✨ sparkle badge on every AI feature is already a cliché. If a feature being AI-powered matters to the user, the copy says what it does; if it doesn't matter, the badge is noise either way.

## Fabricated proof

"Trusted by 10,000+ developers" on an unlaunched product, a testimonial grid of stock-avatar quotes, a logo wall of companies that never said yes — models generate social proof as a *layout*, filled with invented content. This is the design equivalent of unslop-writing's vague attribution, except worse: it's a lie in the interface. Real numbers, real quotes with real names, or cut the section entirely. An empty spot where fake proof used to be is a stronger signal than the fake proof.

## The copy on the screen

"Build faster. Ship smarter." — a headline any of ten thousand products could run is a visual element that says nothing, however good it looks set in type. The test: could a competitor paste this headline unchanged? Then it's slop. Run interface copy through the unslop-writing skill; here the rule is just that copy is part of the design, and generic copy makes honest visuals read as generated anyway.

## Motion

### Not every section fades in

`fade-in-up` on scroll for every block, staggered children everywhere, `hover:scale-105` on cards, pulsing blobs — motion applied by default instead of by decision. The animations skill has the actual decision tree; the de-slop rule is simpler: scroll-triggered entrances on static marketing content are almost always noise, and a card that isn't clickable has no business reacting to hover.

## Looking designed by someone

Stripping tells leaves a clean, mute screen. What separates designed from generated is evidence that somebody made choices only this product would make:

- **Derive the palette from something real.** The product's domain, an existing brand artifact, the content's own colors. A palette with a reason survives the squint test; a tasteful-but-arbitrary one doesn't.
- **One distinctive type decision.** A display face with a point of view, an unusual size contrast, a deliberate pairing. One is enough; three is a costume party.
- **Show the actual product.** A real screenshot, real data, a working embed. Abstract illustration is what models reach for because they've never seen your product. You have.
- **Let content break symmetry.** An odd number of features, one oversized item, a section that's just a sentence. Asymmetry the content earns is the cheapest authenticity there is.
- **Repeat a signature.** One distinctive element — a corner treatment, a rule weight, a motif — used consistently beats ten borrowed ones used once. Consistency of an odd choice is what a point of view looks like from the outside.
- **Restraint is a feature.** The confident version of a screen usually has fewer effects than the generated one, not more. Deleting decoration is de-slopping; adding "personality effects" on top is re-slopping with extra steps.

## Final audit

Squint one last time and ask two questions. "What here would make a designer think a model built this?" — you're checking for survivors: a leftover glow, an emoji that snuck back in, a gradient with no job. And "name the one choice a competitor couldn't copy-paste" — if nothing on the screen answers, the pass isn't done: go back to "Looking designed by someone".
