# Templates

Skeletons for every file a brain skill produces. Copy the shape exactly; the fixed shape is what lets an agent reuse one reading strategy across every brain skill.

## Folder layout

```
~/brains/<person>/
  corpus/            raw, one file per piece, never edited
  notes/             one note per piece, produced in Phase 3

~/.claude/skills/<person>/      (or the project's skills dir)
  SKILL.md
  references/
    <theme-1>.md ... <theme-n>.md
    glossary.md
    article-index.md
```

The corpus and notes never sit inside the skill folder.

## Corpus file — `corpus/<slug>.md`

```md
---
url: https://paulgraham.com/ds.html
title: Do Things that Don't Scale
date: 2013-07
type: essay
---

<raw text, untouched>
```

## Note — `notes/<slug>.md`

```md
# ds — Do Things that Don't Scale

**Thesis:** Startups take off because founders push them: recruit users manually, delight them pathologically, do unscalable things.

## Claims
- **Startups don't take off by themselves.** Founders assume launch = growth; in fact the first users are hand-recruited. Example: Stripe's founders installing the product on people's laptops on the spot (the "Collison installation"). (`ds`)
- **Delight users to a degree that seems pathological.** Early on, your attention is the product's best feature; Wufoo mailed handwritten thank-you notes. (`ds`)
- ...

## Coined terms
- **Collison installation** — closing a signup by setting the product up for the user right there. (`ds`)

## Quotable
- "Startups take off because the founders make them take off." (`ds`)
```

Every bullet ends with the slug. The thesis is a claim with a verb, not a topic.

## Theme reference — `references/<theme>.md`

```md
# Pricing

One paragraph: what this theme is inside their thinking and why it matters to them.

## The core principle

- **Price on value, never on cost.** Value lives in the customer's context, not in your hours; the same product is worth different amounts to different buyers, which is why tiered pricing is legitimate. (`/pricing-your-product-the-dd-test/`)
- **When in doubt, double your price.** Beginners underprice out of fear; doubling a $19 ebook to $39 produced 75% of the revenue from 57% of the sales. (`/a-simple-rule-for-pricing-newbs-who-got-the-fear/`)

## When customers push back

- ...

## Key source articles
`/slug-one/` · `/slug-two/` · `/slug-three/`
```

40 to 100 lines. Each bullet: bold claim, their reasoning, the concrete thing, citations. When two pieces disagree, keep both with dates and say which is later.

## Glossary line — `references/glossary.md`

```md
- **schlep blindness** — The way tedious, unpleasant work hides great startup ideas from you. (`schlep`)
```

Alphabetical, one line each, every coined term.

## Article index line — `references/article-index.md`

```md
## Startups: starting and running one (`references/startups.md`)

- `ds` — **Do Things that Don't Scale** — Startups take off because founders push them: recruit users manually, delight them pathologically.
```

Grouped by theme, each group pointing at its reference file. Complete: the number of lines equals the corpus count.

## SKILL.md skeleton

```md
---
name: <person-slug>
description: The complete <Name> philosophy, distilled from all <N> <pieces> on <site>. Covers <topic>, <topic>, <topic>, <topic>, <topic>, <topic>, <topic>, and <topic>. Use when the user asks about <situation>, <situation>, <situation>, <situation>, or the ideas of <Name> / <alias> / <alias>.
---

# <Name>

This skill encodes the body of work published at [<site>](<index url>) by **<Name>**, <one clause on who they are>. It is distilled from all <N> <pieces>, from "<first title>" (<year>) through "<last title>" (<year>).

## What this is for

Use this skill to answer questions the way <Name> would: <the five or six kinds of question the corpus actually answers>.

When the user asks a broad question, answer from the **core philosophy** below plus the relevant reference file. When they ask about a specific piece or term, open the matching reference file and cite the source by slug, e.g. (`<slug>`) → `<full url>`. `references/article-index.md` lists all <N> with one-line theses; `references/glossary.md` defines every coined term.

## The one-sentence thesis

> **<A sentence they would sign.>**

Everything else is a corollary of this.

## The core philosophy (the load-bearing ideas)

1. **<Bold claim.>** <Two or three sentences of their reasoning, in their terms, with the concrete example.> (`<slug>`, `references/<theme>.md`)
2. ...
<10 to 14 entries>

## The <Name> method, end to end

1. **<Step>** — <one line>. (`<slug>`)
2. ...

## Reference files

- **`references/<theme>.md`** — <what's inside, as a comma list of the ideas>.
- ...
- **`references/glossary.md`** — <N> coined terms, each with its source.
- **`references/article-index.md`** — all <N> <pieces> with slug and one-line thesis, grouped by theme.

## How to answer

- **Write like <Name>**: <three or four observed traits, each with a phrase that shows it>.
- **Prefer the concrete**: <their numbers, their stories, their named concepts>.
- **Always cite** the source slug(s) so the user can read the original.
- If the user asks about a specific piece, check `references/article-index.md` first, then the theme file.
- These are <Name>'s views, distilled faithfully, including contested ones. Present them as their arguments, with their reasoning.
- If the corpus doesn't cover the question, say so. Do not answer in their voice from outside their work.

## Scope

This skill holds only published <Name> material. <What it does not cover, and which skill overrides it when they disagree.>
```

80 to 120 lines. If a section runs long, the overflow belongs in a reference file.
