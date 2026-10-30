---
name: emil-borrow-a-brain
description: Turn a person's or company's publicly available body of work (essays, blog, talk transcripts, newsletter, docs) into a skill that answers questions the way they would, with the source attached to every claim. Collects the complete corpus, extracts each piece into a thesis and sourced claims, clusters them into theme files, a glossary, and a full index, then writes a SKILL.md with the one-sentence thesis, the load-bearing ideas, and how to answer in their voice. Use when the user says "make a skill out of X's essays", "build a knowledge bank from someone's writing", "I want my agent to think like X", or asks where to store and how to structure knowledge distilled from someone else's public work.
disable-model-invocation: true
---

# Borrow a Brain

AI answers any question, but you can't narrow where the answer comes from. A brain skill fixes that: it answers from one person's published work only, and every claim carries the piece it came from. The output is a skill, not a document. A skill is what's always in your agent's context, so it has to be a distillation, never a dump.

The whole method is five phases in strict order: **scope, collect, extract, cluster, write, then test.** The common failure is skipping to writing SKILL.md from what you already know about the person. That produces a generic essay with their name on it. Every sentence in the finished skill traces back to a file in the corpus, or it isn't in the skill.

## Phase 1 — Scope

Settle these before fetching anything. Ask the user only for what you can't infer.

1. **Who, and which corpus.** One person or company, and their primary published body of work. Pick the source they actually write in: essays site, blog archive, talk transcripts, newsletter archive, docs. A corpus is one place, taken whole. Never sample it and never pad it with what you remember about them. The reason the skill is trustworthy is that its boundary is exact.
2. **Source type decides how you collect.**

   What is the corpus?
   ├── Essays or blog with an index page → enumerate the index (or sitemap / RSS), fetch every URL
   ├── Talks or videos → transcripts only; no transcript, no source (WWDC and most conference sites publish them)
   ├── Podcast → episode transcripts from the show's site; skip episodes without one
   ├── Newsletter → the public archive page
   ├── Social posts → only an export or archive the user provides; never scrape a timeline
   └── Books or paywalled content → out. Not public, and reproducing it isn't distillation.

3. **Citation scheme, decided once.** Derive it from the URL shape and write it in the SKILL.md intro. Essays at `site.com/<slug>.html` cite as (`slug`). Paths like `site.com/<slug>/` cite as (`/slug/`). Sessions cite by number. Every claim in every file uses this one scheme, so the reader can open any source in one step.
4. **Is a skill the right container?**

   Who reads the result?
   ├── You and your agents → a skill. This is the default.
   ├── Same, but the corpus is over 500 pieces or changes weekly → still a skill for the distilled layer, plus a search tool (MCP or a script) over the raw corpus for lookups
   └── People without an agent → an app on top of the same corpus and the same distilled files. Build the skill first; the app reads what the skill reads.

## Phase 2 — Collect

- **Enumerate first, then count.** Pull every URL from the index and write the count down. That number goes into the skill's description ("distilled from all 232 essays"), so it has to be true, and it's how you verify the collection later.
- **One markdown file per piece**, named by slug, with frontmatter for url, title, date, and type (skeleton in `templates.md`). Body is the raw text. These files are never edited. When your understanding improves, you re-distill from them; you don't touch them.
- **The corpus lives outside the skill folder**, next to it, e.g. `~/brains/<person>/corpus/`. The skill folder gets installed and loaded into context. A raw corpus inside it either ships to every install or gets globbed into the agent's attention, which is exactly what a skill exists to prevent.
- **Verify before extracting.** File count equals index count. Any fetch that failed gets retried and listed; a piece silently missing means the skill will one day confidently say the person never wrote about something they did.
- Fetch sequentially with a short pause. You are reading a site, not load-testing it.

## Phase 3 — Extract, one piece at a time

Work in batches of 10 to 15 sources per pass and write one note per source to `notes/<slug>.md` (skeleton in `templates.md`). A note holds:

- **A one-line thesis that is a claim, not a topic.** "Startups die of demoralization, not events" is a thesis. "About startup failure" is a table of contents. The thesis line is what the article index is built from, and a topic there tells the agent nothing about what the person argued.
- **Three to eight claims**, each with the reasoning the author gives and the concrete thing attached to it: the number, the story, the example, the coined phrase. A claim without its why gets applied blindly; a claim without its example reads as generic advice.
- **Coined terms** the piece introduces, defined in one line each. A term is coined if they named it or use an existing word in their own sense.
- **Short quotable phrases**, 25 words or fewer. Never a paragraph. The skill distills; it does not reproduce.
- **Every bullet ends with the slug.** No exceptions. A claim without a source can't be checked, and unsourced claims are where the person's views quietly blur into the model's.

Extract positions, not summaries. Cut the throat-clearing and keep the argument. If the piece contradicts an earlier one, note the date; you'll need it in the next phase.

## Phase 4 — Cluster

After every note exists, and not before:

- **Group the claims into 8 to 12 themes**, named by the question a reader brings ("pricing", "fundraising", "writing"), never by the author's chronology or by the site's own categories. Fewer than 8 and the files get too long to load for one question; more than 12 and the agent can't pick.
- **One reference file per theme**, 40 to 100 lines, shaped as: a short intro paragraph, H2 sections, and bullets of the form **bold claim**, reasoning, concrete example, citations. End with a "Key source articles" line. The bullets are the unit the agent quotes from, so each one has to stand alone.
- **Contradictions stay visible.** When the person changed their mind, keep both positions with dates and say which is later. Never average them into a position nobody held.
- **`glossary.md`**: every coined term, alphabetical, one line each, with its slug. This is the file that lets the skill answer "what did they mean by X" without opening a theme file.
- **`article-index.md`**: every source, grouped by theme, as `slug`, **title**, one-line thesis. Complete, not selected. If the count here doesn't match Phase 2, a note is missing.

## Phase 5 — Write SKILL.md

The shape is fixed. Use the skeleton in `templates.md` and keep the section order; the agent reading it later learns where things are once and reuses that across every brain skill you build.

1. **Description line**: "The complete [name] philosophy, distilled from all [N] [pieces] on [site]. Covers [8 to 10 topics]. Use when the user asks about [situations], or the ideas of [name / aliases]." This is the line that triggers the skill, so it lists what people type: situations and aliases, not adjectives.
2. **Intro paragraph**: who they are in one sentence, which sources, how many, the date span, and the citation scheme.
3. **The one-sentence thesis**, as a blockquote, followed by "Everything else is a corollary of this." It has to be a sentence they would sign. If you can't write it, you haven't finished reading; go back to the notes.
4. **The core philosophy**: 10 to 14 numbered load-bearing ideas. Each is a bold claim, two or three sentences of their reasoning in their terms, and citations. The test: the agent should answer 80% of broad questions from this list alone, without opening a reference. SKILL.md is always in context; references are on demand.
5. **The method, end to end**: their advice as an ordered playbook, in the order someone would live it.
6. **Reference files**: one line per file saying what's inside, so the agent picks the right one without opening three.
7. **How to answer**: three or four voice traits you actually observed, with a phrase that shows each; prefer the concrete (their numbers, their stories); always cite; present contested views as theirs, with their reasoning, and do not soften them.
8. **Scope**: what the corpus does not cover, and which other skill overrides this one when they disagree (usually the user's own decisions).

SKILL.md stays between 80 and 120 lines. Everything longer belongs in a reference.

## Phase 6 — Test

Install the skill, open a fresh session, and ask:

- Three broad questions you know the answer to from the corpus. Pass: it leads with their position and cites the right slugs.
- One specific piece by name. Pass: it finds it through the article index and pulls detail from the theme file.
- One question the corpus doesn't cover. Pass: it says the person didn't write about this, instead of producing generic advice in their voice. This is the failure users can't detect themselves.
- One contested position. Pass: it presents the position as theirs, with their reasoning, unsoftened.

Fix a failure by adding to a reference or the core list, never by adding hedges to SKILL.md. Re-run after every fix.

## Keeping it alive

When the person publishes, add the piece to the corpus, write its note, add it to the index, and re-read the core list to see if anything shifted. Update the count in the description. A brain skill with a stale count is a skill that lies about its own boundary.
