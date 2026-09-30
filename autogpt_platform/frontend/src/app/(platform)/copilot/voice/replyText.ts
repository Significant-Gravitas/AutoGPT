/**
 * Turns a growing reply into the text that should be spoken next.
 *
 * Fenced code is dropped as it streams rather than after the fact: by the
 * time the closing fence arrives the opening lines would already have been
 * read aloud.
 *
 * Text is deduplicated across message ids, not only within one. A stream
 * reconnect replays the running turn from its start under fresh ids, and a
 * reader keyed on the id read the whole turn aloud again on every reconnect.
 */

interface Passage {
  id: string;
  /** Whitespace-normalised text consumed from this passage so far. */
  norm: string;
}

export function createReplyTextReader() {
  let passages: Passage[] = [];
  let current = -1;
  let seen = "";
  let buffer = "";
  let insideFence = false;

  return { read, flush, reset };

  /**
   * @param id - the assistant message the text belongs to.
   * @param full - that message's text so far, in whole.
   */
  function read(id: string, full: string): string {
    const continuing =
      current !== -1 && passages[current].id === id && full.startsWith(seen);
    // A passage that has produced nothing yet is not anchored: its first
    // text may turn out to be a replay of an earlier passage.
    if (!continuing || passages[current].norm === "") anchor(id, full);

    const passage = passages[current];
    // New text starts after whatever has been consumed — which a replay
    // that is still catching up has not reached yet, however its raw text
    // happens to be spaced.
    const start = Math.max(seen.length, rawPrefix(full, passage.norm));
    buffer += full.slice(start);
    seen = full;
    passage.norm = longer(passage.norm, normalise(full));

    let speakable = "";
    for (;;) {
      const newline = buffer.indexOf("\n");
      if (newline === -1) break;
      const line = buffer.slice(0, newline + 1);
      buffer = buffer.slice(newline + 1);
      if (isFence(line)) insideFence = !insideFence;
      else if (!insideFence) speakable += line;
    }

    if (!insideFence && buffer && !couldOpenFence(buffer)) {
      speakable += buffer;
      buffer = "";
    }
    return speakable;
  }

  /**
   * Binds `id` to the passage its text belongs to. A replay re-delivers an
   * earlier passage's text — a prefix of it while catching up, then beyond
   * it — and must continue from what was already consumed, whatever id it
   * arrives under. Text matching no passage is genuinely new output: the
   * message after a tool round.
   */
  function anchor(id: string, full: string) {
    const candidate = normalise(full);
    const slot = candidate ? findPassage(candidate) : -1;
    if (slot !== -1) {
      passages[slot].id = id;
      current = slot;
      seen = "";
      return;
    }
    if (continuingOn(id)) {
      // Same message, unrelated text: a rewrite, not new output. Re-anchor
      // without emitting — reading it again is the whole reply twice.
      if (!full.startsWith(seen)) seen = full;
      return;
    }
    passages.push({ id, norm: "" });
    current = passages.length - 1;
    seen = "";
    // Empty text cannot say yet whether it is new output or a restart; the
    // buffer may hold a sentence a replay is about to re-deliver.
    if (candidate) {
      buffer = "";
      insideFence = false;
    }
  }

  function continuingOn(id: string): boolean {
    return current !== -1 && passages[current].id === id;
  }

  /** The current passage first: a same-id rewrite is the common case. */
  function findPassage(candidate: string): number {
    const others = passages
      .map((_, index) => index)
      .filter((i) => i !== current);
    const order = current === -1 ? others : [current, ...others];
    for (const index of order) {
      const { norm } = passages[index];
      if (norm && (norm.startsWith(candidate) || candidate.startsWith(norm))) {
        return index;
      }
    }
    return -1;
  }

  function flush(): string {
    const tail = insideFence ? "" : buffer;
    buffer = "";
    return tail;
  }

  function reset() {
    passages = [];
    current = -1;
    seen = "";
    buffer = "";
    insideFence = false;
  }
}

function normalise(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

function longer(a: string, b: string): string {
  return b.length > a.length ? b : a;
}

/**
 * How much of `full` it takes to cover `norm` once whitespace is collapsed —
 * all of it when `full` has not caught up yet. The server's copy and the
 * streamed one differ in whitespace, so the raw lengths cannot be compared.
 */
function rawPrefix(full: string, norm: string): number {
  let counted = 0;
  let index = 0;
  let pendingSpace = false;
  let started = false;
  while (index < full.length && counted < norm.length) {
    if (/\s/.test(full[index])) {
      if (started) pendingSpace = true;
    } else {
      if (pendingSpace) {
        counted += 1;
        pendingSpace = false;
      }
      counted += 1;
      started = true;
    }
    index += 1;
  }
  return index;
}

function isFence(line: string): boolean {
  return line.trimStart().startsWith("```");
}

/** The line is unfinished and may still turn out to be a fence. */
function couldOpenFence(partial: string): boolean {
  const trimmed = partial.trimStart();
  return "```".startsWith(trimmed) || trimmed.startsWith("```");
}
