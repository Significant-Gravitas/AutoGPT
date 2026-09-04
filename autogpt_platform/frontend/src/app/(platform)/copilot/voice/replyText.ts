/**
 * Turns a growing reply into the text that should be spoken next.
 *
 * Fenced code is dropped as it streams rather than after the fact: by the
 * time the closing fence arrives the opening lines would already have been
 * read aloud.
 */

export function createReplyTextReader() {
  let seen = "";
  let buffer = "";
  let insideFence = false;

  return { read, flush, reset };

  /** @param full - the reply so far, in whole. */
  function read(full: string): string {
    // The stream end swaps the streamed text for the server's copy, which can
    // differ in whitespace. Re-anchor silently: re-emitting would speak the
    // whole reply a second time. New turns reset the reader explicitly.
    if (!full.startsWith(seen)) {
      seen = full;
      return "";
    }
    buffer += full.slice(seen.length);
    seen = full;

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

  function flush(): string {
    const tail = insideFence ? "" : buffer;
    buffer = "";
    return tail;
  }

  function reset() {
    seen = "";
    buffer = "";
    insideFence = false;
  }
}

function isFence(line: string): boolean {
  return line.trimStart().startsWith("```");
}

/** The line is unfinished and may still turn out to be a fence. */
function couldOpenFence(partial: string): boolean {
  const trimmed = partial.trimStart();
  return "```".startsWith(trimmed) || trimmed.startsWith("```");
}
