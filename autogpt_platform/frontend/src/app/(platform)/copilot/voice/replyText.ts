/**
 * Turns a growing reply into the text that should be spoken next.
 *
 * Fenced code is dropped as it streams rather than after the fact: by the
 * time the closing fence arrives the opening lines would already have been
 * read aloud.
 */

export function createReplyTextReader() {
  let messageId = "";
  let seen = "";
  let buffer = "";
  let insideFence = false;

  return { read, flush, reset };

  /**
   * @param id - the assistant message the text belongs to. A tool round
   *   starts a new message, and its text is new output, not a rewrite.
   * @param full - that message's text so far, in whole.
   */
  function read(id: string, full: string): string {
    if (id !== messageId) {
      messageId = id;
      seen = "";
      buffer = "";
      insideFence = false;
    } else if (!full.startsWith(seen)) {
      // Same message, different text: the stream end swapped the streamed
      // copy for the server's, which differs in whitespace. Re-anchor
      // silently — emitting again would read the whole reply twice.
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
    messageId = "";
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
