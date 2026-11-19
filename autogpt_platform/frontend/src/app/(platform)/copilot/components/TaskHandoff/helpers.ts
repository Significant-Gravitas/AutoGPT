const HEADLINE_MAX_CHARS = 160;
const FIRST_SENTENCE = /^(.+?[.!?])\s+(.+)$/s;

/** A result as a headline and the rest: its first sentence leads when it is
 *  short enough to read as one, otherwise the whole result is the body. */
export function splitResult(result: string) {
  const text = result.trim();
  const match = FIRST_SENTENCE.exec(text);
  if (!match || match[1].length > HEADLINE_MAX_CHARS) {
    return { headline: null, body: text };
  }
  return { headline: match[1], body: match[2] };
}

/** Where a workspace file is read from. */
export function fileSourceURL(fileID: string) {
  return `/api/proxy/api/workspace/files/${fileID}/download`;
}
