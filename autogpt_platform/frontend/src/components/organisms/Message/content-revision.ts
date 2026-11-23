import { useState } from "react";

/**
 * A number that changes when `text` stops being a continuation of the text
 * before it, for keying a Streamdown instance.
 *
 * Streamdown memoises each block on its class and source position only, so a
 * text replaced without moving any block (a stored reply that reads "!" where
 * the stream said ".") keeps the old words on screen. Text that only grows,
 * as a stream writes it, keeps the revision, so nothing remounts. The update
 * runs during render, before the children do, so it costs no extra pass of
 * the markdown.
 */
export function useContentRevision(text: string): number {
  const [seen, setSeen] = useState({ text, revision: 0 });
  if (text === seen.text) return seen.revision;
  const revision = text.startsWith(seen.text)
    ? seen.revision
    : seen.revision + 1;
  setSeen({ text, revision });
  return revision;
}
